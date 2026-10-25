"""Faster Whisper STT engine with Enter-to-start/stop recording."""

import asyncio
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..base import STTBase

SAMPLE_RATE = 16000


class FasterWhisperSTT(STTBase):
    """Faster Whisper STT engine.

    Recording is interactive by default: press Enter to start and Enter again
    to stop. On CoralBoard the audio is captured with ``arecord``; on other
    platforms (for example macOS) ``sounddevice`` is used.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.model = None
        self.model_size = self.config.get("model_size") or self.settings.stt_model

    async def initialize(self) -> bool:
        """Initialize faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            device = self.settings.stt_device
            compute_type = self.settings.stt_compute_type
            if device != "cpu" and not compute_type:
                compute_type = "float16"

            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize FasterWhisper: {e}")
            return False

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.model = None
        self._initialized = False

    @property
    def quiet(self) -> bool:
        import os

        return os.environ.get("SAYLESS_QUIET", "0") == "1"

    def _say(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    async def transcribe(self, audio: Union[np.ndarray, bytes, str]) -> Optional[str]:
        """Transcribe audio using Faster Whisper."""
        if not self.model or not self._initialized:
            return None

        try:
            if isinstance(audio, str):
                # Read WAV directly so faster-whisper does not depend on the
                # PyAV version installed on the board.
                import soundfile as sf

                audio_data, sample_rate = sf.read(audio, dtype="float32")
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                if sample_rate != SAMPLE_RATE:
                    duration = audio_data.shape[0] / sample_rate
                    target = int(duration * SAMPLE_RATE)
                    audio_data = np.interp(
                        np.linspace(0, audio_data.shape[0], target, endpoint=False),
                        np.arange(audio_data.shape[0]),
                        audio_data,
                    ).astype(np.float32)
                segments, _ = self.model.transcribe(
                    audio_data,
                    language=self.settings.stt_language,
                    beam_size=self.settings.stt_beam_size,
                )
            elif isinstance(audio, bytes):
                import io

                import soundfile as sf

                audio_data, _ = sf.read(io.BytesIO(audio))
                segments, _ = self.model.transcribe(
                    audio_data,
                    language=self.settings.stt_language,
                    beam_size=self.settings.stt_beam_size,
                )
            else:
                segments, _ = self.model.transcribe(
                    audio,
                    language=self.settings.stt_language,
                    beam_size=self.settings.stt_beam_size,
                )

            text = " ".join(segment.text for segment in segments)
            return text.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    async def _wait_for_enter(self, prompt: str) -> None:
        """Read one Enter keypress without blocking the event loop."""
        await asyncio.to_thread(input, prompt)

    async def _record_sounddevice(self, duration: Optional[float] = None) -> np.ndarray:
        """Record with sounddevice until Enter (or for a fixed duration)."""
        import sounddevice as sd

        frames: list[np.ndarray] = []

        def callback(indata, _frames, _time, _status) -> None:
            frames.append(indata.copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        stream.start()
        try:
            if duration is None:
                await self._wait_for_enter("Recording... press Enter to stop. ")
            else:
                await asyncio.sleep(duration)
        finally:
            stream.stop()
            stream.close()

        if not frames:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).flatten()

    async def _record_arecord(self, duration: Optional[float] = None) -> Optional[str]:
        """Record with arecord. Returns the WAV path."""
        recordings = self.settings.data_dir / "recordings"
        recordings.mkdir(parents=True, exist_ok=True)
        wav_path = recordings / f"input-{uuid.uuid4().hex[:8]}.wav"

        cmd = [
            "arecord",
            "-q",
            "-D",
            self.settings.stt_audio_device,
            "-c",
            "1",
            "-r",
            str(SAMPLE_RATE),
            "-f",
            "S16_LE",
        ]
        if duration is not None:
            cmd += ["-d", str(max(1, int(duration)))]
        cmd.append(str(wav_path))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            if duration is None:
                await self._wait_for_enter("Recording... press Enter to stop. ")
            else:
                await asyncio.sleep(duration)
        finally:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)

        return str(wav_path)

    async def listen(
        self, duration: Optional[float] = None, start_prompt: bool = True
    ) -> Optional[str]:
        """Record and transcribe.

        By default the user presses Enter to start and Enter again to stop.
        Pass ``start_prompt=False`` when the caller already waited for Enter.
        Pass ``duration`` only for non-interactive callers.
        """
        if duration is None and not sys.stdin.isatty():
            raise RuntimeError(
                "Interactive recording needs a terminal. Run sayless from a "
                "terminal, or use 'sayless chat' for text input."
            )

        if duration is None and start_prompt:
            await self._wait_for_enter("Press Enter to start recording... ")

        use_arecord = shutil.which("arecord") is not None
        if use_arecord:
            wav_path = await self._record_arecord(duration)
            try:
                return await self.transcribe(wav_path)
            finally:
                Path(wav_path).unlink(missing_ok=True)

        audio = await self._record_sounddevice(duration)
        if audio.size == 0:
            return None
        return await self.transcribe(audio)
