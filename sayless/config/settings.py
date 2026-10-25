"""Settings and configuration for SayLess."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main settings for SayLess."""
    
    model_config = SettingsConfigDict(
        env_prefix="SAYLESS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # General
    app_name: str = "sayless"
    debug: bool = False
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".sayless")
    
    # STT settings
    stt_enabled: bool = True
    stt_model: str = "tiny.en"  # faster-whisper model size or path
    stt_engine: str = "faster-whisper"
    stt_device: str = "cpu"
    stt_language: str = "en"
    stt_audio_device: str = "hw:1,0"  # ALSA device used by arecord on CoralBoard
    stt_compute_type: str = "int8"
    stt_beam_size: int = 1

    # LLM settings (LiteRT-LM / Gemma on CoralBoard)
    llm_enabled: bool = True
    llm_engine: str = "litert-lm"
    llm_model_name: str = "mobile_actions_q8_ekv1024.litertlm"
    llm_model_path: Path = Field(
        default_factory=lambda: Path("/home/root/models/mobile_actions_q8_ekv1024.litertlm")
    )
    llm_base_url: str = "http://localhost:8000/v1"
    llm_server_host: str = "127.0.0.1"
    llm_server_port: int = 8000
    llm_num_threads: Optional[int] = None
    llm_temperature: float = 0.1
    llm_top_p: float = 0.95
    llm_top_k: int = 64
    llm_max_tokens: int = 512
    
    
    # Queue settings
    queue_enabled: bool = True
    queue_dir: Path = Field(default_factory=lambda: Path.home() / ".sayless" / "queue")
    check_internet_on_start: bool = True
    
    # Tools/Integrations
    enable_phone_tools: bool = True
    enable_maps_tools: bool = True
    enable_weather_tools: bool = True
    google_voice_enabled: bool = False
    google_maps_api_key: Optional[str] = None

    # Weather (Open-Meteo, no API key)
    weather_location: Optional[str] = None
    weather_units: str = "fahrenheit"

    # Navigation / maps
    current_location: Optional[str] = None
    navigation_units: str = "miles"
    navigation_command: Optional[str] = None

    # Phone/text provider. When unset, calls and texts are logged to
    # ~/.sayless/phone.log and reported as simulated.
    phone_call_command: Optional[str] = None
    phone_text_command: Optional[str] = None

    # Seed demo contacts when the address book is empty.
    seed_contacts: bool = True
    
    # MCP
    mcp_config_path: Path = Field(default_factory=lambda: Path.home() / ".sayless" / "mcp.json")
    
    def model_post_init(self, __context: Any) -> None:
        """Create necessary directories after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "recordings").mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
