# say-less

Local voice assistant with STT and LLM - like Siri updated for 2026, running entirely on your machine or a CoralBoard.

## Features

- **Runs on CoralBoard**: Gemma 3 270M via LiteRT-LM, faster-whisper `tiny.en`, and CPU audio capture
- **Modular Components**: STT and LLM can each run separately and be toggled on/off
- **Press Enter to talk**: recording starts on Enter and stops on the next Enter (no fixed 5s window)
- **Tool Calling**: Integrate with phone apps and services via tools/MCP
  - Google Voice for calls and texts
  - Google Maps for directions and place search
  - Local contacts and notes
- **Local Data**: Contacts, notes, and chat history stored under `~/.sayless/`
- **Swappable UI**: CLI and TUI
- **Offline Queue**: Queue tasks when offline - they'll execute automatically when internet returns
- **Local First**: Runs offline when possible, only uses internet when needed

## Architecture

```
sayless/
├── core/       # Main assistant orchestration
├── stt/        # Speech-to-Text (faster-whisper tiny.en)
├── llm/        # Language Model (LiteRT-LM server + Gemma 3 270M)
├── tools/      # External integrations (phone, maps, contacts, notes)
├── data/       # Local contacts, notes, and chat history
├── ui/         # CLI and TUI frontends
├── queue/      # Offline task queue (~/.sayless/queue)
└── cli.py      # Command-line interface
```

The LLM runs in a small OpenAI-compatible server (`sayless.llm.server`) that
wraps LiteRT-LM on the CoralBoard CPU. The assistant and CLI talk to it over
HTTP, so the model process can outlive the UI.

## Installation

```bash
# One-command bootstrap (installs Python deps and starts the LiteRT-LM server)
./scripts/bootstrap.sh

# Or install manually
pip install -e ".[stt,llm,tools]"
sayless init
sayless setup --model-path /path/to/mobile_actions_q8_ekv1024.litertlm
```

## Model

SayLess uses `mobile_actions_q8_ekv1024.litertlm`, a FunctionGemma fine-tune of
Gemma 3 270M for mobile-action tool calling, running through LiteRT-LM. Put the
model file on the device and point `SAYLESS_LLM_MODEL_PATH` at it:

```bash
SAYLESS_LLM_MODEL_PATH=/home/root/models/mobile_actions_q8_ekv1024.litertlm
```

`sayless setup` verifies the file and starts the local server on port 8000:

```bash
python -m sayless.llm.server \
  --model /home/root/models/mobile_actions_q8_ekv1024.litertlm
```

The model uses a prompt-driven JSON protocol instead of native OpenAI tool
calling. SayLess sends the available function signatures in the prompt, and the
model replies with:

```json
{"function_call": {"name": "notes", "arguments": {"action": "create", "title": "demo"}}}
```

The assistant parses that, executes the matching local tool, then asks the model
for a short confirmation. Any other LiteRT-LM chat model can be swapped in via
`SAYLESS_LLM_MODEL_PATH`; a tool-trained model gives the most reliable calls.

## Quick Start

```bash
# Check dependencies (LiteRT-LM, model file, audio)
sayless doctor

# Initialize configuration
sayless init

# Verify the model and start the server
sayless setup

# Run voice assistant (press Enter to start and stop recording)
sayless run

# Text chat mode
sayless chat

# Listen once
sayless listen

# Check configuration
sayless config

# View/manage queue
sayless queue
sayless process-queue
```

## Configuration

Configure via environment variables (prefix `SAYLESS_`) or `.env` file:

```bash
# STT
SAYLESS_STT_ENABLED=true
SAYLESS_STT_ENGINE=faster-whisper
SAYLESS_STT_MODEL=tiny.en
SAYLESS_STT_AUDIO_DEVICE=hw:1,0

# LLM
SAYLESS_LLM_ENABLED=true
SAYLESS_LLM_ENGINE=litert-lm
SAYLESS_LLM_MODEL_NAME=mobile_actions_q8_ekv1024.litertlm
SAYLESS_LLM_MODEL_PATH=/home/root/models/mobile_actions_q8_ekv1024.litertlm
SAYLESS_LLM_BASE_URL=http://localhost:8000/v1

# Tools
SAYLESS_ENABLE_PHONE_TOOLS=true
SAYLESS_ENABLE_MAPS_TOOLS=true
SAYLESS_GOOGLE_MAPS_API_KEY=your_key_here
```

## Queue System

Tasks that require internet are automatically queued to `~/.sayless/queue/` when offline. When connectivity returns, run `sayless process-queue` or it will auto-process.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

