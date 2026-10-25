# 🫡 say less

Local voice assistant with STT and LLM - like Siri updated for 2026, running entirely on your machine or a CoralBoard.

## Features

- **Runs on CoralBoard**: Gemma 3 270M via LiteRT-LM, faster-whisper `tiny.en`, and CPU audio capture
- **Modular Components**: STT and LLM can each run separately and be toggled on/off
- **Press Enter to talk**: recording starts on Enter and stops on the next Enter
- **Tool Calling**: Integrated with local actions and services
  - Seeded contacts, listing, and contact lookup
  - Calls and texts to saved contacts (local command or simulated log)
  - Weather via Open-Meteo (no API key)
  - Google Maps for directions and place search
  - Local notes
- **Local Data**: Contacts, notes, chat history, and phone actions stored under `~/.sayless/`
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
wraps LiteRT-LM on the CoralBoard. The assistant and CLI talk to it over
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

FunctionGemma has its own function-calling protocol, so SayLess sends the tool
declarations through LiteRT-LM and the model emits a native call:

```text
<start_function_call>call:create_contact{name:<escape>Mom<escape>,
phone_number:<escape>1224-879-1662<escape>}<end_function_call>
```

LiteRT-LM parses that into OpenAI-style `tool_calls`. SayLess executes the
matching local tool (for example `contacts` with action `add`) and prints a
short confirmation. The schemas the model sees are the flat, single-purpose
functions in `sayless/tools/model_tools.py` (`create_contact`, `send_text`,
`make_note`, ...), which match how FunctionGemma was trained.

Base chat models without function calling fall back to a JSON prompt
(`{"function_call": ...}`), so any LiteRT-LM chat model can be swapped in via
`SAYLESS_LLM_MODEL_PATH`.

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

# Seed the demo contacts (also happens automatically on a fresh install)
sayless contacts seed
sayless contacts list

# See what SayLess can do
sayless tools

# See preferences SayLess has learned (for example the weather city)
sayless profile

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
SAYLESS_ENABLE_WEATHER_TOOLS=true
SAYLESS_GOOGLE_MAPS_API_KEY=your_key_here

# Seeded contacts (demo list when the address book is empty)
SAYLESS_SEED_CONTACTS=true

# Weather (Open-Meteo, no API key). When unset, SayLess asks for a city the
# first time and remembers it.
SAYLESS_WEATHER_LOCATION=Seattle
SAYLESS_WEATHER_UNITS=fahrenheit

# Navigation. Google Directions is used when a key is set; otherwise SayLess
# falls back to free OpenStreetMap + OSRM routing. The starting point is asked
# on each route; set this only for a fixed default.
# SAYLESS_CURRENT_LOCATION=1 Market St, San Francisco
SAYLESS_NAVIGATION_UNITS=miles
# SAYLESS_NAVIGATION_COMMAND=open "https://maps.google.com/?saddr={origin}&daddr={destination}"

# Calls and texts. Set these to run a real phone app or script; {number}
# and {message} are substituted. When unset, actions are logged to
# ~/.sayless/phone.log and reported as simulated.
# SAYLESS_PHONE_CALL_COMMAND=open "tel:{number}"
# SAYLESS_PHONE_TEXT_COMMAND=my-sms-script --to {number} --body {message}
```

### Contacts and phone actions

On startup SayLess adds any missing demo contacts (Judy Sellers, Mom, Alex
Rivera, Priya Patel, and Sam Chen). Existing contacts are left alone. Try:

```text
Who are my contacts?
What's Judy's number?
Send Judy a text saying I'm running late.
Call Mom.
```

Contact and phone requests are routed locally rather than by the language
model, because the small FunctionGemma fine-tune often picks the wrong
function for them. Contact saves merge by name, so adding an email later
updates the phone contact instead of creating a duplicate.

### Weather

Weather requests use the free Open-Meteo API:

```text
What's the weather?
What's the weather in Seattle tomorrow?
Will it rain in New York this week?
```

Set `SAYLESS_WEATHER_LOCATION` to change the default city and
`SAYLESS_WEATHER_UNITS` to `celsius` for metric output.

The first time you ask about the weather without naming a city, SayLess asks
for one and saves it in `~/.sayless/profile/user.json`, so later requests work
without asking again. Run `sayless profile` to see saved preferences.

### Navigation and travel time

Navigation uses the Google Directions API when `SAYLESS_GOOGLE_MAPS_API_KEY`
is set. Without a key it falls back to free OpenStreetMap geocoding and OSRM
routing, so it still works on the board.

```text
How long will it take me to get home?
How long does it take to get to work?
Navigate to 123 Main St.
Walking directions to the park.
How long will it take me to get to work by transit?
Directions from home to SFO.
```

If you do not say how you are travelling, SayLess asks which mode to use:

```text
How long will it take me to get home?
Should I use driving, walking, or transit?
walking
On foot, it'll take about 24 minutes to get home, about 1.2 miles.
```

You can also include the mode in the question ("...by transit", "walking
directions..."). Spoken addresses are cleaned up before geocoding, so
`1 2 0 1 Tennessee straight San Francisco` is looked up as
`1201 Tennessee Street, San Francisco`.

SayLess does not assume where you are. Since your current location can change,
it asks for a starting point on every request that does not already include
one:

```text
How long will it take me to get home?
Where are you starting from? Give me a city or address, or say
'use my current location' to estimate it.
Oakland
Should I use driving, walking, or transit?
driving
It'll take about 24 minutes to get home from Oakland, about 12 miles.
```

The answer is used for that route only and is not saved. "Use my current
location" estimates from the network for that one request. If you want a fixed
starting point, set `SAYLESS_CURRENT_LOCATION`; otherwise SayLess always asks.

The first time you ask for directions to home or work, SayLess also asks for
that address and saves it. Later questions reuse the saved home/work address
but still ask where the trip starts:

```text
How long will it take me to get home?
What's your home address? I'll remember it for next time.
I live at 1 Market St, San Francisco
Got it: 1 Market St, San Francisco. Where are you starting from? ...
Oakland
Should I use driving, walking, or transit?
driving
It'll take about 24 minutes to get home from Oakland, about 12 miles.
```

You can name the starting point up front instead: `How long will it take me to
get home from Oakland?` or `Directions from home to SFO.`

Set `SAYLESS_CURRENT_LOCATION` only if you want a fixed starting point; by
default SayLess asks each time. Set `SAYLESS_NAVIGATION_UNITS=km` for metric
distances. `SAYLESS_NAVIGATION_COMMAND` can launch a maps app, with `{origin}`,
`{destination}`, and `{label}` substituted.

Saved addresses can be corrected any time:

```text
My home is 1201 Tennessee Street, San Francisco.
Set my work address to 1 Market St, San Francisco.
```

The free OSRM fallback only has car routing, so walking, biking, and transit
times are estimated from the route distance and marked `(estimated)`. Google
Directions returns real times for every mode when a key is configured, but the
**legacy Directions API** must be enabled for that key. If Google rejects the
request, SayLess prints the reason in its tool output and falls back to OSRM.

### Notes and lists

Notes are stored locally under `~/.sayless/notes/`. "Make a list" is a
note too:

```text
Make a list of groceries: milk, eggs
Make a grocery list with milk and eggs
Make a note called Ideas with build a robot
Make a list
What should I put on the list?
milk, eggs, bread
List my notes
Search my notes for milk
```

A bare "make a list" asks what to add and saves the next thing you say as the
list contents.

## Queue System

Tasks that require internet are automatically queued to `~/.sayless/queue/` when offline. When connectivity returns, run `sayless process-queue` or it will auto-process.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

