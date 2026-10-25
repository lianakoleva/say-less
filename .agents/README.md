# SayLess Agent Instructions

This directory contains agent-specific instructions and configuration for AI assistants working on this project.

## Project Overview
SayLess is a local voice assistant with STT and LLM - like Siri updated for 2026. It runs locally on CPU and on Google CoralBoard (Gemma 3 270M via LiteRT-LM, faster-whisper `tiny.en`).

## Architecture
- Modular components: STT and LLM each separately runnable and configurable
- Tool calling to phone apps (Google Voice, Google Maps) and local data (contacts, notes)
- Offline queue in `~/.sayless/queue` for tasks requiring internet
- UI abstraction layer (CLI and TUI)
- All data stored locally in `~/.sayless/`

## Development Guidelines
- Keep components modular and independently testable
- UI should be swappable via `sayless/ui/` abstraction
- All user data goes in `~/.sayless/` dotfile folder
- Queue system must handle offline/online transitions
- Support CPU-only inference
- Maintain conversation history in `~/.sayless/chats/`

## Quiet Mode (ALWAYS USE IN AUTOMATED/AGENT CONTEXTS)
When running `sayless` commands, ALWAYS use quiet mode to reduce verbose output.

**Usage:**
```bash
sayless --quiet run
SAYLESS_QUIET=1 sayless run
SAYLESS_QUIET=1 sayless chat
```

**Behavior:**
- Suppress initialization banners and verbose component status
- Only show essential output (user input, assistant responses, errors)
- Use `YOU:`/`ASSISTANT:` prefixes in quiet mode
- Skip decorative panels and tables where possible

## Git Worktrees
Use git worktrees for new work. Create a worktree per task/feature in a separate directory. The main repo should be kept as the combination of all git worktrees - I'll maintain my local repo there for testing.
```bash
git worktree add ../sayless-<feature> -b feature/<name>
cd ../sayless-<feature>
```
Remove worktrees when done with `git worktree remove ../sayless-<feature>`.
