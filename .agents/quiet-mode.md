# Quiet Mode

When running `sayless` commands in automated environments or when requested, use quiet mode.

## Usage
Set environment variable: `SAYLESS_QUIET=1`

Or use flag: `sayless --quiet run`

## Behavior
- Suppress initialization banners and verbose component status
- Only show essential output (user input, assistant responses, errors)
- Reduce logging noise
- Skip decorative panels and tables where possible

## Implementation
Quiet mode is controlled via environment variable or CLI flag.
