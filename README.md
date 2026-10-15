# Pipecat + Ollama Voice Kit

Fully local voice assistant pipeline — mic in, Slovak speech out, no cloud APIs, no API keys:

```
mic → faster-whisper (STT) → Ollama (LLM) → Piper (TTS) → speakers
        Silero VAD turn-taking in between
```

Python port of the working browser-based proof of concept; built on Pipecat 1.12 API
(verified against `examples/voice/voice-piper.py` and `examples/function-calling/function-calling-ollama.py`).

## Prerequisites

1. **mise** — installs Python 3.12, uv and Ollama for this directory.
2. **PortAudio** (system lib, only for `--transport local`): `sudo pacman -S portaudio`.
3. Working mic + speakers (PipeWire on your Omarchy desktop).

## Install

```bash
cd ollama_pipekit
mise install          # python, uv, ollama
mise run ollama       # in a second terminal, unless Ollama already runs as a service
mise run install      # .venv + Python deps, creates .env, pulls OLLAMA_MODEL
mise run configure    # optional: pick language, models, voice (rewrites .env)
```

`requirements.txt` selects the pipecat extras actually needed: `whisper` (faster-whisper),
`piper` (local TTS), `local` (mic via PyAudio), `webrtc` (browser client) and `runner`
(the prebuilt web UI server). Ollama needs no extra — it rides the built-in
OpenAI-compatible base; Silero VAD ships with core (onnxruntime).

## Run

```bash
# Local mic + speakers (direct audio)
mise run bot -- --transport local

# Browser client (prebuilt WebRTC UI at http://localhost:7860) — like your previous POC
mise run bot -- --transport webrtc

# Plain `mise run bot` shows an interactive transport picker.
```

First run downloads:
- faster-whisper `small` model (~460 MB, cached in `~/.cache/huggingface`)
- Piper voice `sk_SK-sklabca-medium` (~60 MB, cached in `~/.cache/pipecat/piper`)

## Configuration (env vars or .env)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `llama3.2` | any pulled model; tool calling needs llama3.2+ |
| `BOT_LANGUAGE` | `sk` | `sk` or `en`: prompt, greeting, STT language, default voice |
| `STT_LANGUAGE` | = `BOT_LANGUAGE` | whisper language override; empty for auto-detect |
| `WHISPER_MODEL` | `small` | tiny / base / small / medium / large-v3 |
| `PIPER_VOICE` | per language | any Piper voice id |

To add a language, add an entry to `LANGUAGES` in `bot.py` (voice + 3 short texts).

Remote Ollama (e.g. homelab server): set `OLLAMA_URL=http://<host>:11434/v1` and
make sure Ollama there listens on the network (`OLLAMA_HOST=0.0.0.0`).

## Function calling

One demo tool (`get_current_time`) is wired up. Add functions taking
`params: FunctionCallParams` plus typed kwargs, register them in
`LLMContext(tools=[...])`, and Ollama models that support tools will call them.
See `examples/function-calling/function-calling-ollama.py` upstream.

## Troubleshooting

- **Cannot open mic (PortAudio error)** — `local` transport needs PortAudio;
  on Arch/Omarchy ensure `pipewire-pulse`/ALSA is functional: `python -c "import pyaudio"` should not error.
- **No sound out** — check `audio_out_sample_rate` (22050 set for Piper); if you swap TTS, revisit.
- **Ollama connection refused** — `curl http://localhost:11434/api/tags` should list models.
- **Whisper too slow** — drop `WHISPER_MODEL=base` or `tiny`.
- **Interrupt quality** — barge-in uses Silero VAD; speak over the bot and it stops.