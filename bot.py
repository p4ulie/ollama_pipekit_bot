#!/usr/bin/env python3
"""Pipecat + Ollama voice bot — fully local pipeline.

Stack (no cloud APIs, no keys):
  LocalAudioTransport (mic/speakers) -> WhisperSTTService (faster-whisper, local)
  -> OLLamaLLMService (Ollama OpenAI-compatible) -> PiperTTSService (local TTS)
  -> back out via speakers. Silero VAD for turn-taking.

Based on pipecat 1.12 examples (voice-piper.py, function-calling-ollama.py).
Run:  mise run bot -- --transport local      # mic + speakers directly
      mise run bot -- --transport webrtc     # browser client at http://localhost:7860
"""
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

# pyaudio is only needed for the direct-mic transport; import lazily so the bot
# module loads (and --transport webrtc works) on machines without PortAudio.
try:
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    HAS_LOCAL_AUDIO = True
except ImportError:
    HAS_LOCAL_AUDIO = False
    logger.warning("pyaudio not available - 'local' transport disabled, use --transport webrtc")

load_dotenv(override=True)

# ==== configuration ==========================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # tiny|base|small|medium|large-v3

# Per-language texts + default Piper voice. Add a language by adding an entry
# (voice ids: https://huggingface.co/rhasspy/piper-voices).
LANGUAGES = {
    "sk": {
        "voice": "sk_SK-lili-medium",  # the only Slovak Piper voice
        "prompt": (
            "Si priateľský hlasový asistent. Odpovedaj stručne a priamo, "
            "maximálne 2–3 krátke vety. Tvoje odpovede sa čítajú nahlas — "
            "nepoužívaj markdown, emoji, zoznamy ani formátovanie."
        ),
        "intro": "Predstav sa užívateľovi.",
        "wait": "Chvíľku, pozerám na to.",
    },
    "en": {
        "voice": "en_US-ryan-high",
        "prompt": (
            "You are a friendly voice assistant. Answer briefly and directly, "
            "at most 2–3 short sentences. Your replies are read aloud — "
            "do not use markdown, emoji, lists or formatting."
        ),
        "intro": "Introduce yourself to the user.",
        "wait": "One moment, let me check.",
    },
}

BOT_LANGUAGE = os.getenv("BOT_LANGUAGE", "sk")
if BOT_LANGUAGE not in LANGUAGES:
    raise SystemExit(f"BOT_LANGUAGE={BOT_LANGUAGE!r} unknown; pick one of {', '.join(LANGUAGES)}")
LANG = LANGUAGES[BOT_LANGUAGE]

STT_LANGUAGE = os.getenv("STT_LANGUAGE", BOT_LANGUAGE)  # "" = whisper auto-detect
PIPER_VOICE = os.getenv("PIPER_VOICE", LANG["voice"])
SYSTEM_PROMPT = LANG["prompt"]

# ==== tool example (function calling) ========================================

from pipecat.services.llm_service import FunctionCallParams


async def get_current_time(params: FunctionCallParams):
    """Get the current local date and time."""
    import datetime
    await params.result_callback({"time": datetime.datetime.now().isoformat(timespec="seconds")})


# ==== transports =============================================================

transport_params = {
    # Browser client with prebuilt UI at http://localhost:7860 (no third-party key)
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}



async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot (Ollama: {} @ {}, STT: whisper-{}, TTS: piper {})",
                OLLAMA_MODEL, OLLAMA_URL, WHISPER_MODEL, PIPER_VOICE)

    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(
            model=WHISPER_MODEL,
            language=STT_LANGUAGE,
        ),
    )

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(
            voice=PIPER_VOICE,       # auto-downloads on first run to ~/.cache/pipecat/piper
        ),
    )

    llm = OLLamaLLMService(
        base_url=OLLAMA_URL,
        settings=OLLamaLLMService.Settings(
            model=OLLAMA_MODEL,
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame(LANG["wait"]))

    context = LLMContext(tools=[get_current_time])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),            # mic audio / WebRTC input
            stt,                          # faster-whisper (local, segmented)
            user_aggregator,
            llm,                          # Ollama
            tts,                          # Piper (local)
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=22050,  # piper native rate; service resamples as needed
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    async def greet():
        context.add_message({"role": "developer", "content": LANG["intro"]})
        await worker.queue_frames([LLMRunFrame()])

    if HAS_LOCAL_AUDIO and isinstance(transport, LocalAudioTransport):
        await greet()  # mic/speakers are always "connected"; no client events
    else:
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await greet()

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


def run_local():
    """Mic + speakers directly. Pipecat's runner has no 'local' transport, so bypass it."""
    if not HAS_LOCAL_AUDIO:
        raise SystemExit("pyaudio not available - use --transport webrtc")
    import asyncio
    args = RunnerArguments()
    args.handle_sigint = True
    args.pipeline_idle_timeout_secs = None  # nobody "leaves" a local session
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))
    asyncio.run(run_bot(transport, args))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] in (["--transport", "local"], ["-t", "local"]):
        run_local()
    else:
        from pipecat.runner.run import main
        main()