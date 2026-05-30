from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aiortc.mediastreams import AudioStreamTrack
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from scenarios import SCENARIOS

load_dotenv(override=True)


DEFAULT_MAX_TURNS = 8
DEFAULT_AGENT_TEMPERATURE = 0.3
DEFAULT_CALLER_TEMPERATURE = 0.7
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_BOT_URL = "http://localhost:7860"
DEFAULT_TRANSCRIBE_ASR_URL = "ws://192.168.7.228:8081"
DEFAULT_WEBRTC_CONNECT_TIMEOUT_SECONDS = 20.0
DEFAULT_VOICE_REPLY_TIMEOUT_SECONDS = 35.0
DEFAULT_VOICE_SILENCE_SECONDS = 1.1
DEFAULT_VOICE_RMS_THRESHOLD = 350
CALLER_AUDIO_SAMPLE_RATE = 48000
TRANSCRIPTION_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class EvalConfig:
    agent_model: str
    caller_model: str
    judge_model: str
    agent_base_url: str
    caller_base_url: str
    judge_base_url: str
    max_turns: int
    output_path: Path
    runs_path: Path
    latency_path: Path
    today: date
    agent_temperature: float
    caller_temperature: float
    judge_temperature: float
    disable_thinking: bool
    bot_url: str = DEFAULT_BOT_URL
    bot_request_timeout_seconds: float = 120.0
    webrtc_connect_timeout_seconds: float = DEFAULT_WEBRTC_CONNECT_TIMEOUT_SECONDS
    voice_reply_timeout_seconds: float = DEFAULT_VOICE_REPLY_TIMEOUT_SECONDS
    voice_silence_seconds: float = DEFAULT_VOICE_SILENCE_SECONDS
    voice_rms_threshold: int = DEFAULT_VOICE_RMS_THRESHOLD
    transcribe_asr_url: str = DEFAULT_TRANSCRIBE_ASR_URL
    caller_tts_provider: str = "gradium"
    caller_tts_voice: str | None = None


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_positive_float(name: str, default: float) -> float:
    return max(0.1, _env_float(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_today(value: str | None) -> date:
    if not value:
        return date.today()
    return date.fromisoformat(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run voice scenario evals against an already-running Pipecat bot served by "
            "`uv run bot.py`."
        )
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario id to run. May be repeated or comma-separated. Defaults to all.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N selected scenarios.")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=_env_int("EVAL_MAX_TURNS", DEFAULT_MAX_TURNS),
        help=f"Maximum caller/agent turns per scenario. Default: {DEFAULT_MAX_TURNS}.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("EVAL_RESULTS_PATH", "results.json"),
        help="Path for the full JSON result file. Default: results.json.",
    )
    parser.add_argument(
        "--runs",
        default=os.getenv("EVAL_RUNS_PATH", "runs.jsonl"),
        help="Path for the trend JSONL file. Default: runs.jsonl.",
    )
    parser.add_argument(
        "--latency-path",
        default=os.getenv("LATENCY_METRICS_PATH", "latency.jsonl"),
        help="Optional live-call latency JSONL path to summarize alongside eval results.",
    )
    parser.add_argument(
        "--today",
        default=os.getenv("EVAL_TODAY"),
        help="ISO date used in the bot prompt for reproducible relative-date scenarios.",
    )
    parser.add_argument(
        "--bot-url",
        default=os.getenv("EVAL_BOT_URL", DEFAULT_BOT_URL),
        help=f"Running Pipecat voice bot URL. Default: {DEFAULT_BOT_URL}.",
    )
    parser.add_argument(
        "--transcribe-asr-url",
        default=os.getenv("EVAL_TRANSCRIBE_ASR_URL", os.getenv("NVIDIA_ASR_URL", DEFAULT_TRANSCRIBE_ASR_URL)),
        help=(
            "WebSocket ASR URL used by the evaluator to transcribe the bot's returned audio. "
            f"Default: {DEFAULT_TRANSCRIBE_ASR_URL}."
        ),
    )
    parser.add_argument(
        "--caller-tts-provider",
        choices=["gradium", "say"],
        default=os.getenv("EVAL_CALLER_TTS_PROVIDER", "gradium"),
        help="TTS provider used to speak simulated caller turns into the live voice bot.",
    )
    parser.add_argument(
        "--caller-tts-voice",
        default=os.getenv("EVAL_CALLER_TTS_VOICE"),
        help="Optional caller TTS voice id/name. Defaults to GRADIUM_VOICE_ID for Gradium.",
    )
    parser.add_argument("--list", action="store_true", help="List scenario ids and exit.")
    return parser


def _build_config(args: argparse.Namespace) -> EvalConfig:
    default_base_url = _first_env(
        "EVAL_BASE_URL",
        "NEMOTRON_LLM_URL",
        "NIM_BASE_URL",
        "NVIDIA_BASE_URL",
        default="http://192.168.7.228:8000/v1",
    )
    agent_model = _first_env(
        "EVAL_AGENT_MODEL",
        "EVAL_MODEL",
        "NEMOTRON_LLM_MODEL",
        default="nvidia/nemotron-3-super",
    )
    caller_model = _first_env("EVAL_CALLER_MODEL", "EVAL_MODEL", default=agent_model)
    judge_model = _first_env("EVAL_JUDGE_MODEL", "EVAL_MODEL", default=agent_model)

    return EvalConfig(
        agent_model=agent_model or "nvidia/nemotron-3-super",
        caller_model=caller_model or "nvidia/nemotron-3-super",
        judge_model=judge_model or "nvidia/nemotron-3-super",
        agent_base_url=_first_env("EVAL_AGENT_BASE_URL", "EVAL_BASE_URL", default=default_base_url)
        or "",
        caller_base_url=_first_env(
            "EVAL_CALLER_BASE_URL",
            "EVAL_BASE_URL",
            default=default_base_url,
        )
        or "",
        judge_base_url=_first_env("EVAL_JUDGE_BASE_URL", "EVAL_BASE_URL", default=default_base_url)
        or "",
        max_turns=max(1, args.max_turns),
        output_path=Path(args.output),
        runs_path=Path(args.runs),
        latency_path=Path(args.latency_path),
        today=_parse_today(args.today),
        agent_temperature=_env_float("EVAL_AGENT_TEMPERATURE", DEFAULT_AGENT_TEMPERATURE),
        caller_temperature=_env_float("EVAL_CALLER_TEMPERATURE", DEFAULT_CALLER_TEMPERATURE),
        judge_temperature=_env_float("EVAL_JUDGE_TEMPERATURE", DEFAULT_JUDGE_TEMPERATURE),
        disable_thinking=_env_bool("EVAL_DISABLE_THINKING", True),
        bot_url=args.bot_url.rstrip("/"),
        bot_request_timeout_seconds=_env_positive_float("EVAL_BOT_REQUEST_TIMEOUT_SECONDS", 120.0),
        webrtc_connect_timeout_seconds=_env_positive_float(
            "EVAL_WEBRTC_CONNECT_TIMEOUT_SECONDS",
            DEFAULT_WEBRTC_CONNECT_TIMEOUT_SECONDS,
        ),
        voice_reply_timeout_seconds=_env_positive_float(
            "EVAL_VOICE_REPLY_TIMEOUT_SECONDS",
            DEFAULT_VOICE_REPLY_TIMEOUT_SECONDS,
        ),
        voice_silence_seconds=_env_positive_float(
            "EVAL_VOICE_SILENCE_SECONDS",
            DEFAULT_VOICE_SILENCE_SECONDS,
        ),
        voice_rms_threshold=max(
            1,
            _env_int("EVAL_VOICE_RMS_THRESHOLD", DEFAULT_VOICE_RMS_THRESHOLD),
        ),
        transcribe_asr_url=args.transcribe_asr_url,
        caller_tts_provider=args.caller_tts_provider,
        caller_tts_voice=args.caller_tts_voice
        or os.getenv("GRADIUM_VOICE_ID")
        or "Eu9iL_CYe8N-Gkx_",
    )


def _client(base_url: str, role: str) -> OpenAI:
    api_key = _first_env(
        f"EVAL_{role}_API_KEY",
        "EVAL_API_KEY",
        "NEMOTRON_LLM_API_KEY",
        "NIM_API_KEY",
        "NVIDIA_API_KEY",
        default="EMPTY",
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def _json_post(url: str, payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _json_get(url: str, timeout_seconds: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _require_gradium_api_key() -> str:
    api_key = _first_env("EVAL_CALLER_TTS_API_KEY", "GRADIUM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gradium caller TTS requires EVAL_CALLER_TTS_API_KEY or GRADIUM_API_KEY."
        )
    return api_key


def _pcm_rms(pcm: bytes) -> float:
    import numpy as np

    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def _audio_frame_to_pcm(frame: Any, *, sample_rate: int) -> bytes:
    import numpy as np
    from av import AudioFrame, AudioResampler

    resampler = AudioResampler(format="s16", layout="mono", rate=sample_rate)
    out = bytearray()
    frames = resampler.resample(frame)
    for resampled in frames:
        array = resampled.to_ndarray()
        if array.ndim > 1:
            array = array[0]
        out.extend(array.astype(np.int16).tobytes())
    return bytes(out)


def _resample_pcm(pcm: bytes, *, source_rate: int, target_rate: int) -> bytes:
    import numpy as np
    from av import AudioFrame, AudioResampler

    if source_rate == target_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return b""
    frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
    frame.sample_rate = source_rate
    resampler = AudioResampler(format="s16", layout="mono", rate=target_rate)
    out = bytearray()
    for resampled in resampler.resample(frame):
        array = resampled.to_ndarray()
        if array.ndim > 1:
            array = array[0]
        out.extend(array.astype(np.int16).tobytes())
    return bytes(out)


def _decode_audio_file_to_pcm(path: Path, *, target_sample_rate: int) -> bytes:
    import av
    import numpy as np
    from av import AudioResampler

    out = bytearray()
    resampler = AudioResampler(format="s16", layout="mono", rate=target_sample_rate)
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                array = resampled.to_ndarray()
                if array.ndim > 1:
                    array = array[0]
                out.extend(array.astype(np.int16).tobytes())
    return bytes(out)


def _synthesize_with_say(text: str, *, sample_rate: int, voice: str | None) -> bytes:
    with tempfile.TemporaryDirectory(prefix="voice-eval-") as tmpdir:
        output_path = Path(tmpdir) / "caller.aiff"
        command = ["say"]
        if voice:
            command.extend(["-v", voice])
        command.extend(["-o", str(output_path), text])
        subprocess.run(command, check=True, capture_output=True)
        pcm = _decode_audio_file_to_pcm(output_path, target_sample_rate=sample_rate)
    if not pcm:
        raise RuntimeError("macOS `say` produced an empty audio file.")
    return pcm


async def _synthesize_with_gradium(text: str, *, voice: str | None) -> tuple[bytes, int]:
    import websockets

    api_key = _require_gradium_api_key()
    url = os.getenv("EVAL_CALLER_TTS_URL", "wss://api.gradium.ai/api/speech/tts")
    context_id = uuid.uuid4().hex
    setup = {
        "type": "setup",
        "output_format": "pcm",
        "voice_id": voice or os.getenv("GRADIUM_VOICE_ID", "Eu9iL_CYe8N-Gkx_"),
        "close_ws_on_eos": True,
        "client_req_id": context_id,
    }
    headers = {"x-api-key": api_key, "x-api-source": "eval_runner"}
    audio = bytearray()
    async with websockets.connect(url, additional_headers=headers) as websocket:
        await websocket.send(json.dumps(setup))
        await websocket.send(json.dumps({"type": "text", "text": text, "client_req_id": context_id}))
        await websocket.send(json.dumps({"type": "end_of_stream", "client_req_id": context_id}))
        async for message in websocket:
            payload = json.loads(message)
            if payload.get("client_req_id") not in {None, context_id}:
                continue
            if payload.get("type") == "audio":
                audio.extend(base64.b64decode(payload.get("audio", "")))
            elif payload.get("type") == "end_of_stream":
                break
            elif payload.get("type") == "error":
                raise RuntimeError(f"Gradium caller TTS error: {payload}")
    if not audio:
        raise RuntimeError("Gradium caller TTS returned no audio.")
    return bytes(audio), CALLER_AUDIO_SAMPLE_RATE


async def _synthesize_caller_audio(text: str, config: EvalConfig) -> bytes:
    if config.caller_tts_provider == "say":
        return await asyncio.to_thread(
            _synthesize_with_say,
            text,
            sample_rate=CALLER_AUDIO_SAMPLE_RATE,
            voice=config.caller_tts_voice,
        )
    pcm, sample_rate = await _synthesize_with_gradium(text, voice=config.caller_tts_voice)
    return _resample_pcm(
        pcm,
        source_rate=sample_rate,
        target_rate=CALLER_AUDIO_SAMPLE_RATE,
    )


async def _transcribe_pcm_with_nvidia(
    pcm: bytes,
    *,
    url: str,
    timeout_seconds: float,
) -> str:
    import websockets

    if not pcm:
        return ""

    final_text = ""
    latest_text = ""
    async with websockets.connect(url, ping_interval=20.0, ping_timeout=20.0) as websocket:
        try:
            ready = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            payload = json.loads(ready)
            if payload.get("type") == "transcript":
                latest_text = str(payload.get("text") or "")
        except TimeoutError:
            pass

        chunk_size = TRANSCRIPTION_SAMPLE_RATE * 2 // 10
        for offset in range(0, len(pcm), chunk_size):
            await websocket.send(pcm[offset : offset + chunk_size])
        await websocket.send(json.dumps({"type": "reset", "finalize": True}))

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            except TimeoutError:
                break
            payload = json.loads(message)
            if payload.get("type") != "transcript":
                continue
            text = str(payload.get("text") or "").strip()
            if text:
                latest_text = text
            if payload.get("is_final"):
                final_text = text
                break
    return (final_text or latest_text).strip()


class QueuedAudioTrack(AudioStreamTrack):
    """aiortc audio track that emits queued caller PCM and silence otherwise."""

    def __init__(self, *, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = sample_rate // 50
        self.bytes_per_frame = self.samples_per_frame * 2
        self.timestamp = 0
        self.started_at = time.monotonic()
        self.chunks: deque[bytes] = deque()

    def enqueue_pcm(self, pcm: bytes) -> float:
        remainder = len(pcm) % self.bytes_per_frame
        if remainder:
            pcm += bytes(self.bytes_per_frame - remainder)
        for offset in range(0, len(pcm), self.bytes_per_frame):
            self.chunks.append(pcm[offset : offset + self.bytes_per_frame])
        return (len(pcm) / 2) / self.sample_rate

    async def recv(self):
        import numpy as np
        from av import AudioFrame

        target_time = self.started_at + (self.timestamp / self.sample_rate)
        delay = target_time - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

        chunk = self.chunks.popleft() if self.chunks else bytes(self.bytes_per_frame)
        samples = np.frombuffer(chunk, dtype=np.int16)
        frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.timestamp
        frame.time_base = Fraction(1, self.sample_rate)
        self.timestamp += self.samples_per_frame
        return frame


class LiveWebRTCSession:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.pc = None
        self.audio_track: QueuedAudioTrack | None = None
        self.remote_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._reader_tasks: list[asyncio.Task] = []
        self._ping_task: asyncio.Task | None = None

    async def connect(self, scenario: Mapping[str, Any], *, today: date) -> None:
        from aiortc import RTCConfiguration, RTCIceServer, RTCSessionDescription

        start_response = await asyncio.to_thread(
            _json_post,
            f"{self.config.bot_url}/start",
            {
                "transport": "webrtc",
                "enableDefaultIceServers": True,
                "body": {
                    "eval": True,
                    "scenario_id": scenario.get("id"),
                    "from_number": scenario.get("from_number"),
                    "today": today.isoformat(),
                    "initial_bookings": scenario.get("initial_bookings", []),
                },
            },
            self.config.bot_request_timeout_seconds,
        )
        session_id = start_response.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError(f"Pipecat /start did not return sessionId: {start_response!r}")

        ice_servers = []
        for server in (start_response.get("iceConfig") or {}).get("iceServers", []):
            urls = server.get("urls") if isinstance(server, dict) else None
            if urls:
                ice_servers.append(RTCIceServer(urls=urls))

        self.pc = self._create_peer_connection(RTCConfiguration(iceServers=ice_servers))
        self.audio_track = QueuedAudioTrack(sample_rate=CALLER_AUDIO_SAMPLE_RATE)
        audio_transceiver = self.pc.addTransceiver("audio", direction="sendrecv")
        audio_transceiver.sender.replaceTrack(self.audio_track)
        self.pc.addTransceiver("video", direction="recvonly")
        data_channel = self.pc.createDataChannel("chat", ordered=True)
        self._setup_data_channel(data_channel)

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self._wait_for_ice_gathering_complete()

        local_description = self.pc.localDescription
        offer_url = f"{self.config.bot_url}/sessions/{session_id}/api/offer"
        answer = await asyncio.to_thread(
            _json_post,
            offer_url,
            {"sdp": local_description.sdp, "type": local_description.type},
            self.config.bot_request_timeout_seconds,
        )
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        await self._wait_until_connected()

    def _create_peer_connection(self, configuration: Any):
        from aiortc import RTCPeerConnection

        pc = RTCPeerConnection(configuration)

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                self._reader_tasks.append(asyncio.create_task(self._read_remote_audio(track)))

        return pc

    def _setup_data_channel(self, data_channel: Any) -> None:
        @data_channel.on("open")
        def on_open():
            async def ping_loop():
                while data_channel.readyState == "open":
                    data_channel.send(f"ping: {int(time.time() * 1000)}")
                    await asyncio.sleep(1.0)

            self._ping_task = asyncio.create_task(ping_loop())

    async def _wait_for_ice_gathering_complete(self) -> None:
        if self.pc.iceGatheringState == "complete":
            return
        done = asyncio.Event()

        @self.pc.on("icegatheringstatechange")
        def on_ice_gathering_state_change():
            if self.pc.iceGatheringState == "complete":
                done.set()

        await asyncio.wait_for(done.wait(), timeout=self.config.webrtc_connect_timeout_seconds)

    async def _wait_until_connected(self) -> None:
        if self.pc.connectionState == "connected":
            return
        connected = asyncio.Event()

        @self.pc.on("connectionstatechange")
        def on_connection_state_change():
            if self.pc.connectionState == "connected":
                connected.set()

        await asyncio.wait_for(connected.wait(), timeout=self.config.webrtc_connect_timeout_seconds)

    async def _read_remote_audio(self, track: Any) -> None:
        from aiortc.mediastreams import MediaStreamError

        while True:
            try:
                frame = await track.recv()
            except (MediaStreamError, asyncio.CancelledError):
                break
            pcm = _audio_frame_to_pcm(frame, sample_rate=TRANSCRIPTION_SAMPLE_RATE)
            if pcm:
                await self.remote_audio_queue.put(pcm)

    def _drain_remote_audio(self) -> None:
        while True:
            try:
                self.remote_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def initial_greeting(self) -> dict[str, Any]:
        return await self.capture_agent_reply()

    async def reply(self, caller_text: str) -> dict[str, Any]:
        self._drain_remote_audio()
        started_at = time.perf_counter()
        caller_pcm = await _synthesize_caller_audio(caller_text, self.config)
        if not self.audio_track:
            raise RuntimeError("WebRTC audio track is not connected.")
        self.audio_track.enqueue_pcm(caller_pcm)
        result = await self.capture_agent_reply()
        result["elapsed_ms"] = (time.perf_counter() - started_at) * 1000.0
        return result

    async def capture_agent_reply(self) -> dict[str, Any]:
        started_at = time.perf_counter()
        pcm = bytearray()
        speech_started = False
        silence_seconds = 0.0
        speech_seconds = 0.0
        deadline = time.monotonic() + self.config.voice_reply_timeout_seconds

        while time.monotonic() < deadline:
            try:
                chunk = await asyncio.wait_for(self.remote_audio_queue.get(), timeout=0.25)
            except TimeoutError:
                continue

            duration = (len(chunk) / 2) / TRANSCRIPTION_SAMPLE_RATE
            is_speech = _pcm_rms(chunk) >= self.config.voice_rms_threshold
            if is_speech:
                speech_started = True
                silence_seconds = 0.0
                speech_seconds += duration
            elif speech_started:
                silence_seconds += duration

            if speech_started:
                pcm.extend(chunk)
                if (
                    speech_seconds >= 0.25
                    and silence_seconds >= self.config.voice_silence_seconds
                ):
                    break

        text = ""
        if pcm:
            text = await _transcribe_pcm_with_nvidia(
                bytes(pcm),
                url=self.config.transcribe_asr_url,
                timeout_seconds=self.config.bot_request_timeout_seconds,
            )
        return {
            "text": text,
            "tool_calls": [],
            "ended": False,
            "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
        }

    async def close(self) -> None:
        if self._ping_task:
            self._ping_task.cancel()
        for task in self._reader_tasks:
            task.cancel()
        if self.pc:
            await self.pc.close()


class VoiceBotAgentClient:
    """Client for the already-running Pipecat voice bot on localhost:7860."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model: str | None = None
        self.initial_agent_text = ""
        self._session: LiveWebRTCSession | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=self.config.bot_request_timeout_seconds + 10)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError("Timed out waiting for live voice bot operation.") from exc

    def reset(self, scenario: Mapping[str, Any], *, today: date) -> None:
        if self._session:
            self._run(self._session.close())
        self.initial_agent_text = ""
        self._session = LiveWebRTCSession(self.config)
        self._run(self._session.connect(scenario, today=today))
        greeting = self._run(self._session.initial_greeting())
        self.initial_agent_text = str(greeting.get("text") or "").strip()

    def reply(self, caller_text: str) -> dict[str, Any]:
        if not self._session:
            raise RuntimeError("Voice bot session has not been reset.")
        return self._run(self._session.reply(caller_text))

    def close(self) -> None:
        if self._session:
            self._run(self._session.close())
            self._session = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class RunningVoiceBot:
    """Uses an already-running Pipecat voice bot; never starts a different agent."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.client: VoiceBotAgentClient | None = None

    def __enter__(self) -> VoiceBotAgentClient:
        self._assert_voice_bot_is_running()
        self.client = VoiceBotAgentClient(self.config)
        return self.client

    def __exit__(self, *args) -> None:
        if self.client:
            self.client.close()

    def _assert_voice_bot_is_running(self) -> None:
        try:
            status = _json_get(
                f"{self.config.bot_url}/status",
                timeout_seconds=min(5.0, self.config.bot_request_timeout_seconds),
            )
        except (HTTPError, OSError, URLError) as exc:
            raise RuntimeError(
                f"No running Pipecat voice bot was reachable at {self.config.bot_url}. "
                "Start it in another terminal with `uv run bot.py`; eval_runner.py will not "
                "start `bot.py` or use BOT_EVAL_SERVER."
            ) from exc

        transports = status.get("transports", [])
        if status.get("status") != "ready" or "webrtc" not in transports:
            raise RuntimeError(
                f"{self.config.bot_url} is reachable, but it does not look like a Pipecat "
                f"WebRTC voice runner. /status returned: {status!r}"
            )


def _scenario_ids(raw_ids: Iterable[str]) -> list[str]:
    scenario_ids: list[str] = []
    for raw in raw_ids:
        scenario_ids.extend(part.strip() for part in raw.split(",") if part.strip())
    return scenario_ids


def _select_scenarios(ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    scenarios = list(SCENARIOS)
    if ids:
        by_id = {scenario["id"]: scenario for scenario in scenarios}
        missing = [scenario_id for scenario_id in ids if scenario_id not in by_id]
        if missing:
            available = ", ".join(sorted(by_id))
            raise SystemExit(f"Unknown scenario id(s): {', '.join(missing)}. Available: {available}")
        scenarios = [by_id[scenario_id] for scenario_id in ids]
    if limit is not None:
        scenarios = scenarios[: max(0, limit)]
    return scenarios


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"Could not find JSON object in judge response: {text!r}")
    return json.loads(text[start : end + 1])


def _extract_judge_verdict(text: str) -> dict[str, Any]:
    try:
        return _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        passed_match = re.search(r'"?passed"?\s*:\s*(true|false)', text, re.IGNORECASE)
        reason_match = re.search(
            r'"?reason"?\s*:\s*(?:"(?P<quoted>[^"]+)"|(?P<plain>[^\n}]+))',
            text,
            re.IGNORECASE,
        )
        if passed_match:
            reason = ""
            if reason_match:
                reason = (reason_match.group("quoted") or reason_match.group("plain") or "").strip()
                reason = reason.rstrip(",")
            return {
                "passed": passed_match.group(1).lower() == "true",
                "reason": reason or "Judge returned a malformed reason field.",
            }
        raise


def _safe_json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {"_raw": value}
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _chat_completion(
    client: OpenAI,
    *,
    disable_thinking: bool,
    **kwargs: Any,
):
    if not disable_thinking:
        return client.chat.completions.create(**kwargs)

    try:
        return client.chat.completions.create(
            **kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except BadRequestError as exc:
        message = str(exc).lower()
        if "chat_template_kwargs" in message or "extra" in message or "unknown" in message:
            return client.chat.completions.create(**kwargs)
        raise


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        value = part.get("text") or part.get("content")
        return value if isinstance(value, str) else ""
    value = getattr(part, "text", None) or getattr(part, "content", None)
    return value if isinstance(value, str) else ""


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(_content_part_text(part) for part in content).strip()

    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = model_extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _response_debug(response: Any) -> str:
    try:
        choice = response.choices[0]
        message = choice.message
        payload = message.model_dump(exclude_none=True)
        return f"finish_reason={getattr(choice, 'finish_reason', None)!r}, message={payload!r}"
    except Exception:
        return repr(response)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = rank - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _round_ms(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value))


def _format_ms(value: Any) -> str:
    return "n/a" if value is None else f"{value}ms"


def caller_turn(
    client: OpenAI,
    model: str,
    persona: str,
    transcript: list[dict[str, str]],
    *,
    temperature: float,
    disable_thinking: bool,
) -> str:
    system_prompt = (
        f"{persona}\n\n"
        "You are the CALLER on a phone call with a dental front-desk agent. "
        "Reply in one short, natural turn. Stay in character. "
        "Do not solve the agent's work for them. "
        "Only write [END] after the agent has actually met your goal or told you to seek "
        "emergency care. Do not write [END] in your first turn, and do not write [END] when "
        "you are giving details, correcting information, or approving an action."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for turn in transcript:
        if turn["speaker"] == "agent":
            messages.append({"role": "user", "content": turn["text"]})
        elif turn["speaker"] == "caller":
            messages.append({"role": "assistant", "content": turn["text"]})

    for _ in range(2):
        response = _chat_completion(
            client,
            disable_thinking=disable_thinking,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=260,
        )
        text = _message_text(response.choices[0].message)
        if text:
            return text

        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous reply was empty. Reply now as the caller in one short natural "
                    "turn, using normal message content."
                ),
            }
        )

    raise RuntimeError(f"Caller simulator returned empty content: {_response_debug(response)}")


def agent_reply(
    client: Any,
    model: str,
    agent_messages: list[dict[str, Any]],
    *,
    temperature: float,
    disable_thinking: bool,
    caller_text: str | None = None,
    max_tool_rounds: int = 6,
) -> dict[str, Any]:
    if caller_text is not None and hasattr(client, "reply"):
        return client.reply(caller_text)
    raise RuntimeError(
        "eval_runner.py is configured for live voice evaluation only. The agent client must "
        "be the running Pipecat bot behind --bot-url, not an in-process LLM/tool harness."
    )


def _build_judge_messages(
    scenario: Mapping[str, Any],
    transcript: list[dict[str, str]],
    tool_calls: list[dict[str, Any]],
    *,
    today: date,
) -> list[dict[str, str]]:
    conversation = "\n".join(f'{turn["speaker"]}: {turn["text"]}' for turn in transcript)
    tool_log = "\n".join(json.dumps(call, sort_keys=True) for call in tool_calls) or "(none)"
    prompt = (
        f"Scenario id: {scenario['id']}\n"
        f"Bot prompt date: {today.isoformat()} ({today.strftime('%A, %B %d, %Y')})\n"
        f"Success criteria: {scenario['criteria']}\n\n"
        f"Transcript:\n{conversation}\n\n"
        f"Tool calls made by the AGENT:\n{tool_log}\n\n"
        "This is a live voice eval against the running Pipecat bot. Internal tool calls may "
        "not be observable unless the bot exposes them over the voice transport; when the "
        "tool log is empty, judge from the spoken transcript and do not fail solely because "
        "tool telemetry is unavailable.\n\n"
        "Judge the user outcome first: pass when the agent gave the caller the answer, action, "
        "or safety direction the caller was looking for, and there is no material factual, "
        "safety, privacy, or task-completion error.\n\n"
        "For live voice transcripts, tolerate minor ASR/TTS artifacts, homophones, punctuation "
        "loss, numeric formatting differences, filler, benign repetition, and awkward wording. "
        "Normalize likely transcription artifacts when the intended answer is clear from the "
        "surrounding context or the caller's response. For example, an office-hours answer "
        "should not fail just because ASR rendered part of '5:00 PM' oddly, if the answer is "
        "still understandable as Monday-Friday 8:00 AM-5:00 PM.\n\n"
        "Treat style preferences such as concision as secondary unless the scenario is "
        "specifically testing that behavior or the extra wording materially confuses, "
        "misleads, or frustrates the caller. Do not fail only because the answer could have "
        "been phrased more elegantly.\n\n"
        "Still fail when the agent omits the requested answer or action, gives a materially "
        "wrong date/time/status, claims an appointment was booked/canceled/rescheduled without "
        "adequate evidence, provides unsafe medical guidance, misses an emergency escalation, "
        "reveals inappropriate private information, or the transcript is too garbled to know "
        "whether the caller got what they needed.\n\n"
        "Did the AGENT satisfy the caller's goal and the material parts of the criteria? "
        "Reply ONLY as JSON: "
        '{"passed": true/false, "reason": "<one sentence>"}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a user-outcome evaluator for a dental front-desk voice agent. "
                "Score whether the caller got what they needed while preserving strictness "
                "for safety, privacy, factual accuracy, and task completion."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def judge(
    client: OpenAI,
    model: str,
    scenario: Mapping[str, Any],
    transcript: list[dict[str, str]],
    tool_calls: list[dict[str, Any]],
    *,
    today: date,
    temperature: float,
    disable_thinking: bool,
) -> dict[str, Any]:
    messages = _build_judge_messages(scenario, transcript, tool_calls, today=today)
    content = ""
    last_error: Exception | None = None
    for _ in range(2):
        response = _chat_completion(
            client,
            disable_thinking=disable_thinking,
            model=model,
            temperature=temperature,
            max_tokens=500,
            messages=messages,
        )
        content = _message_text(response.choices[0].message)
        if not content:
            last_error = RuntimeError(f"Judge returned empty content: {_response_debug(response)}")
        else:
            try:
                verdict = _extract_judge_verdict(content)
                return {
                    "passed": bool(verdict.get("passed")),
                    "reason": str(verdict.get("reason", "No reason provided.")),
                }
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc

        messages.append(
            {
                "role": "assistant",
                "content": content or "INVALID",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    'Return only valid JSON now, exactly like: {"passed": false, '
                    '"reason": "one sentence"}'
                ),
            }
        )

    if last_error:
        return {
            "passed": False,
            "reason": f"Judge output was not valid JSON: {last_error}",
        }
    return {
        "passed": False,
        "reason": "Judge output was empty or invalid.",
    }


def run_scenario(
    scenario: Mapping[str, Any],
    config: EvalConfig,
    *,
    agent_client: Any,
    caller_client: OpenAI,
    judge_client: OpenAI,
) -> dict[str, Any]:
    if hasattr(agent_client, "reset"):
        agent_client.reset(scenario, today=config.today)
    transcript: list[dict[str, str]] = []
    initial_agent_text = str(getattr(agent_client, "initial_agent_text", "") or "").strip()
    if initial_agent_text:
        transcript.append({"speaker": "agent", "text": initial_agent_text})
    tool_calls: list[dict[str, Any]] = []
    agent_reply_ms: list[int] = []
    started_at = time.perf_counter()
    agent_messages: list[dict[str, Any]] = []

    for _ in range(config.max_turns):
        caller_text = caller_turn(
            caller_client,
            config.caller_model,
            scenario["persona"],
            transcript,
            temperature=config.caller_temperature,
            disable_thinking=config.disable_thinking,
        )
        caller_done = "[END]" in caller_text
        caller_text = caller_text.replace("[END]", "").strip()
        if caller_text:
            transcript.append({"speaker": "caller", "text": caller_text})
        if caller_done and not caller_text:
            break

        agent_messages.append({"role": "user", "content": caller_text})
        reply = agent_reply(
            agent_client,
            config.agent_model,
            agent_messages,
            temperature=config.agent_temperature,
            disable_thinking=config.disable_thinking,
            caller_text=caller_text,
        )
        agent_reply_ms.append(_round_ms(reply["elapsed_ms"]) or 0)
        tool_calls.extend(reply["tool_calls"])
        if reply["text"]:
            transcript.append({"speaker": "agent", "text": reply["text"]})
        if reply["ended"] or caller_done:
            break

    verdict = judge(
        judge_client,
        config.judge_model,
        scenario,
        transcript,
        tool_calls,
        today=config.today,
        temperature=config.judge_temperature,
        disable_thinking=config.disable_thinking,
    )
    duration_ms = (time.perf_counter() - started_at) * 1000.0
    return {
        "id": scenario["id"],
        "passed": verdict["passed"],
        "reason": verdict["reason"],
        "turns": len(transcript),
        "duration_ms": _round_ms(duration_ms),
        "p95_agent_reply_ms": _round_ms(_percentile([float(v) for v in agent_reply_ms], 95)),
        "agent_reply_ms": agent_reply_ms,
        "tool_calls": tool_calls,
        "transcript": transcript,
    }


def latency_summary(path: Path) -> dict[str, Any]:
    ttfa_values: list[float] = []
    ttla_values: list[float] = []
    legacy_values: list[float] = []
    if not path.exists():
        return {
            "path": str(path),
            "completed_turns": 0,
            "p95_ttfa_ms": None,
            "p95_ttla_ms": None,
            "p95_legacy_latency_ms": None,
        }

    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "voice_latency_turn":
                if record.get("ttfa_ms") is not None:
                    ttfa_values.append(float(record["ttfa_ms"]))
                if record.get("ttla_ms") is not None:
                    ttla_values.append(float(record["ttla_ms"]))
            elif record.get("latency_ms") is not None:
                legacy_values.append(float(record["latency_ms"]))

    return {
        "path": str(path),
        "completed_turns": len(ttla_values),
        "p95_ttfa_ms": _round_ms(_percentile(ttfa_values, 95)),
        "p95_ttla_ms": _round_ms(_percentile(ttla_values, 95)),
        "p95_legacy_latency_ms": _round_ms(_percentile(legacy_values, 95)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.list:
        for scenario in SCENARIOS:
            print(scenario["id"])
        return

    config = _build_config(args)
    scenarios = _select_scenarios(_scenario_ids(args.scenario), args.limit)
    if not scenarios:
        raise SystemExit("No scenarios selected.")

    caller_client = _client(config.caller_base_url, "CALLER")
    judge_client = _client(config.judge_base_url, "JUDGE")

    results = []
    local_agent_model = config.agent_model
    try:
        with RunningVoiceBot(config) as agent_client:
            for index, scenario in enumerate(scenarios, start=1):
                print(f"[{index}/{len(scenarios)}] {scenario['id']}")
                results.append(
                    run_scenario(
                        scenario,
                        config,
                        agent_client=agent_client,
                        caller_client=caller_client,
                        judge_client=judge_client,
                    )
                )
                local_agent_model = agent_client.model or local_agent_model
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    passed = sum(1 for result in results if result["passed"])
    agent_reply_ms = [
        float(value) for result in results for value in result.get("agent_reply_ms", [])
    ]
    live_latency = latency_summary(config.latency_path)
    p95_latency_ms = live_latency["p95_ttla_ms"] or live_latency["p95_legacy_latency_ms"]
    output = {
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": time.time(),
        "mode": "voice",
        "agent_source": f"running Pipecat WebRTC voice bot at {config.bot_url}",
        "agent_model": local_agent_model,
        "caller_model": config.caller_model,
        "judge_model": config.judge_model,
        "caller_tts_provider": config.caller_tts_provider,
        "transcribe_asr_url": config.transcribe_asr_url,
        "scenario_count": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3),
        "p95_agent_reply_ms": _round_ms(_percentile(agent_reply_ms, 95)),
        "p95_latency_ms": p95_latency_ms,
        "live_latency": live_latency,
        "scenarios": results,
    }
    write_json(config.output_path, output)
    append_jsonl(
        config.runs_path,
        {
            "timestamp": output["timestamp"],
            "pass_rate": output["pass_rate"],
            "p95_agent_reply_ms": output["p95_agent_reply_ms"],
            "p95_latency_ms": output["p95_latency_ms"],
            "live_p95_ttfa_ms": live_latency["p95_ttfa_ms"],
            "live_p95_ttla_ms": live_latency["p95_ttla_ms"],
            "scenario_count": output["scenario_count"],
        },
    )

    print(
        f"{passed}/{len(results)} passed · pass_rate={output['pass_rate']:.3f} · "
        f"voice_p95_agent={_format_ms(output['p95_agent_reply_ms'])} · "
        f"live_p95_ttla={_format_ms(live_latency['p95_ttla_ms'])}"
    )
    failures = [result for result in results if not result["passed"]]
    for failure in failures:
        print(f"FAIL {failure['id']}: {failure['reason']}")


if __name__ == "__main__":
    main()
