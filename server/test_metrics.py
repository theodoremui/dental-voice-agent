import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipecat.frames.frames import (  # noqa: E402
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402

from metrics import LatencyLogger, LatencyStats  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.wall_now = 1779990000.123

    def monotonic(self):
        return self.now

    def wall(self):
        return self.wall_now


async def _feed(processor: LatencyLogger, *frames):
    for frame in frames:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)


def _audio_frame() -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=b"\0" * 320, sample_rate=16000, num_channels=1)


def _make_logger(tmp_path, clock: FakeClock, *, window: int = 10):
    path = tmp_path / "latency.jsonl"
    processor = LatencyLogger(
        path=path,
        enabled=True,
        log_every=1,
        window=window,
        transport_label="test",
        call_sid="CA123",
        stream_sid="MZ123",
        from_number="+14155551234",
        monotonic_time=clock.monotonic,
        wall_time=clock.wall,
    )

    async def push_frame(frame, direction):
        return None

    processor.push_frame = push_frame
    return processor, path


def _records(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_latency_stats_exact_percentiles():
    stats = LatencyStats(window=100)

    stats.add(ttfa_ms=100, ttla_ms=1000)
    assert stats.p50_ttfa_ms == 100
    assert stats.p95_ttfa_ms == 100

    stats.add(ttfa_ms=200, ttla_ms=3000)
    assert stats.p50_ttfa_ms == 150
    assert stats.p95_ttfa_ms == 195
    assert stats.p50_ttla_ms == 2000
    assert stats.p95_ttla_ms == 2900

    stats = LatencyStats(window=100)
    for value in range(1, 101):
        stats.add(ttfa_ms=value, ttla_ms=value * 10)

    assert stats.p50_ttfa_ms == pytest.approx(50.5)
    assert stats.p95_ttfa_ms == pytest.approx(95.05)


def test_latency_stats_rolling_window_drops_oldest_samples():
    stats = LatencyStats(window=2)
    stats.add(ttfa_ms=100, ttla_ms=1000)
    stats.add(ttfa_ms=200, ttla_ms=2000)
    stats.add(ttfa_ms=300, ttla_ms=3000)

    assert stats.completed_turns == 3
    assert stats.p50_ttfa_ms == 250
    assert stats.p95_ttfa_ms == 295


def test_completed_turn_records_ttfa_ttla_and_metadata(tmp_path):
    clock = FakeClock()
    processor, path = _make_logger(tmp_path, clock)

    clock.now = 10.0
    asyncio.run(_feed(processor, UserStoppedSpeakingFrame()))
    clock.now = 10.842
    asyncio.run(_feed(processor, _audio_frame()))
    clock.now = 12.38
    asyncio.run(_feed(processor, BotStoppedSpeakingFrame()))

    records = _records(path)
    assert records == [
        {
            "type": "voice_latency_turn",
            "ts": 1779990000.123,
            "turn_id": 1,
            "transport": "test",
            "call_sid": "CA123",
            "stream_sid": "MZ123",
            "from_number": "+14155551234",
            "ttfa_ms": 842,
            "ttla_ms": 2380,
            "completed_turns": 1,
            "p50_ttfa_ms": 842,
            "p95_ttfa_ms": 842,
            "p50_ttla_ms": 2380,
            "p95_ttla_ms": 2380,
        }
    ]


def test_multiple_audio_chunks_count_only_first_for_ttfa(tmp_path):
    clock = FakeClock()
    processor, path = _make_logger(tmp_path, clock)

    clock.now = 1.0
    asyncio.run(_feed(processor, UserStoppedSpeakingFrame()))
    clock.now = 1.2
    asyncio.run(_feed(processor, _audio_frame()))
    clock.now = 1.9
    asyncio.run(_feed(processor, _audio_frame()))
    clock.now = 2.0
    asyncio.run(_feed(processor, BotStoppedSpeakingFrame()))

    [record] = _records(path)
    assert record["ttfa_ms"] == 200
    assert record["ttla_ms"] == 1000


def test_interrupted_turn_is_logged_incomplete_and_excluded(tmp_path):
    clock = FakeClock()
    processor, path = _make_logger(tmp_path, clock)

    async def noop():
        return None

    processor._start_interruption = noop
    processor.stop_all_metrics = noop

    clock.now = 3.0
    asyncio.run(_feed(processor, UserStoppedSpeakingFrame()))
    clock.now = 3.4
    asyncio.run(_feed(processor, _audio_frame()))
    clock.now = 3.6
    asyncio.run(_feed(processor, InterruptionFrame()))

    [record] = _records(path)
    assert record["type"] == "voice_latency_turn_incomplete"
    assert record["status"] == "interrupted"
    assert record["ttfa_ms"] == 400
    assert record["ttla_ms"] is None
    assert record["completed_turns"] == 0
    assert record["excluded_from_percentiles"] is True
    assert processor.stats.completed_turns == 0


def test_initial_bot_greeting_without_user_stop_is_ignored(tmp_path):
    clock = FakeClock()
    processor, path = _make_logger(tmp_path, clock)

    clock.now = 1.0
    asyncio.run(_feed(processor, _audio_frame()))
    clock.now = 1.5
    asyncio.run(_feed(processor, BotStoppedSpeakingFrame()))

    assert _records(path) == []
    assert processor.stats.completed_turns == 0


def test_back_to_back_turns_each_produce_one_completed_record(tmp_path):
    clock = FakeClock()
    processor, path = _make_logger(tmp_path, clock)

    clock.now = 1.0
    asyncio.run(_feed(processor, UserStoppedSpeakingFrame()))
    clock.now = 1.5
    asyncio.run(_feed(processor, _audio_frame()))
    clock.now = 2.0
    asyncio.run(_feed(processor, BotStoppedSpeakingFrame()))

    clock.now = 4.0
    asyncio.run(_feed(processor, UserStoppedSpeakingFrame()))
    clock.now = 4.25
    asyncio.run(_feed(processor, _audio_frame()))
    clock.now = 5.0
    asyncio.run(_feed(processor, BotStoppedSpeakingFrame()))

    records = _records(path)
    assert [record["type"] for record in records] == ["voice_latency_turn", "voice_latency_turn"]
    assert [record["turn_id"] for record in records] == [1, 2]
    assert [record["completed_turns"] for record in records] == [1, 2]
    assert [record["ttfa_ms"] for record in records] == [500, 250]
