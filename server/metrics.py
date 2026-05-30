from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int, *, minimum: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(f"Ignoring invalid {name}={value!r}; using {default}")
        return default
    return max(minimum, parsed)


def _round_ms(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value))


@dataclass
class LatencyTurn:
    turn_id: int
    started_at: float
    first_audio_at: float | None = None
    last_audio_at: float | None = None
    status: str = "active"

    def mark_first_audio(self, now: float) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = now

    def mark_bot_stopped(self, now: float) -> bool:
        self.last_audio_at = now
        if self.first_audio_at is None:
            self.status = "missing_first_audio"
            return False
        self.status = "completed"
        return True

    @property
    def ttfa_ms(self) -> float | None:
        if self.first_audio_at is None:
            return None
        return (self.first_audio_at - self.started_at) * 1000.0

    @property
    def ttla_ms(self) -> float | None:
        if self.last_audio_at is None:
            return None
        return (self.last_audio_at - self.started_at) * 1000.0


class LatencyStats:
    """Exact rolling percentile stats for completed voice turns."""

    def __init__(self, window: int = 500):
        self._ttfa_ms: deque[float] = deque(maxlen=max(1, window))
        self._ttla_ms: deque[float] = deque(maxlen=max(1, window))
        self.completed_turns = 0

    def add(self, *, ttfa_ms: float, ttla_ms: float) -> None:
        self._ttfa_ms.append(ttfa_ms)
        self._ttla_ms.append(ttla_ms)
        self.completed_turns += 1

    @property
    def p50_ttfa_ms(self) -> float | None:
        return self._percentile(self._ttfa_ms, 50)

    @property
    def p95_ttfa_ms(self) -> float | None:
        return self._percentile(self._ttfa_ms, 95)

    @property
    def p50_ttla_ms(self) -> float | None:
        return self._percentile(self._ttla_ms, 50)

    @property
    def p95_ttla_ms(self) -> float | None:
        return self._percentile(self._ttla_ms, 95)

    def rounded_snapshot(self) -> dict[str, int | None]:
        return {
            "completed_turns": self.completed_turns,
            "p50_ttfa_ms": _round_ms(self.p50_ttfa_ms),
            "p95_ttfa_ms": _round_ms(self.p95_ttfa_ms),
            "p50_ttla_ms": _round_ms(self.p50_ttla_ms),
            "p95_ttla_ms": _round_ms(self.p95_ttla_ms),
        }

    @staticmethod
    def _percentile(values: deque[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]

        rank = (len(ordered) - 1) * (percentile / 100.0)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        fraction = rank - lower_index
        lower = ordered[lower_index]
        upper = ordered[upper_index]
        return lower + (upper - lower) * fraction


class LatencyLogger(FrameProcessor):
    """Pipecat processor that records per-turn outbound voice latency."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        enabled: bool | None = None,
        log_every: int | None = None,
        window: int | None = None,
        transport_label: str | None = None,
        call_sid: str | None = None,
        stream_sid: str | None = None,
        from_number: str | None = None,
        monotonic_time: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._enabled = _env_bool("LATENCY_METRICS_ENABLED", True) if enabled is None else enabled
        metrics_path = os.getenv("LATENCY_METRICS_PATH", "latency.jsonl") if path is None else path
        self._path = Path(metrics_path) if metrics_path else None
        self._log_every = (
            _env_int("LATENCY_METRICS_LOG_EVERY", 1, minimum=0)
            if log_every is None
            else max(0, log_every)
        )
        self._stats = LatencyStats(
            _env_int("LATENCY_METRICS_WINDOW", 500, minimum=1)
            if window is None
            else max(1, window)
        )
        self._transport_label = transport_label
        self._call_sid = call_sid
        self._stream_sid = stream_sid
        self._from_number = from_number
        self._monotonic_time = monotonic_time
        self._wall_time = wall_time
        self._active_turn: LatencyTurn | None = None
        self._next_turn_id = 1

    @property
    def stats(self) -> LatencyStats:
        return self._stats

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._enabled:
            self._observe_frame(frame)

        await self.push_frame(frame, direction)

    def _observe_frame(self, frame) -> None:
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._finish_incomplete("replaced")
            self._start_turn()
        elif isinstance(frame, OutputAudioRawFrame):
            if self._active_turn:
                self._active_turn.mark_first_audio(self._monotonic_time())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._active_turn:
                self._finish_bot_turn()
        elif isinstance(frame, InterruptionFrame):
            self._finish_incomplete("interrupted")
        elif isinstance(frame, CancelFrame):
            self._finish_incomplete("canceled", reason=frame.reason)
        elif isinstance(frame, EndFrame):
            self._finish_incomplete("ended", reason=frame.reason)

    def _start_turn(self) -> None:
        self._active_turn = LatencyTurn(
            turn_id=self._next_turn_id,
            started_at=self._monotonic_time(),
        )
        self._next_turn_id += 1

    def _finish_bot_turn(self) -> None:
        if not self._active_turn:
            return

        turn = self._active_turn
        self._active_turn = None
        completed = turn.mark_bot_stopped(self._monotonic_time())
        if not completed or turn.ttfa_ms is None or turn.ttla_ms is None:
            self._record_incomplete(turn)
            return

        self._stats.add(ttfa_ms=turn.ttfa_ms, ttla_ms=turn.ttla_ms)
        record = self._base_record("voice_latency_turn", turn)
        record.update(
            {
                "ttfa_ms": _round_ms(turn.ttfa_ms),
                "ttla_ms": _round_ms(turn.ttla_ms),
                **self._stats.rounded_snapshot(),
            }
        )
        self._write_record(record)
        self._log_completed(record)

    def _finish_incomplete(self, status: str, *, reason: Any | None = None) -> None:
        if not self._active_turn:
            return

        turn = self._active_turn
        self._active_turn = None
        turn.status = status
        self._record_incomplete(turn, reason=reason)

    def _record_incomplete(self, turn: LatencyTurn, *, reason: Any | None = None) -> None:
        record = self._base_record("voice_latency_turn_incomplete", turn)
        record.update(
            {
                "status": turn.status,
                "ttfa_ms": _round_ms(turn.ttfa_ms),
                "ttla_ms": _round_ms(turn.ttla_ms),
                "completed_turns": self._stats.completed_turns,
                "excluded_from_percentiles": True,
            }
        )
        if reason is not None:
            record["reason"] = str(reason)

        self._write_record(record)
        logger.info(
            "Voice latency turn {turn_id} {status}; excluded from percentiles "
            "(ttfa={ttfa_ms}ms ttla={ttla_ms}ms completed={completed_turns})",
            **record,
        )

    def _base_record(self, record_type: str, turn: LatencyTurn) -> dict[str, Any]:
        return {
            "type": record_type,
            "ts": self._wall_time(),
            "turn_id": turn.turn_id,
            "transport": self._transport_label,
            "call_sid": self._call_sid,
            "stream_sid": self._stream_sid,
            "from_number": self._from_number,
        }

    def _write_record(self, record: dict[str, Any]) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _log_completed(self, record: dict[str, Any]) -> None:
        if self._log_every == 0 or record["completed_turns"] % self._log_every != 0:
            return

        logger.info(
            "Voice latency turn {turn_id}: ttfa={ttfa_ms}ms ttla={ttla_ms}ms "
            "completed={completed_turns} p50_ttfa={p50_ttfa_ms}ms "
            "p95_ttfa={p95_ttfa_ms}ms p50_ttla={p50_ttla_ms}ms p95_ttla={p95_ttla_ms}ms",
            **record,
        )
