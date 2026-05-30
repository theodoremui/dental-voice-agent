import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    CancelTaskFrame,
    EndFrame,
    EndTaskFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    InterruptionTaskFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StopFrame,
    StopTaskFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

EventWriter = Callable[[dict[str, Any]], None]
SummaryWriter = Callable[[dict[str, Any]], None]
Clock = Callable[[], float]


@dataclass
class _TurnLatency:
    turn_id: int
    stopped_speaking_at: float
    first_audio_at: float | None = None
    last_audio_at: float | None = None
    audio_frames: int = 0
    audio_bytes: int = 0


class VoiceLatencyTracker:
    """Tracks per-turn outbound audio latency from caller stop to emitted audio."""

    def __init__(
        self,
        *,
        event_writer: EventWriter,
        summary_writer: SummaryWriter | None = None,
        monotonic_clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
    ):
        self._event_writer = event_writer
        self._summary_writer = summary_writer
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._active_turn: _TurnLatency | None = None
        self._next_turn_id = 1
        self._completed_ttfa_ms: list[int] = []
        self._completed_ttla_ms: list[int] = []

    @property
    def completed_turns(self) -> int:
        return len(self._completed_ttfa_ms)

    def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if direction != FrameDirection.DOWNSTREAM:
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._start_turn()
        elif isinstance(frame, OutputAudioRawFrame):
            self._record_audio(frame)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._complete_turn()
        elif isinstance(frame, (InterruptionFrame, InterruptionTaskFrame)):
            self.flush("interrupted")
        elif isinstance(frame, (CancelFrame, CancelTaskFrame)):
            self.flush("cancelled")
        elif isinstance(frame, (EndFrame, EndTaskFrame, StopFrame, StopTaskFrame)):
            self.flush("ended")

    def flush(self, status: str) -> None:
        if self._active_turn is None:
            return

        event = self._build_event(self._active_turn, status)
        self._active_turn = None
        self._event_writer(event)

    def _start_turn(self) -> None:
        if self._active_turn is not None:
            self.flush("superseded")

        self._active_turn = _TurnLatency(
            turn_id=self._next_turn_id,
            stopped_speaking_at=self._monotonic_clock(),
        )
        self._next_turn_id += 1

    def _record_audio(self, frame: OutputAudioRawFrame) -> None:
        if self._active_turn is None:
            return

        now = self._monotonic_clock()
        if self._active_turn.first_audio_at is None:
            self._active_turn.first_audio_at = now

        self._active_turn.last_audio_at = now
        self._active_turn.audio_frames += 1
        self._active_turn.audio_bytes += len(frame.audio)

    def _complete_turn(self) -> None:
        if self._active_turn is None or self._active_turn.first_audio_at is None:
            return

        event = self._build_event(self._active_turn, "complete")
        self._active_turn = None
        self._completed_ttfa_ms.append(event["ttfa_ms"])
        self._completed_ttla_ms.append(event["ttla_ms"])
        self._event_writer(event)
        self._write_summary()

    def _build_event(self, turn: _TurnLatency, status: str) -> dict[str, Any]:
        event: dict[str, Any] = {
            "ts": self._wall_clock(),
            "turn_id": turn.turn_id,
            "status": status,
            "audio_frames": turn.audio_frames,
            "audio_bytes": turn.audio_bytes,
        }

        if turn.first_audio_at is not None and turn.last_audio_at is not None:
            event["ttfa_ms"] = round((turn.first_audio_at - turn.stopped_speaking_at) * 1000)
            event["ttla_ms"] = round((turn.last_audio_at - turn.stopped_speaking_at) * 1000)
            event["audio_duration_ms"] = round((turn.last_audio_at - turn.first_audio_at) * 1000)

        return event

    def _write_summary(self) -> None:
        if self._summary_writer is None:
            return

        self._summary_writer(
            {
                "completed_turns": self.completed_turns,
                "ttfa_p50_ms": nearest_rank_percentile(self._completed_ttfa_ms, 50),
                "ttfa_p95_ms": nearest_rank_percentile(self._completed_ttfa_ms, 95),
                "ttla_p50_ms": nearest_rank_percentile(self._completed_ttla_ms, 50),
                "ttla_p95_ms": nearest_rank_percentile(self._completed_ttla_ms, 95),
            }
        )


def nearest_rank_percentile(values: list[int], percentile: int) -> int:
    if not values:
        raise ValueError("nearest-rank percentile requires at least one value")

    sorted_values = sorted(values)
    index = ceil(percentile / 100 * len(sorted_values)) - 1
    return sorted_values[max(index, 0)]


class LatencyLogger(FrameProcessor):
    """Pipecat processor that emits voice TTFA/TTLA events and summaries."""

    def __init__(self, path: str = "latency.jsonl"):
        super().__init__()
        self._path = Path(path)
        self._tracker = VoiceLatencyTracker(
            event_writer=self._write_event,
            summary_writer=self._log_summary,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._tracker.process_frame(frame, direction)
        await self.push_frame(frame, direction)

    def _write_event(self, event: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")

    def _log_summary(self, summary: dict[str, int]) -> None:
        logger.info(
            "voice_latency completed_turns={} ttfa_p50_ms={} ttfa_p95_ms={} "
            "ttla_p50_ms={} ttla_p95_ms={}",
            summary["completed_turns"],
            summary["ttfa_p50_ms"],
            summary["ttfa_p95_ms"],
            summary["ttla_p50_ms"],
            summary["ttla_p95_ms"],
        )


STAGE_NAMES = (
    "vad_started",
    "vad_stopped",
    "final_transcript",
    "user_turn_stopped",
    "llm_first_text_or_tool",
    "first_output_audio",
    "last_output_audio",
)


@dataclass
class _StageTurn:
    turn_id: int
    vad_started: float | None = None
    vad_stopped: float | None = None
    final_transcript: float | None = None
    user_turn_stopped: float | None = None
    llm_first_text_or_tool: float | None = None
    first_output_audio: float | None = None
    last_output_audio: float | None = None
    audio_frames: int = 0
    audio_bytes: int = 0

    @property
    def first_observed_at(self) -> float | None:
        observed = [
            getattr(self, stage)
            for stage in STAGE_NAMES
            if getattr(self, stage) is not None
        ]
        return min(observed) if observed else None


class VoiceStageLatencyTracker:
    """Tracks compact per-turn stage timings without storing user text or tool payloads."""

    def __init__(
        self,
        *,
        event_writer: EventWriter,
        summary_writer: SummaryWriter | None = None,
        monotonic_clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
    ):
        self._event_writer = event_writer
        self._summary_writer = summary_writer
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._active_turn: _StageTurn | None = None
        self._next_turn_id = 1
        self._completed_turns = 0
        self._completed_true_ttfa_ms: list[int] = []
        self._completed_turn_ttfa_ms: list[int] = []

    @property
    def completed_turns(self) -> int:
        return self._completed_turns

    def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
        *,
        observe_upstream: bool = False,
        observe_turn: bool = True,
        observe_transcript: bool = True,
        observe_llm: bool = True,
        observe_audio: bool = True,
        finalize: bool = True,
    ) -> None:
        if direction == FrameDirection.UPSTREAM:
            if not observe_upstream:
                return
        elif direction != FrameDirection.DOWNSTREAM:
            return

        if observe_turn:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                self._handle_vad_started()
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self._set_stage_once(self._ensure_turn(), "vad_stopped")
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._set_stage_once(self._ensure_turn(), "user_turn_stopped")
            elif isinstance(frame, UserStartedSpeakingFrame):
                self._ensure_turn()

        if observe_transcript and isinstance(frame, TranscriptionFrame) and frame.finalized:
            self._set_stage_once(self._ensure_turn(), "final_transcript")

        if observe_llm:
            if isinstance(frame, LLMTextFrame) and frame.text.strip():
                turn = self._active_turn
                if turn is not None and turn.llm_first_text_or_tool is None:
                    self._set_stage_once(turn, "llm_first_text_or_tool")
            elif isinstance(frame, (FunctionCallsStartedFrame, FunctionCallInProgressFrame)):
                turn = self._active_turn
                if turn is not None and turn.llm_first_text_or_tool is None:
                    self._set_stage_once(turn, "llm_first_text_or_tool")

        if observe_audio and isinstance(frame, OutputAudioRawFrame):
            self._record_audio(frame)

        if not finalize:
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            self._complete_turn()
        elif isinstance(frame, (InterruptionFrame, InterruptionTaskFrame)):
            self.flush("interrupted")
        elif isinstance(frame, (CancelFrame, CancelTaskFrame)):
            self.flush("cancelled")
        elif isinstance(frame, (EndFrame, EndTaskFrame, StopFrame, StopTaskFrame)):
            self.flush("ended")

    def flush(self, status: str) -> None:
        if self._active_turn is None:
            return

        event = self._build_event(self._active_turn, status)
        self._active_turn = None
        self._event_writer(event)

    def _handle_vad_started(self) -> None:
        if self._active_turn is None:
            self._start_turn("vad_started")
            return

        if self._active_turn.vad_started is None:
            self._set_stage_once(self._active_turn, "vad_started")
            return

        if (
            self._active_turn.user_turn_stopped is None
            and self._active_turn.llm_first_text_or_tool is None
            and self._active_turn.first_output_audio is None
        ):
            return

        self.flush("superseded")
        self._start_turn("vad_started")

    def _start_turn(self, stage_name: str) -> None:
        if self._active_turn is not None:
            self.flush("superseded")

        turn = _StageTurn(turn_id=self._next_turn_id)
        self._set_stage_once(turn, stage_name)
        self._active_turn = turn
        self._next_turn_id += 1

    def _ensure_turn(self) -> _StageTurn:
        if self._active_turn is None:
            self._active_turn = _StageTurn(turn_id=self._next_turn_id)
            self._next_turn_id += 1
        return self._active_turn

    def _set_stage_once(self, turn: _StageTurn, stage_name: str) -> None:
        if getattr(turn, stage_name) is None:
            setattr(turn, stage_name, self._monotonic_clock())

    def _record_audio(self, frame: OutputAudioRawFrame) -> None:
        if self._active_turn is None:
            return

        now = self._monotonic_clock()
        if self._active_turn.first_output_audio is None:
            self._active_turn.first_output_audio = now
        self._active_turn.last_output_audio = now
        self._active_turn.audio_frames += 1
        self._active_turn.audio_bytes += len(frame.audio)

    def _complete_turn(self) -> None:
        if self._active_turn is None:
            return

        event = self._build_event(self._active_turn, "complete")
        self._active_turn = None
        self._completed_turns += 1

        true_ttfa_ms = event["durations_ms"]["vad_stopped_to_first_output_audio"]
        if true_ttfa_ms is not None:
            self._completed_true_ttfa_ms.append(true_ttfa_ms)

        turn_ttfa_ms = event["durations_ms"]["user_turn_stopped_to_first_output_audio"]
        if turn_ttfa_ms is not None:
            self._completed_turn_ttfa_ms.append(turn_ttfa_ms)

        self._event_writer(event)
        self._write_summary()

    def _build_event(self, turn: _StageTurn, status: str) -> dict[str, Any]:
        stages = {stage: getattr(turn, stage) for stage in STAGE_NAMES}
        offsets_ms = self._stage_offsets_ms(turn, stages)
        durations_ms = self._durations_ms(stages)

        return {
            "ts": self._wall_clock(),
            "turn_id": turn.turn_id,
            "status": status,
            "stages_ms": offsets_ms,
            "durations_ms": durations_ms,
            "true_ttfa_ms": durations_ms["vad_stopped_to_first_output_audio"],
            "turn_ttfa_ms": durations_ms["user_turn_stopped_to_first_output_audio"],
            "audio_frames": turn.audio_frames,
            "audio_bytes": turn.audio_bytes,
        }

    def _stage_offsets_ms(
        self, turn: _StageTurn, stages: dict[str, float | None]
    ) -> dict[str, int | None]:
        base = turn.first_observed_at
        if base is None:
            return dict.fromkeys(STAGE_NAMES)

        return {
            stage: round((timestamp - base) * 1000) if timestamp is not None else None
            for stage, timestamp in stages.items()
        }

    def _durations_ms(self, stages: dict[str, float | None]) -> dict[str, int | None]:
        return {
            "vad_started_to_vad_stopped": self._elapsed_ms(
                stages["vad_started"], stages["vad_stopped"]
            ),
            "vad_stopped_to_final_transcript": self._elapsed_ms(
                stages["vad_stopped"], stages["final_transcript"]
            ),
            "vad_stopped_to_user_turn_stopped": self._elapsed_ms(
                stages["vad_stopped"], stages["user_turn_stopped"]
            ),
            "vad_stopped_to_llm_first_text_or_tool": self._elapsed_ms(
                stages["vad_stopped"], stages["llm_first_text_or_tool"]
            ),
            "vad_stopped_to_first_output_audio": self._elapsed_ms(
                stages["vad_stopped"], stages["first_output_audio"]
            ),
            "vad_stopped_to_last_output_audio": self._elapsed_ms(
                stages["vad_stopped"], stages["last_output_audio"]
            ),
            "user_turn_stopped_to_first_output_audio": self._elapsed_ms(
                stages["user_turn_stopped"], stages["first_output_audio"]
            ),
            "llm_first_text_or_tool_to_first_output_audio": self._elapsed_ms(
                stages["llm_first_text_or_tool"], stages["first_output_audio"]
            ),
        }

    def _elapsed_ms(self, start: float | None, end: float | None) -> int | None:
        if start is None or end is None:
            return None
        return round((end - start) * 1000)

    def _write_summary(self) -> None:
        if self._summary_writer is None:
            return

        self._summary_writer(
            {
                "completed_turns": self.completed_turns,
                "true_ttfa_p50_ms": percentile_or_none(self._completed_true_ttfa_ms, 50),
                "true_ttfa_p95_ms": percentile_or_none(self._completed_true_ttfa_ms, 95),
                "turn_ttfa_p50_ms": percentile_or_none(self._completed_turn_ttfa_ms, 50),
                "turn_ttfa_p95_ms": percentile_or_none(self._completed_turn_ttfa_ms, 95),
            }
        )


def percentile_or_none(values: list[int], percentile: int) -> int | None:
    if not values:
        return None
    return nearest_rank_percentile(values, percentile)


class StageLatencyLogger(FrameProcessor):
    """Pipecat processor tap for stage-level latency JSONL logging."""

    def __init__(
        self,
        path: str = "stage_latency.jsonl",
        *,
        tracker: VoiceStageLatencyTracker | None = None,
        observe_upstream: bool = False,
        observe_turn: bool = True,
        observe_transcript: bool = True,
        observe_llm: bool = True,
        observe_audio: bool = True,
        finalize: bool = True,
    ):
        super().__init__()
        self._path = Path(path)
        self._tracker = tracker or VoiceStageLatencyTracker(
            event_writer=self._write_event,
            summary_writer=self._log_summary,
        )
        self._observe_upstream = observe_upstream
        self._observe_turn = observe_turn
        self._observe_transcript = observe_transcript
        self._observe_llm = observe_llm
        self._observe_audio = observe_audio
        self._finalize = finalize

    @property
    def tracker(self) -> VoiceStageLatencyTracker:
        return self._tracker

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._tracker.process_frame(
            frame,
            direction,
            observe_upstream=self._observe_upstream,
            observe_turn=self._observe_turn,
            observe_transcript=self._observe_transcript,
            observe_llm=self._observe_llm,
            observe_audio=self._observe_audio,
            finalize=self._finalize,
        )
        await self.push_frame(frame, direction)

    def _write_event(self, event: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")

    def _log_summary(self, summary: dict[str, Any]) -> None:
        logger.info(
            "voice_stage_latency completed_turns={} true_ttfa_p50_ms={} "
            "true_ttfa_p95_ms={} turn_ttfa_p50_ms={} turn_ttfa_p95_ms={}",
            summary["completed_turns"],
            summary["true_ttfa_p50_ms"],
            summary["true_ttfa_p95_ms"],
            summary["turn_ttfa_p50_ms"],
            summary["turn_ttfa_p95_ms"],
        )
