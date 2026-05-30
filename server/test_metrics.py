import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from metrics import VoiceLatencyTracker, VoiceStageLatencyTracker, nearest_rank_percentile


class SequenceClock:
    def __init__(self, *values: float):
        self._values = list(values)

    def __call__(self) -> float:
        if not self._values:
            raise AssertionError("clock exhausted")
        return self._values.pop(0)


def audio(size: int = 4) -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=b"x" * size, sample_rate=24000, num_channels=1)


class VoiceLatencyTrackerTests(unittest.TestCase):
    def test_complete_turn_records_ttfa_and_ttla(self):
        events = []
        summaries = []
        tracker = VoiceLatencyTracker(
            event_writer=events.append,
            summary_writer=summaries.append,
            monotonic_clock=SequenceClock(10.0, 10.5, 11.25),
            wall_clock=SequenceClock(1770000000.123),
        )

        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(3), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(5), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(
            events,
            [
                {
                    "ts": 1770000000.123,
                    "turn_id": 1,
                    "status": "complete",
                    "audio_frames": 2,
                    "audio_bytes": 8,
                    "ttfa_ms": 500,
                    "ttla_ms": 1250,
                    "audio_duration_ms": 750,
                }
            ],
        )
        self.assertEqual(
            summaries,
            [
                {
                    "completed_turns": 1,
                    "ttfa_p50_ms": 500,
                    "ttfa_p95_ms": 500,
                    "ttla_p50_ms": 1250,
                    "ttla_p95_ms": 1250,
                }
            ],
        )

    def test_bot_speaking_frames_do_not_finalize_multi_segment_response(self):
        events = []
        tracker = VoiceLatencyTracker(
            event_writer=events.append,
            monotonic_clock=SequenceClock(1.0, 1.2, 1.8),
            wall_clock=SequenceClock(2.0),
        )

        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)

        self.assertEqual(events, [])

        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "complete")
        self.assertEqual(events[0]["audio_frames"], 2)

    def test_greeting_audio_without_user_turn_is_ignored(self):
        events = []
        tracker = VoiceLatencyTracker(
            event_writer=events.append,
            monotonic_clock=SequenceClock(5.0, 5.3),
            wall_clock=SequenceClock(6.0),
        )

        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(events, [])

        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["ttfa_ms"], 300)

    def test_interruption_records_incomplete_turn_without_summary(self):
        events = []
        summaries = []
        tracker = VoiceLatencyTracker(
            event_writer=events.append,
            summary_writer=summaries.append,
            monotonic_clock=SequenceClock(1.0, 1.4),
            wall_clock=SequenceClock(2.0),
        )

        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(events[0]["status"], "interrupted")
        self.assertEqual(events[0]["ttfa_ms"], 400)
        self.assertEqual(summaries, [])
        self.assertEqual(tracker.completed_turns, 0)

    def test_new_user_turn_supersedes_previous_incomplete_turn(self):
        events = []
        summaries = []
        tracker = VoiceLatencyTracker(
            event_writer=events.append,
            summary_writer=summaries.append,
            monotonic_clock=SequenceClock(1.0, 1.2, 2.0, 2.4),
            wall_clock=SequenceClock(3.0, 4.0),
        )

        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(events[0]["status"], "superseded")
        self.assertEqual(events[1]["status"], "complete")
        self.assertEqual(summaries[0]["completed_turns"], 1)
        self.assertEqual(tracker.completed_turns, 1)

    def test_upstream_frames_are_ignored(self):
        events = []
        tracker = VoiceLatencyTracker(
            event_writer=events.append,
            monotonic_clock=SequenceClock(1.0),
            wall_clock=SequenceClock(2.0),
        )

        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(events, [])

    def test_nearest_rank_percentile(self):
        self.assertEqual(nearest_rank_percentile([100, 200, 300, 400], 50), 200)
        self.assertEqual(nearest_rank_percentile([100, 200, 300, 400], 95), 400)
        self.assertEqual(nearest_rank_percentile([30, 10, 20], 50), 20)
        self.assertEqual(nearest_rank_percentile([30, 10, 20], 95), 30)


class VoiceStageLatencyTrackerTests(unittest.TestCase):
    def test_complete_turn_records_stage_offsets_without_payloads(self):
        events = []
        summaries = []
        tracker = VoiceStageLatencyTracker(
            event_writer=events.append,
            summary_writer=summaries.append,
            monotonic_clock=SequenceClock(1.0, 1.6, 1.8, 1.85, 2.2, 2.6, 3.0),
            wall_clock=SequenceClock(1770000000.456),
        )

        tracker.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(
            TranscriptionFrame("please book me", "caller", "ts", finalized=True),
            FrameDirection.DOWNSTREAM,
        )
        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMTextFrame("Sure"), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(3), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(5), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["status"], "complete")
        self.assertEqual(event["audio_frames"], 2)
        self.assertEqual(event["audio_bytes"], 8)
        self.assertEqual(
            event["stages_ms"],
            {
                "vad_started": 0,
                "vad_stopped": 600,
                "final_transcript": 800,
                "user_turn_stopped": 850,
                "llm_first_text_or_tool": 1200,
                "first_output_audio": 1600,
                "last_output_audio": 2000,
            },
        )
        self.assertEqual(event["true_ttfa_ms"], 1000)
        self.assertEqual(event["turn_ttfa_ms"], 750)
        self.assertNotIn("please book me", str(event))
        self.assertEqual(
            summaries,
            [
                {
                    "completed_turns": 1,
                    "true_ttfa_p50_ms": 1000,
                    "true_ttfa_p95_ms": 1000,
                    "turn_ttfa_p50_ms": 750,
                    "turn_ttfa_p95_ms": 750,
                }
            ],
        )

    def test_missing_stages_are_null_instead_of_crashing(self):
        events = []
        tracker = VoiceStageLatencyTracker(
            event_writer=events.append,
            monotonic_clock=SequenceClock(5.0),
            wall_clock=SequenceClock(6.0),
        )

        tracker.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["stages_ms"]["user_turn_stopped"], 0)
        self.assertIsNone(event["stages_ms"]["vad_started"])
        self.assertIsNone(event["stages_ms"]["first_output_audio"])
        self.assertIsNone(event["durations_ms"]["vad_stopped_to_first_output_audio"])
        self.assertIsNone(event["true_ttfa_ms"])
        self.assertEqual(event["audio_frames"], 0)

    def test_vad_upstream_downstream_duplicates_do_not_supersede_turn(self):
        events = []
        tracker = VoiceStageLatencyTracker(
            event_writer=events.append,
            monotonic_clock=SequenceClock(1.0, 1.6, 2.0),
            wall_clock=SequenceClock(3.0),
        )

        tracker.process_frame(
            VADUserStartedSpeakingFrame(),
            FrameDirection.UPSTREAM,
            observe_upstream=True,
        )
        tracker.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(
            VADUserStoppedSpeakingFrame(),
            FrameDirection.UPSTREAM,
            observe_upstream=True,
        )
        tracker.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(audio(), FrameDirection.DOWNSTREAM)
        tracker.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "complete")
        self.assertEqual(events[0]["true_ttfa_ms"], 400)
        self.assertEqual(events[0]["stages_ms"]["vad_started"], 0)
        self.assertEqual(events[0]["stages_ms"]["vad_stopped"], 600)


if __name__ == "__main__":
    unittest.main()
