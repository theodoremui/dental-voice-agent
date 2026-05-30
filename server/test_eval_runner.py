import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_runner  # noqa: E402


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _config(tmp_path):
    return eval_runner.EvalConfig(
        agent_model="agent",
        caller_model="caller",
        judge_model="judge",
        agent_base_url="http://example.test/v1",
        caller_base_url="http://example.test/v1",
        judge_base_url="http://example.test/v1",
        max_turns=1,
        output_path=tmp_path / "results.json",
        runs_path=tmp_path / "runs.jsonl",
        latency_path=tmp_path / "latency.jsonl",
        today=date(2026, 5, 30),
        agent_temperature=0.3,
        caller_temperature=0.7,
        judge_temperature=0.0,
        disable_thinking=True,
    )


def test_caller_end_marker_does_not_skip_agent_turn(monkeypatch, tmp_path):
    agent_calls = []

    def fake_caller_turn(*args, **kwargs):
        return "Please book a cleaning next Tuesday at 2:30 PM. [END]"

    def fake_agent_reply(*args, **kwargs):
        agent_calls.append(args)
        return {
            "text": "I can help with that.",
            "tool_calls": [],
            "ended": False,
            "elapsed_ms": 12.3,
        }

    def fake_judge(*args, **kwargs):
        return {"passed": False, "reason": "not relevant"}

    monkeypatch.setattr(eval_runner, "caller_turn", fake_caller_turn)
    monkeypatch.setattr(eval_runner, "agent_reply", fake_agent_reply)
    monkeypatch.setattr(eval_runner, "judge", fake_judge)

    result = eval_runner.run_scenario(
        {"id": "x", "persona": "caller", "criteria": "criteria"},
        _config(tmp_path),
        agent_client=object(),
        caller_client=object(),
        judge_client=object(),
    )

    assert len(agent_calls) == 1
    assert result["transcript"] == [
        {"speaker": "caller", "text": "Please book a cleaning next Tuesday at 2:30 PM."},
        {"speaker": "agent", "text": "I can help with that."},
    ]


def test_running_voice_bot_uses_existing_webrtc_runner(monkeypatch, tmp_path):
    requests = []

    class FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_client = FakeClient()

    def fake_urlopen(url, timeout):
        requests.append((url, timeout))
        return _FakeResponse({"status": "ready", "transports": ["webrtc", "daily"]})

    monkeypatch.setattr(eval_runner, "urlopen", fake_urlopen)
    monkeypatch.setattr(eval_runner, "VoiceBotAgentClient", lambda config: fake_client)

    with eval_runner.RunningVoiceBot(_config(tmp_path)) as client:
        assert client is fake_client

    assert requests == [("http://localhost:7860/status", 5.0)]
    assert fake_client.closed is True


def test_judge_prompt_prioritizes_user_outcome_for_voice_artifacts():
    messages = eval_runner._build_judge_messages(
        {"id": "office_hours", "criteria": "Agent answered office hours concisely."},
        [
            {
                "speaker": "caller",
                "text": "Hi, I just need your office hours.",
            },
            {
                "speaker": "agent",
                "text": "Our office hours are Monday through Friday, eight oh AM to five dollo PM.",
            },
            {
                "speaker": "caller",
                "text": "No, that's all I needed. Thank you!",
            },
        ],
        [],
        today=date(2026, 5, 30),
    )

    prompt = messages[1]["content"]

    assert "Judge the user outcome first" in prompt
    assert "tolerate minor ASR/TTS artifacts" in prompt
    assert "5:00 PM" in prompt
    assert "style preferences such as concision as secondary" in prompt
    assert "Still fail when" in prompt
