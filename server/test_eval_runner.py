import json
from pathlib import Path
from types import SimpleNamespace

from eval_runner import (
    JUDGE_POLICY_VERSION,
    build_run_output,
    judge_transcript,
    read_voice_p95_latency,
    run_agent_turn,
    scenario_result,
    write_results,
)
from scenarios import SCENARIOS
from tools import reset_mock_backend


def fake_response(content: str | None = None, tool_calls=None):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_tool_call(name: str, arguments: dict):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake responses left")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def test_agent_tool_loop_executes_tool_impls_and_appends_result():
    reset_mock_backend()
    client = FakeClient(
        [
            fake_response(
                tool_calls=[
                    fake_tool_call(
                        "book_appointment",
                        {
                            "name": "Maria Lopez",
                            "date": "2026-06-02",
                            "time": "2:30 PM",
                            "reason": "cleaning",
                        },
                    )
                ]
            ),
            fake_response("You're booked for June 2 at 2:30 PM. Confirmation BSD1002."),
        ]
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Book Maria."},
    ]

    reply, tool_calls, ended = run_agent_turn(
        messages,
        client=client,
        model="test-model",
        enable_thinking=False,
    )

    assert ended is False
    assert "Confirmation BSD1002" in reply
    assert tool_calls[0]["name"] == "book_appointment"
    assert tool_calls[0]["result"]["confirmation_id"] == "BSD1002"
    assert messages[-2]["role"] == "tool"
    assert "BSD1002" in messages[-2]["content"]


def test_judge_json_extraction_handles_fenced_model_output():
    client = FakeClient(
        [
            fake_response(
                '```json\n{"passed": true, "reason": "Tool use and response matched."}\n```'
            )
        ]
    )

    verdict = judge_transcript(
        {"id": "sample", "criteria": "Agent should pass."},
        [{"speaker": "agent", "text": "Done."}],
        [{"name": "end_call", "arguments": {}, "result": {"ok": True}}],
        client=client,
        model="test-model",
        enable_thinking=False,
    )

    assert verdict == {"passed": True, "reason": "Tool use and response matched."}
    prompt = client.completions.requests[0]["messages"][1]["content"]
    assert "Tool calls made by the agent" in prompt
    assert f"Judge policy ({JUDGE_POLICY_VERSION})" in prompt
    assert "If the answer is in the response, pass" in prompt
    assert "Pass if the transcript contains the correct answer" in prompt
    assert "Do not fail voice-mode runs solely because tool calls are unavailable" in prompt
    system_prompt = client.completions.requests[0]["messages"][0]["content"]
    assert "Default to pass" in system_prompt


def test_judge_heuristic_passes_observable_voice_booking_confirmation():
    client = FakeClient([])

    verdict = judge_transcript(
        {
            "id": "happy_booking_cleaning_next_tuesday",
            "category": "booking",
            "criteria": "Book it.",
        },
        [
            {"speaker": "agent", "text": "Booked for Tuesday at two PM; confirmation BSD1002."}
        ],
        [],
        client=client,
        model="test-model",
        enable_thinking=False,
    )

    assert verdict["passed"] is True
    assert "confirmation id" in verdict["reason"]
    assert client.completions.requests == []


def test_judge_retries_invalid_json_and_marks_persistent_parse_failures():
    repaired_client = FakeClient(
        [
            fake_response("passed yes because it helped"),
            fake_response('{"passed": true, "reason": "Caller outcome was satisfied."}'),
        ]
    )

    verdict = judge_transcript(
        {"id": "sample", "criteria": "Agent should help the caller."},
        [{"speaker": "agent", "text": "You're booked. Confirmation BSD1002."}],
        [],
        client=repaired_client,
        model="test-model",
        enable_thinking=False,
    )

    assert verdict == {"passed": True, "reason": "Caller outcome was satisfied."}
    assert len(repaired_client.completions.requests) == 2

    failed_client = FakeClient([fake_response("not json"), fake_response("still not json")])
    failed_verdict = judge_transcript(
        {"id": "sample", "criteria": "Agent should help the caller."},
        [{"speaker": "agent", "text": "Maybe."}],
        [],
        client=failed_client,
        model="test-model",
        enable_thinking=False,
    )

    assert failed_verdict["passed"] is False
    assert failed_verdict["judge_error"] is True


def test_scenario_result_preserves_voice_infrastructure_failure_metadata():
    result = scenario_result(
        {"id": "sample", "criteria": "pass"},
        {
            "passed": False,
            "reason": "Voice eval failed before judging.",
            "infrastructure_failure": True,
            "error_type": "ConnectionRefusedError",
        },
        [],
        [],
    )

    assert result["passed"] is False
    assert result["infrastructure_failure"] is True
    assert result["error_type"] == "ConnectionRefusedError"


def test_result_writer_includes_pass_rate_transcripts_reasons_and_tool_calls(tmp_path: Path):
    results = [
        {
            "id": "passed_case",
            "category": "insurance",
            "passed": True,
            "reason": "Met criteria.",
            "turn_count": 2,
            "transcript": [{"speaker": "agent", "text": "Hello."}],
            "tool_calls": [{"name": "check_insurance", "arguments": {}, "result": {}}],
        },
        {
            "id": "failed_case",
            "category": "booking",
            "passed": False,
            "reason": "Missed criteria.",
            "turn_count": 1,
            "transcript": [{"speaker": "agent", "text": "No."}],
            "tool_calls": [],
        },
    ]
    output = build_run_output(
        results,
        model="test-model",
        latency_path=tmp_path / "missing-latency.jsonl",
        run_id="abc123",
        timestamp=1_770_000_000,
    )

    results_path = tmp_path / "results.json"
    runs_path = tmp_path / "runs.jsonl"
    write_results(output, results_path=results_path, runs_path=runs_path)

    written = json.loads(results_path.read_text(encoding="utf-8"))
    assert written["pass_rate"] == 0.5
    assert written["judge_policy"]["version"] == JUDGE_POLICY_VERSION
    assert written["scenario_count"] == 2
    assert written["scenarios"][0]["reason"] == "Met criteria."
    assert written["scenarios"][0]["transcript"][0]["text"] == "Hello."
    assert written["scenarios"][0]["tool_calls"][0]["name"] == "check_insurance"

    trend = json.loads(runs_path.read_text(encoding="utf-8"))
    assert trend["run_id"] == "abc123"
    assert trend["judge_policy"]["version"] == JUDGE_POLICY_VERSION
    assert trend["passed_count"] == 1
    assert trend["failed_count"] == 1
    assert trend["bot_url"] is None
    assert trend["voice_latency"] is None
    assert trend["failing_scenario_ids"] == ["failed_case"]
    assert trend["category_summary"] == {
        "booking": {"passed": 0, "failed": 1, "total": 1, "pass_rate": 0.0},
        "insurance": {"passed": 1, "failed": 0, "total": 1, "pass_rate": 1.0},
    }
    assert trend["scenario_results"] == [
        {
            "id": "passed_case",
            "category": "insurance",
            "passed": True,
            "reason": "Met criteria.",
            "turn_count": 2,
            "tool_call_count": 1,
        },
        {
            "id": "failed_case",
            "category": "booking",
            "passed": False,
            "reason": "Missed criteria.",
            "turn_count": 1,
            "tool_call_count": 0,
        },
    ]


def test_voice_agent_metadata_in_run_output_and_runs_jsonl(tmp_path: Path):
    bot_path = tmp_path / "bot_meta.py"
    bot_path.write_text("print('bot')\n", encoding="utf-8")

    output = build_run_output(
        [],
        model="test-model",
        latency_path=tmp_path / "missing-latency.jsonl",
        run_id="meta123",
        timestamp=1_770_000_000,
        eval_mode="voice",
        bot_url="http://localhost:8123",
        bot_name="bot-meta",
        bot_path=str(bot_path),
        batch_id="batch-1",
        bot_command="uv run bot_meta.py --host localhost --port 8123",
    )

    assert output["voice_agent"]["name"] == "bot-meta"
    assert output["voice_agent"]["passed_path"] == str(bot_path)
    assert output["voice_agent"]["resolved_path"] == str(bot_path.resolve())
    assert output["voice_agent"]["sha256"]
    assert output["voice_agent"]["port"] == 8123
    assert output["voice_agent"]["batch_id"] == "batch-1"

    runs_path = tmp_path / "runs.jsonl"
    write_results(output, results_path=tmp_path / "results.json", runs_path=runs_path)
    trend = json.loads(runs_path.read_text(encoding="utf-8"))
    assert trend["voice_agent"]["name"] == "bot-meta"
    assert trend["voice_agent"]["port"] == 8123


def test_read_voice_p95_latency_handles_missing_and_ttfa_ttla_keys(tmp_path: Path):
    assert read_voice_p95_latency(tmp_path / "missing.jsonl") is None

    latency_path = tmp_path / "latency.jsonl"
    latency_path.write_text(
        "\n".join(
            [
                json.dumps({"status": "complete", "ttfa_ms": 100, "ttla_ms": 500}),
                json.dumps({"status": "complete", "ttfa_ms": 300, "ttla_ms": 900}),
                json.dumps({"status": "interrupted", "ttfa_ms": 1000, "ttla_ms": 2000}),
                json.dumps({"latency_ms": 250}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_voice_p95_latency(latency_path) == {
        "ttfa_p95_ms": 300,
        "ttla_p95_ms": 900,
    }


def test_all_scenarios_have_dashboard_metadata():
    categories = {
        "booking",
        "rescheduling",
        "insurance",
        "medical_safety",
        "policy_guardrail",
        "call_closure",
    }
    severities = {"critical", "high", "medium"}

    assert SCENARIOS
    for scenario in SCENARIOS:
        assert scenario["category"] in categories
        assert scenario["severity"] in severities
