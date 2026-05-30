from __future__ import annotations

import json
from pathlib import Path

import streamlit_dashboard


def _result(**overrides):
    payload = {
        "run_id": "abc123",
        "timestamp": 1780176177.0,
        "scenario_count": 2,
        "passed": 1,
        "pass_rate": 0.5,
        "p95_agent_reply_ms": 1200,
        "p95_latency_ms": 3400,
        "live_latency": {"completed_turns": 3},
        "scenarios": [
            {
                "id": "office_hours",
                "passed": True,
                "reason": "Met criteria.",
                "turns": 2,
                "duration_ms": 1000,
                "p95_agent_reply_ms": 900,
                "agent_reply_ms": [800, 900],
                "tool_calls": [],
                "transcript": [],
            },
            {
                "id": "emergency_swelling",
                "passed": False,
                "reason": "Did not escalate.",
                "turns": 3,
                "duration_ms": 2000,
                "p95_agent_reply_ms": 1300,
                "agent_reply_ms": [1300],
                "tool_calls": [{"name": "book_appointment"}],
                "transcript": [],
            },
        ],
    }
    payload.update(overrides)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_result_file_accepts_single_run_results(tmp_path):
    path = tmp_path / "results.json"
    payload = _result()
    _write_json(path, payload)

    loaded = streamlit_dashboard.load_result_file(path)

    assert loaded.error is None
    assert loaded.source_kind == "single"
    assert loaded.result == payload


def test_load_result_file_reports_missing_and_malformed_files(tmp_path):
    missing = streamlit_dashboard.load_result_file(tmp_path / "missing.json")

    assert missing.source_kind == "error"
    assert "No file found" in str(missing.error)

    malformed_path = tmp_path / "bad.json"
    malformed_path.write_text("{not json", encoding="utf-8")
    malformed = streamlit_dashboard.load_result_file(malformed_path)

    assert malformed.source_kind == "error"
    assert "Could not parse" in str(malformed.error)


def test_load_jsonl_rows_ignores_bad_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": 1, "pass_rate": 0.5}),
                "not-json",
                json.dumps(["not", "an", "object"]),
                json.dumps({"timestamp": 2, "pass_rate": 1.0}),
            ]
        ),
        encoding="utf-8",
    )

    loaded = streamlit_dashboard.load_jsonl_rows(path)

    assert loaded.error is None
    assert loaded.bad_line_count == 2
    assert [row["timestamp"] for row in loaded.rows] == [1, 2]


def test_load_result_file_normalizes_selected_batch_result(tmp_path):
    path = tmp_path / "batch_results.json"
    payload = {
        "batch_id": "batch",
        "bots": [
            {"summary": {"bot": "bot0"}, "result": _result(run_id="first")},
            {"summary": {"bot": "bot1"}, "result": _result(run_id="second")},
        ],
    }
    _write_json(path, payload)

    loaded = streamlit_dashboard.load_result_file(path, selected_batch_key="2:bot1")

    assert loaded.source_kind == "batch"
    assert [option.label for option in loaded.batch_options] == ["bot0", "bot1"]
    assert loaded.selected_batch_key == "2:bot1"
    assert loaded.result is not None
    assert loaded.result["run_id"] == "second"


def test_scenario_table_and_counts_flatten_result():
    payload = _result()

    counts = streamlit_dashboard.pass_fail_counts(payload)
    rows = streamlit_dashboard.scenario_table_rows(payload)
    failures = streamlit_dashboard.failure_summaries(payload)

    assert counts == {
        "scenario_count": 2,
        "passed": 1,
        "failed": 1,
        "pass_rate": 0.5,
    }
    assert rows == [
        {
            "status": "Pass",
            "id": "office_hours",
            "reason": "Met criteria.",
            "turns": 2,
            "duration_ms": 1000,
            "p95_reply_ms": 900,
            "tool_calls": 0,
        },
        {
            "status": "Fail",
            "id": "emergency_swelling",
            "reason": "Did not escalate.",
            "turns": 3,
            "duration_ms": 2000,
            "p95_reply_ms": 1300,
            "tool_calls": 1,
        },
    ]
    assert failures == [{"id": "emergency_swelling", "reason": "Did not escalate."}]


def test_build_eval_command_uses_argument_list_and_expected_paths(tmp_path):
    request = streamlit_dashboard.EvalRunRequest(
        scenario_id="office_hours",
        bot_url="http://127.0.0.1:7860",
        max_turns=4,
        today="2026-05-30",
        caller_tts_provider="say",
        output_path=tmp_path / "eval_dashboard_runs" / "run" / "results.json",
        runs_path=tmp_path / "runs.jsonl",
        latency_path=tmp_path / "eval_dashboard_runs" / "run" / "latency.jsonl",
    )

    command = streamlit_dashboard.build_eval_command(request)

    assert command == [
        "uv",
        "run",
        "python",
        "eval_runner.py",
        "--scenario",
        "office_hours",
        "--bot-url",
        "http://127.0.0.1:7860",
        "--max-turns",
        "4",
        "--output",
        str(tmp_path / "eval_dashboard_runs" / "run" / "results.json"),
        "--runs",
        str(tmp_path / "runs.jsonl"),
        "--latency-path",
        str(tmp_path / "eval_dashboard_runs" / "run" / "latency.jsonl"),
        "--caller-tts-provider",
        "say",
        "--today",
        "2026-05-30",
    ]
    assert all(isinstance(part, str) for part in command)
