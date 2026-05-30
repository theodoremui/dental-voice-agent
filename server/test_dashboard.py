import json
from pathlib import Path

from dashboard import (
    calculate_insights,
    filter_runs,
    normalize_latest_results,
    normalize_runs,
    read_jsonl,
    read_results,
)
from scenarios import SCENARIOS


def make_run(run_id: str, statuses: dict[str, bool]) -> dict:
    return {
        "run_id": run_id,
        "timestamp_iso": f"2026-05-30T0{run_id[-1]}:00:00Z",
        "eval_mode": "voice",
        "model": "test-model",
        "scenario_count": len(statuses),
        "pass_rate": sum(statuses.values()) / len(statuses),
        "voice_latency": {"ttfa_p95_ms": 1000, "ttla_p95_ms": 3000},
        "scenario_results": [
            {
                "id": scenario_id,
                "category": "booking",
                "passed": passed,
                "reason": f"{scenario_id} {'passed' if passed else 'failed'}",
                "turn_count": 2,
                "tool_call_count": 1,
            }
            for scenario_id, passed in statuses.items()
        ],
    }


def test_old_minimal_runs_rows_normalize_without_crashing():
    scenario_id = SCENARIOS[0]["id"]
    runs = normalize_runs(
        [
            {
                "run_id": "legacy",
                "timestamp_iso": "2026-05-30T12:00:00Z",
                "eval_mode": "text",
                "model": "legacy-model",
                "scenario_count": 1,
                "pass_rate": 0.0,
                "voice_p95_latency_ms": None,
                "failing_scenario_ids": [scenario_id],
            }
        ]
    )

    assert runs[0]["run_id"] == "legacy"
    assert runs[0]["failed_count"] == 1
    assert runs[0]["scenario_results"][0]["id"] == scenario_id
    assert runs[0]["scenario_results"][0]["category"] == SCENARIOS[0]["category"]


def test_missing_results_still_allows_trend_data(tmp_path: Path):
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text(
        json.dumps(
            {
                "run_id": "run1",
                "timestamp_iso": "2026-05-30T12:00:00Z",
                "eval_mode": "voice",
                "model": "test-model",
                "scenario_count": 1,
                "pass_rate": 1.0,
                "failing_scenario_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert read_results(tmp_path / "missing-results.json") is None
    assert normalize_runs(read_jsonl(runs_path))[0]["pass_rate"] == 1.0


def test_missing_runs_still_allows_latest_run_detail(tmp_path: Path):
    scenario_id = SCENARIOS[0]["id"]
    results = {
        "run_id": "latest",
        "timestamp_iso": "2026-05-30T12:00:00Z",
        "eval_mode": "voice",
        "model": "test-model",
        "scenario_count": 1,
        "passed_count": 0,
        "pass_rate": 0.0,
        "voice_latency": {"ttfa_p95_ms": 1200, "ttla_p95_ms": 3400},
        "scenarios": [
            {
                "id": scenario_id,
                "passed": False,
                "reason": "Missed criteria.",
                "turn_count": 1,
                "transcript": [{"speaker": "agent", "text": "Hello."}],
                "tool_calls": [],
            }
        ],
    }

    assert read_jsonl(tmp_path / "missing-runs.jsonl") == []
    latest = normalize_latest_results(results)
    assert latest is not None
    assert latest["failed_count"] == 1
    assert latest["scenarios"][0]["transcript"][0]["text"] == "Hello."
    assert latest["scenarios"][0]["category"] == SCENARIOS[0]["category"]


def test_insight_calculations_find_regressions_recoveries_persistent_and_flaky():
    runs = normalize_runs(
        [
            make_run("run1", {"a": True, "b": False, "c": False, "d": True}),
            make_run("run2", {"a": True, "b": False, "c": True, "d": False}),
            make_run("run3", {"a": False, "b": False, "c": False, "d": True}),
        ]
    )

    insights = calculate_insights(runs)

    assert insights["regressions"] == ["a", "c"]
    assert insights["recoveries"] == ["d"]
    assert insights["persistent_failures"] == ["b"]
    assert insights["flaky_scenarios"] == ["c", "d"]


def test_run_filters_apply_date_mode_model_and_bot_url():
    runs = normalize_runs(
        [
            {
                **make_run("run1", {"a": True}),
                "timestamp_iso": "2026-05-29T12:00:00Z",
                "eval_mode": "text",
                "model": "old-model",
                "bot_url": "http://localhost:7860",
            },
            {
                **make_run("run2", {"a": False}),
                "timestamp_iso": "2026-05-30T12:00:00Z",
                "eval_mode": "voice",
                "model": "new-model",
                "bot_url": "http://localhost:7861",
            },
        ]
    )

    filtered = filter_runs(
        runs,
        start_date=runs[1]["timestamp_dt"].date(),
        end_date=runs[1]["timestamp_dt"].date(),
        eval_modes={"voice"},
        models={"new-model"},
        bot_urls={"http://localhost:7861"},
    )

    assert [run["run_id"] for run in filtered] == ["run2"]
