import json
from pathlib import Path

from dashboard import (
    DEFAULT_BATCH_BOTS,
    build_batch_command,
    build_batch_leaderboard,
    build_batch_scenario_matrix,
    calculate_insights,
    filter_runs,
    normalize_batch_results,
    normalize_batch_runs,
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


def test_batch_results_normalize_bot_summaries_and_preserve_failure_metadata():
    first_scenario = SCENARIOS[0]["id"]
    second_scenario = SCENARIOS[1]["id"]
    raw = {
        "batch_id": "batch-1",
        "timestamp_iso": "2026-05-30T12:00:00Z",
        "bot_order": ["bot0", "bot1"],
        "scenario_order": [first_scenario, second_scenario],
        "bots": {
            "bot0": {
                "bot_id": "bot0",
                "bot_name": "bot0",
                "bot_path": "bot0.py",
                "scenario_count": 2,
                "passed_count": 1,
                "pass_rate": 0.5,
                "judge_error_count": 1,
                "voice_latency": {"ttfa_p95_ms": 1500, "ttla_p95_ms": 3200},
                "scenarios": [
                    {"id": first_scenario, "passed": True},
                    {
                        "id": second_scenario,
                        "passed": False,
                        "reason": "Judge returned malformed JSON.",
                        "judge_error": True,
                        "artifacts": {"eval_log": "batch_artifacts/bot0/eval.log"},
                    },
                ],
            },
            "bot1": {
                "bot_id": "bot1",
                "bot_name": "bot1",
                "bot_path": "bot1.py",
                "scenario_count": 2,
                "passed_count": 0,
                "infrastructure_failure_count": 1,
                "voice_latency": {"ttfa_p95_ms": 900, "ttla_p95_ms": 2500},
                "scenarios": [
                    {
                        "id": first_scenario,
                        "passed": False,
                        "infrastructure_failure": True,
                        "error_type": "startup_timeout",
                    },
                    {"id": second_scenario, "passed": False},
                ],
            },
        },
    }

    batch = normalize_batch_results(raw)

    assert batch is not None
    assert batch["expected_scenario_count"] == len(SCENARIOS)
    assert batch["covered_scenario_count"] == 2
    assert batch["bots"]["bot0"]["failed_count"] == 1
    assert batch["bots"]["bot0"]["scenarios"][1]["judge_error"] is True
    assert batch["bots"]["bot1"]["scenarios"][0]["infrastructure_failure"] is True
    assert batch["bots"]["bot1"]["category_summary"][SCENARIOS[0]["category"]]["failed"] == 2


def test_batch_scenario_matrix_includes_all_configured_scenarios_and_default_bots():
    first_scenario = SCENARIOS[0]["id"]
    raw = {
        "batch_id": "batch-1",
        "bot_order": ["bot0"],
        "scenario_order": [first_scenario],
        "bots": {
            "bot0": {
                "bot_id": "bot0",
                "bot_path": "bot0.py",
                "scenario_count": 1,
                "passed_count": 1,
                "scenarios": [{"id": first_scenario, "passed": True}],
            }
        },
    }

    batch = normalize_batch_results(raw)
    assert batch is not None
    matrix = build_batch_scenario_matrix(batch)

    assert len(matrix) == len(SCENARIOS)
    assert matrix[0]["scenario_id"] == first_scenario
    assert matrix[0]["bot0"] == "PASS"
    assert "bot3" in matrix[0]
    assert matrix[-1]["scenario_id"] == SCENARIOS[-1]["id"]


def test_batch_leaderboard_sorts_by_pass_rate():
    raw = {
        "batch_id": "batch-1",
        "bot_order": ["bot0", "bot1"],
        "bots": {
            "bot0": {"bot_id": "bot0", "scenario_count": 2, "passed_count": 1},
            "bot1": {"bot_id": "bot1", "scenario_count": 2, "passed_count": 2},
        },
    }

    batch = normalize_batch_results(raw)
    assert batch is not None
    leaderboard = build_batch_leaderboard(batch)

    assert [row["bot"] for row in leaderboard] == ["bot1", "bot0"]
    assert leaderboard[0]["pass_rate"] == 1.0


def test_batch_command_targets_all_four_bots_without_limit():
    command = build_batch_command()

    assert command == "uv run python batch_eval_runner.py --bots bot0.py bot1.py bot2.py bot3.py"
    assert all(bot in command for bot in DEFAULT_BATCH_BOTS)
    assert "--limit" not in command


def test_batch_runs_normalize_history_rows():
    rows = normalize_batch_runs(
        [
            {
                "batch_id": "batch-1",
                "timestamp_iso": "2026-05-30T12:00:00Z",
                "bot_id": "bot3",
                "bot_path": "bot3.py",
                "scenario_count": 25,
                "passed_count": 20,
                "voice_latency": {"ttfa_p95_ms": 1000, "ttla_p95_ms": 2500},
            }
        ]
    )

    assert rows[0]["bot_id"] == "bot3"
    assert rows[0]["pass_rate"] == 0.8
    assert rows[0]["failed_count"] == 5
