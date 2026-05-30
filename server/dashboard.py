import json
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from scenarios import SCENARIOS

KNOWN_CATEGORIES = [
    "booking",
    "rescheduling",
    "insurance",
    "medical_safety",
    "policy_guardrail",
    "call_closure",
]
STATUS_OPTIONS = ["pass", "fail"]


def scenario_metadata_by_id() -> dict[str, dict[str, str]]:
    return {
        str(scenario["id"]): {
            "category": str(scenario.get("category", "uncategorized")),
            "severity": str(scenario.get("severity", "medium")),
        }
        for scenario in SCENARIOS
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def read_results(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_timestamp(value: Any, iso_value: Any = None) -> datetime | None:
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str) and value:
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except ValueError:
            iso_value = value
    if isinstance(iso_value, str) and iso_value:
        raw = iso_value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def normalize_voice_latency(row: dict[str, Any]) -> dict[str, int | None]:
    raw = row.get("voice_latency")
    ttfa = None
    ttla = None
    if isinstance(raw, dict):
        ttfa = raw.get("ttfa_p95_ms")
        ttla = raw.get("ttla_p95_ms")
    if ttfa is None:
        ttfa = row.get("voice_p95_latency_ms")
    return {
        "ttfa_p95_ms": safe_int(ttfa) if ttfa is not None else None,
        "ttla_p95_ms": safe_int(ttla) if ttla is not None else None,
    }


def get_bot_url(row: dict[str, Any]) -> str | None:
    bot_url = row.get("bot_url")
    if isinstance(bot_url, str) and bot_url:
        return bot_url
    voice_agent = row.get("voice_agent")
    if isinstance(voice_agent, dict):
        agent_url = voice_agent.get("bot_url")
        if isinstance(agent_url, str) and agent_url:
            return agent_url
    return None


def normalize_scenario_result(result: dict[str, Any]) -> dict[str, Any]:
    metadata = scenario_metadata_by_id().get(str(result.get("id", "")), {})
    tool_calls = result.get("tool_calls", [])
    tool_call_count = result.get("tool_call_count")
    if tool_call_count is None and isinstance(tool_calls, list):
        tool_call_count = len(tool_calls)

    normalized = {
        "id": str(result.get("id", "")),
        "category": str(result.get("category") or metadata.get("category", "uncategorized")),
        "severity": str(result.get("severity") or metadata.get("severity", "medium")),
        "passed": bool(result.get("passed")),
        "reason": str(result.get("reason", "")),
        "turn_count": safe_int(result.get("turn_count")),
        "tool_call_count": safe_int(tool_call_count),
    }
    if "transcript" in result:
        normalized["transcript"] = result.get("transcript") or []
    if "tool_calls" in result:
        normalized["tool_calls"] = result.get("tool_calls") or []
    return normalized


def infer_legacy_scenario_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    failing_ids = row.get("failing_scenario_ids") or []
    if not isinstance(failing_ids, list):
        return []

    metadata = scenario_metadata_by_id()
    inferred = [
        normalize_scenario_result(
            {
                "id": scenario_id,
                "category": metadata.get(str(scenario_id), {}).get("category", "uncategorized"),
                "passed": False,
                "reason": "Failed in legacy trend row.",
            }
        )
        for scenario_id in failing_ids
    ]

    scenario_count = safe_int(row.get("scenario_count"))
    if scenario_count == len(SCENARIOS):
        failing = {str(scenario_id) for scenario_id in failing_ids}
        for scenario in SCENARIOS:
            scenario_id = str(scenario["id"])
            if scenario_id in failing:
                continue
            inferred.append(
                normalize_scenario_result(
                    {
                        "id": scenario_id,
                        "category": scenario.get("category"),
                        "severity": scenario.get("severity"),
                        "passed": True,
                    }
                )
            )
    return inferred


def normalize_category_summary(value: Any) -> dict[str, dict[str, int | float]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, int | float]] = {}
    for category, raw_row in value.items():
        if not isinstance(raw_row, dict):
            continue
        total = safe_int(raw_row.get("total"))
        passed = safe_int(raw_row.get("passed"))
        failed = safe_int(raw_row.get("failed"), max(total - passed, 0))
        if total == 0:
            total = passed + failed
        pass_rate = safe_float(raw_row.get("pass_rate"), passed / total if total else 0.0)
        normalized[str(category)] = {
            "passed": passed,
            "failed": failed,
            "total": total,
            "pass_rate": round(pass_rate, 3),
        }
    return normalized


def build_category_summary_from_scenarios(
    scenarios: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    summary: dict[str, dict[str, int | float]] = {}
    for scenario in scenarios:
        normalized = normalize_scenario_result(scenario)
        category = normalized["category"]
        row = summary.setdefault(category, {"passed": 0, "failed": 0, "total": 0, "pass_rate": 0.0})
        row["total"] = int(row["total"]) + 1
        if normalized["passed"]:
            row["passed"] = int(row["passed"]) + 1
        else:
            row["failed"] = int(row["failed"]) + 1
    for row in summary.values():
        total = int(row["total"])
        row["pass_rate"] = round(int(row["passed"]) / total, 3) if total else 0.0
    return dict(sorted(summary.items()))


def normalize_run(row: dict[str, Any], index: int = 0) -> dict[str, Any]:
    normalized = deepcopy(row)
    scenario_results = row.get("scenario_results")
    if isinstance(scenario_results, list):
        scenarios = [
            normalize_scenario_result(result)
            for result in scenario_results
            if isinstance(result, dict)
        ]
    else:
        scenarios = infer_legacy_scenario_results(row)

    scenario_count = safe_int(row.get("scenario_count"), len(scenarios))
    passed_count = row.get("passed_count")
    pass_rate = safe_float(
        row.get("pass_rate"),
        safe_int(passed_count) / scenario_count if scenario_count and passed_count is not None else 0.0,
    )
    if passed_count is None and scenario_count:
        passed_count = round(pass_rate * scenario_count)
    failed_count = row.get("failed_count")
    if failed_count is None:
        failed_count = scenario_count - safe_int(passed_count)

    timestamp = parse_timestamp(row.get("timestamp"), row.get("timestamp_iso"))
    voice_latency = normalize_voice_latency(row)

    raw_failing_ids = row.get("failing_scenario_ids", [])
    if not isinstance(raw_failing_ids, list):
        raw_failing_ids = []

    normalized.update(
        {
            "run_id": str(row.get("run_id") or f"run-{index + 1}"),
            "timestamp": row.get("timestamp"),
            "timestamp_iso": row.get("timestamp_iso") or (timestamp.isoformat() if timestamp else ""),
            "timestamp_dt": timestamp,
            "eval_mode": str(row.get("eval_mode", "unknown")),
            "model": str(row.get("model", "unknown")),
            "bot_url": get_bot_url(row),
            "scenario_count": scenario_count,
            "passed_count": safe_int(passed_count),
            "failed_count": safe_int(failed_count),
            "pass_rate": round(pass_rate, 3),
            "voice_p95_latency_ms": voice_latency["ttfa_p95_ms"],
            "voice_latency": voice_latency,
            "failing_scenario_ids": [str(item) for item in raw_failing_ids if item is not None],
            "category_summary": normalize_category_summary(row.get("category_summary"))
            or build_category_summary_from_scenarios(scenarios),
            "scenario_results": scenarios,
            "_source_index": index,
        }
    )
    return normalized


def normalize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_run(row, index=index) for index, row in enumerate(rows)]
    return sorted(
        normalized,
        key=lambda row: (
            row["timestamp_dt"] or datetime.min.replace(tzinfo=UTC),
            row["_source_index"],
        ),
    )


def normalize_latest_results(results: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(results, dict):
        return None
    normalized = deepcopy(results)
    scenarios = [
        normalize_scenario_result(scenario)
        for scenario in results.get("scenarios", [])
        if isinstance(scenario, dict)
    ]
    scenario_count = safe_int(results.get("scenario_count"), len(scenarios))
    passed_count = safe_int(
        results.get("passed_count"),
        sum(1 for scenario in scenarios if bool(scenario.get("passed"))),
    )
    failed_count = safe_int(results.get("failed_count"), scenario_count - passed_count)
    pass_rate = safe_float(
        results.get("pass_rate"),
        passed_count / scenario_count if scenario_count else 0.0,
    )
    voice_latency = normalize_voice_latency(results)
    normalized.update(
        {
            "run_id": str(results.get("run_id", "latest")),
            "eval_mode": str(results.get("eval_mode", "unknown")),
            "model": str(results.get("model", "unknown")),
            "bot_url": get_bot_url(results),
            "scenario_count": scenario_count,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "pass_rate": round(pass_rate, 3),
            "voice_p95_latency_ms": voice_latency["ttfa_p95_ms"],
            "voice_latency": voice_latency,
            "category_summary": normalize_category_summary(results.get("category_summary"))
            or build_category_summary_from_scenarios(scenarios),
            "scenarios": scenarios,
        }
    )
    return normalized


def latest_results_as_run(results: dict[str, Any] | None) -> dict[str, Any] | None:
    latest = normalize_latest_results(results)
    if latest is None:
        return None
    timestamp = parse_timestamp(latest.get("timestamp"), latest.get("timestamp_iso"))
    return normalize_run(
        {
            "run_id": latest["run_id"],
            "timestamp": latest.get("timestamp"),
            "timestamp_iso": latest.get("timestamp_iso") or (timestamp.isoformat() if timestamp else ""),
            "eval_mode": latest["eval_mode"],
            "model": latest["model"],
            "bot_url": latest.get("bot_url"),
            "scenario_count": latest["scenario_count"],
            "passed_count": latest["passed_count"],
            "failed_count": latest["failed_count"],
            "pass_rate": latest["pass_rate"],
            "voice_p95_latency_ms": latest["voice_p95_latency_ms"],
            "voice_latency": latest["voice_latency"],
            "failing_scenario_ids": [
                scenario["id"] for scenario in latest["scenarios"] if not scenario["passed"]
            ],
            "category_summary": latest["category_summary"],
            "scenario_results": latest["scenarios"],
        }
    )


def filter_runs(
    runs: list[dict[str, Any]],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    eval_modes: set[str] | None = None,
    models: set[str] | None = None,
    bot_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for run in runs:
        timestamp = run.get("timestamp_dt")
        run_date = timestamp.date() if isinstance(timestamp, datetime) else None
        if start_date is not None and run_date is not None and run_date < start_date:
            continue
        if end_date is not None and run_date is not None and run_date > end_date:
            continue
        if eval_modes and run.get("eval_mode") not in eval_modes:
            continue
        if models and run.get("model") not in models:
            continue
        if bot_urls and (run.get("bot_url") or "n/a") not in bot_urls:
            continue
        filtered.append(run)
    return filtered


def filter_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    categories: set[str] | None = None,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered = []
    for scenario in scenarios:
        normalized = normalize_scenario_result(scenario)
        status = "pass" if normalized["passed"] else "fail"
        if categories and normalized["category"] not in categories:
            continue
        if statuses and status not in statuses:
            continue
        filtered.append(normalized)
    return filtered


def runs_with_filtered_scenarios(
    runs: list[dict[str, Any]],
    *,
    categories: set[str] | None = None,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered_runs = []
    for run in runs:
        next_run = deepcopy(run)
        next_run["scenario_results"] = filter_scenarios(
            run.get("scenario_results", []),
            categories=categories,
            statuses=statuses,
        )
        next_run["category_summary"] = build_category_summary_from_scenarios(
            next_run["scenario_results"]
        )
        filtered_runs.append(next_run)
    return filtered_runs


def status_by_run(run: dict[str, Any]) -> dict[str, bool]:
    return {
        str(scenario["id"]): bool(scenario["passed"])
        for scenario in run.get("scenario_results", [])
        if scenario.get("id")
    }


def calculate_insights(runs: list[dict[str, Any]]) -> dict[str, list[str]]:
    if not runs:
        return {
            "regressions": [],
            "recoveries": [],
            "persistent_failures": [],
            "flaky_scenarios": [],
        }

    latest_status = status_by_run(runs[-1])
    previous_status = status_by_run(runs[-2]) if len(runs) >= 2 else {}

    regressions = sorted(
        scenario_id
        for scenario_id, passed in latest_status.items()
        if passed is False and previous_status.get(scenario_id) is True
    )
    recoveries = sorted(
        scenario_id
        for scenario_id, passed in latest_status.items()
        if passed is True and previous_status.get(scenario_id) is False
    )

    scenario_history: dict[str, list[bool]] = defaultdict(list)
    for run in runs:
        for scenario_id, passed in status_by_run(run).items():
            scenario_history[scenario_id].append(passed)

    persistent_failures = sorted(
        scenario_id
        for scenario_id, history in scenario_history.items()
        if len(history[-3:]) == 3 and all(passed is False for passed in history[-3:])
    )
    flaky_scenarios = sorted(
        scenario_id
        for scenario_id, history in scenario_history.items()
        if sum(1 for before, after in zip(history, history[1:], strict=False) if before != after) > 1
    )
    return {
        "regressions": regressions,
        "recoveries": recoveries,
        "persistent_failures": persistent_failures,
        "flaky_scenarios": flaky_scenarios,
    }


def calculate_category_health(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"latest": [], "persistent_failing_categories": []}

    latest_summary = runs[-1].get("category_summary") or {}
    latest = [
        {
            "category": category,
            "passed": int(row.get("passed", 0)),
            "failed": int(row.get("failed", 0)),
            "total": int(row.get("total", 0)),
            "pass_rate": float(row.get("pass_rate", 0.0)),
        }
        for category, row in sorted(latest_summary.items())
    ]

    last_three = runs[-3:]
    failing_category_sets = [
        {
            str(scenario.get("category", "uncategorized"))
            for scenario in run.get("scenario_results", [])
            if not scenario.get("passed")
        }
        for run in last_three
    ]
    persistent_categories = (
        sorted(set.intersection(*failing_category_sets))
        if len(failing_category_sets) == 3
        else []
    )
    return {
        "latest": latest,
        "persistent_failing_categories": persistent_categories,
    }


def build_failure_matrix(runs: list[dict[str, Any]]) -> dict[str, Any]:
    run_labels = [run_label(run) for run in runs]
    scenario_ids = sorted(
        {
            str(scenario["id"])
            for run in runs
            for scenario in run.get("scenario_results", [])
            if scenario.get("id")
        }
    )
    rows = []
    for scenario_id in scenario_ids:
        row = {"scenario_id": scenario_id}
        for run, label in zip(runs, run_labels, strict=False):
            status = status_by_run(run).get(scenario_id)
            if status is True:
                row[label] = "PASS"
            elif status is False:
                row[label] = "FAIL"
            else:
                row[label] = ""
        rows.append(row)
    return {"columns": ["scenario_id", *run_labels], "rows": rows}


def scenario_history(runs: list[dict[str, Any]], scenario_id: str) -> list[dict[str, Any]]:
    history = []
    for run in runs:
        status = status_by_run(run).get(scenario_id)
        if status is None:
            continue
        scenario = next(
            (
                item
                for item in run.get("scenario_results", [])
                if str(item.get("id")) == scenario_id
            ),
            {},
        )
        history.append(
            {
                "run": run_label(run),
                "run_id": run.get("run_id"),
                "timestamp": run.get("timestamp_iso"),
                "passed": bool(status),
                "category": scenario.get("category", "uncategorized"),
                "reason": scenario.get("reason", ""),
                "turn_count": scenario.get("turn_count", 0),
                "tool_call_count": scenario.get("tool_call_count", 0),
            }
        )
    return history


def run_label(run: dict[str, Any]) -> str:
    run_id = str(run.get("run_id", "unknown"))
    timestamp = run.get("timestamp_dt")
    if isinstance(timestamp, datetime):
        return f"{timestamp.strftime('%m-%d %H:%M')} {run_id}"
    return run_id


def format_percent(value: Any) -> str:
    return f"{safe_float(value) * 100:.1f}%"


def format_latency(value: Any) -> str:
    return f"{safe_int(value)} ms" if value is not None else "n/a"


def compact_reason(reason: str, limit: int = 180) -> str:
    if len(reason) <= limit:
        return reason
    return reason[: limit - 1].rstrip() + "..."


def summarize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    name = call.get("name") or call.get("function", {}).get("name") or "unknown"
    arguments = call.get("arguments")
    result = call.get("result")
    summary: dict[str, Any] = {"name": name}
    if isinstance(arguments, dict):
        summary["argument_keys"] = sorted(arguments.keys())
    if isinstance(result, dict):
        summary["result_keys"] = sorted(result.keys())
        if "ok" in result:
            summary["ok"] = result["ok"]
        if "confirmation_id" in result:
            summary["confirmation_id"] = result["confirmation_id"]
        if "status" in result:
            summary["status"] = result["status"]
    return summary


def render_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: render_json_safe(item) for key, item in value.items() if key != "timestamp_dt"}
    if isinstance(value, list):
        return [render_json_safe(item) for item in value]
    return value


def main() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Eval Results Dashboard", layout="wide")
    st.title("Eval Results Dashboard")

    with st.sidebar:
        st.header("Artifacts")
        results_path = st.text_input("Latest results", "results.json")
        runs_path = st.text_input("Run history", "runs.jsonl")

    raw_runs = read_jsonl(runs_path)
    raw_results = read_results(results_path)
    runs = normalize_runs(raw_runs)
    latest_results = normalize_latest_results(raw_results)

    all_run_dates = [run["timestamp_dt"].date() for run in runs if run.get("timestamp_dt")]
    min_date = min(all_run_dates) if all_run_dates else None
    max_date = max(all_run_dates) if all_run_dates else None

    with st.sidebar:
        st.header("Filters")
        selected_range = None
        if min_date and max_date:
            selected_range = st.date_input("Date range", value=(min_date, max_date))
        eval_modes = sorted({run["eval_mode"] for run in runs if run.get("eval_mode")})
        models = sorted({run["model"] for run in runs if run.get("model")})
        bot_urls = sorted({run.get("bot_url") or "n/a" for run in runs})

        selected_modes = set(st.multiselect("Eval mode", eval_modes, default=eval_modes))
        selected_models = set(st.multiselect("Model", models, default=models))
        selected_bot_urls = set(st.multiselect("Bot URL", bot_urls, default=bot_urls))
        selected_categories = set(
            st.multiselect("Category", KNOWN_CATEGORIES, default=KNOWN_CATEGORIES)
        )
        selected_statuses = set(st.multiselect("Status", STATUS_OPTIONS, default=STATUS_OPTIONS))

    start_date = None
    end_date = None
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range

    run_filtered = filter_runs(
        runs,
        start_date=start_date,
        end_date=end_date,
        eval_modes=selected_modes or None,
        models=selected_models or None,
        bot_urls=selected_bot_urls or None,
    )
    scenario_filtered_runs = runs_with_filtered_scenarios(
        run_filtered,
        categories=selected_categories or None,
        statuses=selected_statuses or None,
    )

    fallback_latest_run = latest_results_as_run(raw_results)
    visible_runs = scenario_filtered_runs
    if not visible_runs and fallback_latest_run is not None:
        visible_runs = runs_with_filtered_scenarios(
            [fallback_latest_run],
            categories=selected_categories or None,
            statuses=selected_statuses or None,
        )

    scenario_options = sorted(
        {
            scenario["id"]
            for run in visible_runs
            for scenario in run.get("scenario_results", [])
            if scenario.get("id")
        }
        or {str(scenario["id"]) for scenario in SCENARIOS}
    )
    default_scenario = scenario_options[0] if scenario_options else None
    if latest_results:
        failed_latest = [
            scenario["id"]
            for scenario in latest_results["scenarios"]
            if not scenario["passed"] and scenario["id"] in scenario_options
        ]
        if failed_latest:
            default_scenario = failed_latest[0]

    with st.sidebar:
        selected_scenario = st.selectbox(
            "Scenario",
            scenario_options,
            index=scenario_options.index(default_scenario) if default_scenario in scenario_options else 0,
        )

    if not raw_runs:
        st.info(f"No run history found at `{runs_path}`. Trend views will appear after eval runs.")
    if raw_results is None:
        st.info(f"No latest results found at `{results_path}`. Latest-run transcript views are hidden.")
    if not raw_runs and raw_results is None:
        st.stop()

    latest_for_kpis = visible_runs[-1] if visible_runs else None
    previous_for_kpis = visible_runs[-2] if len(visible_runs) >= 2 else None
    pass_delta = None
    if latest_for_kpis and previous_for_kpis:
        pass_delta = latest_for_kpis["pass_rate"] - previous_for_kpis["pass_rate"]

    kpi_cols = st.columns(4)
    kpi_cols[0].metric(
        "Latest Pass Rate",
        format_percent(latest_for_kpis["pass_rate"]) if latest_for_kpis else "n/a",
        f"{pass_delta * 100:+.1f} pp" if pass_delta is not None else None,
    )
    kpi_cols[1].metric(
        "Failing Scenarios",
        str(latest_for_kpis["failed_count"]) if latest_for_kpis else "n/a",
    )
    kpi_cols[2].metric(
        "TTFA p95",
        format_latency(latest_for_kpis["voice_latency"]["ttfa_p95_ms"])
        if latest_for_kpis
        else "n/a",
    )
    kpi_cols[3].metric(
        "Latest Run",
        str(latest_for_kpis["run_id"]) if latest_for_kpis else "n/a",
    )

    insights = calculate_insights(scenario_filtered_runs)
    insight_cols = st.columns(4)
    insight_cols[0].caption("New Regressions")
    insight_cols[0].write(", ".join(insights["regressions"]) or "None")
    insight_cols[1].caption("Recoveries")
    insight_cols[1].write(", ".join(insights["recoveries"]) or "None")
    insight_cols[2].caption("Persistent Failures")
    insight_cols[2].write(", ".join(insights["persistent_failures"]) or "None")
    insight_cols[3].caption("Flaky Scenarios")
    insight_cols[3].write(", ".join(insights["flaky_scenarios"]) or "None")

    trend_tab, matrix_tab, category_tab, latest_tab, drilldown_tab, raw_tab = st.tabs(
        [
            "Trend",
            "Failure Matrix",
            "Category Health",
            "Latest Run",
            "Scenario Drilldown",
            "Raw Artifacts",
        ]
    )

    with trend_tab:
        if not run_filtered:
            st.warning("No runs match the current run-level filters.")
        else:
            trend_rows = [
                {
                    "run": run_label(run),
                    "timestamp": run.get("timestamp_dt"),
                    "pass_rate": run["pass_rate"],
                    "pass_rate_percent": run["pass_rate"] * 100,
                    "TTFA p95": run["voice_latency"]["ttfa_p95_ms"],
                    "TTLA p95": run["voice_latency"]["ttla_p95_ms"],
                    "failed_count": run["failed_count"],
                }
                for run in run_filtered
            ]
            trend_df = pd.DataFrame(trend_rows)
            st.subheader("Pass Rate")
            st.line_chart(trend_df, x="run", y="pass_rate_percent")
            st.subheader("Voice Latency")
            st.line_chart(trend_df, x="run", y=["TTFA p95", "TTLA p95"])
            st.subheader("Runs")
            table_df = pd.DataFrame(
                [
                    {
                        "run_id": run["run_id"],
                        "timestamp": run["timestamp_iso"],
                        "mode": run["eval_mode"],
                        "model": run["model"],
                        "bot_url": run.get("bot_url") or "n/a",
                        "pass_rate": format_percent(run["pass_rate"]),
                        "passed": run["passed_count"],
                        "failed": run["failed_count"],
                        "TTFA p95": format_latency(run["voice_latency"]["ttfa_p95_ms"]),
                        "TTLA p95": format_latency(run["voice_latency"]["ttla_p95_ms"]),
                    }
                    for run in run_filtered
                ]
            )
            st.dataframe(table_df, use_container_width=True, hide_index=True)

    with matrix_tab:
        matrix = build_failure_matrix(scenario_filtered_runs)
        if not matrix["rows"]:
            st.warning("No scenario status data is available for the current filters.")
        else:
            matrix_df = pd.DataFrame(matrix["rows"], columns=matrix["columns"])

            def color_status(value: str) -> str:
                if value == "PASS":
                    return "background-color: #d7f0df; color: #134e2f"
                if value == "FAIL":
                    return "background-color: #f8d7da; color: #842029"
                return ""

            st.dataframe(
                matrix_df.style.map(color_status),
                use_container_width=True,
                hide_index=True,
            )

    with category_tab:
        health = calculate_category_health(scenario_filtered_runs)
        if not health["latest"]:
            st.warning("No category data is available for the current filters.")
        else:
            health_df = pd.DataFrame(health["latest"])
            health_df["pass_rate_percent"] = health_df["pass_rate"] * 100
            st.bar_chart(health_df, x="category", y="pass_rate_percent")
            st.dataframe(health_df, use_container_width=True, hide_index=True)
            st.subheader("Persistent Failing Categories")
            st.write(", ".join(health["persistent_failing_categories"]) or "None")

    with latest_tab:
        if latest_results is None:
            st.warning("Latest results artifact is missing, so transcripts and full tool calls are unavailable.")
        else:
            scenarios = filter_scenarios(
                latest_results["scenarios"],
                categories=selected_categories or None,
                statuses=selected_statuses or None,
            )
            scenarios = sorted(scenarios, key=lambda scenario: (scenario["passed"], scenario["id"]))
            if not scenarios:
                st.warning("No latest-run scenarios match the scenario-level filters.")
            for scenario in scenarios:
                status = "PASS" if scenario["passed"] else "FAIL"
                with st.expander(f"{status} - {scenario['id']}", expanded=not scenario["passed"]):
                    st.write(scenario["reason"] or "No judge reason recorded.")
                    st.caption(
                        f"{scenario['category']} - severity {scenario['severity']} - "
                        f"{scenario['turn_count']} turns - {scenario['tool_call_count']} tool calls"
                    )
                    transcript = scenario.get("transcript") or []
                    if transcript:
                        st.subheader("Transcript")
                        for turn in transcript:
                            speaker = turn.get("speaker", "unknown")
                            text = turn.get("text", "")
                            st.markdown(f"**{speaker}:** {text}")
                    tool_calls = scenario.get("tool_calls") or []
                    if tool_calls:
                        st.subheader("Tool Calls")
                        st.json([summarize_tool_call(call) for call in tool_calls])

    with drilldown_tab:
        if not selected_scenario:
            st.warning("No scenario is selected.")
        else:
            st.subheader(selected_scenario)
            history = scenario_history(scenario_filtered_runs, selected_scenario)
            if history:
                st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
            else:
                st.info("No historical status is available for this scenario in the filtered window.")

            latest_scenario = None
            if latest_results is not None:
                latest_scenario = next(
                    (
                        scenario
                        for scenario in latest_results["scenarios"]
                        if scenario["id"] == selected_scenario
                    ),
                    None,
                )
            if latest_scenario:
                st.subheader("Latest Judge Reason")
                st.write(latest_scenario["reason"] or "No judge reason recorded.")
                transcript = latest_scenario.get("transcript") or []
                if transcript:
                    st.subheader("Latest Transcript")
                    for turn in transcript:
                        st.markdown(f"**{turn.get('speaker', 'unknown')}:** {turn.get('text', '')}")
            else:
                st.info("Latest-run transcript is unavailable for this scenario.")

    with raw_tab:
        st.subheader("results.json")
        if raw_results is None:
            st.warning("Missing or invalid latest results artifact.")
        else:
            st.json(render_json_safe(raw_results))
        st.subheader("runs.jsonl")
        if not raw_runs:
            st.warning("Missing or empty run history artifact.")
        else:
            st.json(render_json_safe(raw_runs))


if __name__ == "__main__":
    main()
