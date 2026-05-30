from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from scenarios import SCENARIOS

DEFAULT_RESULTS_PATH = Path("results.json")
DEFAULT_RUNS_PATH = Path("runs.jsonl")
DEFAULT_LATENCY_PATH = Path("latency.jsonl")
DEFAULT_EVAL_RUNNER = Path("eval_runner.py")
DEFAULT_BOT_URL = "http://localhost:7860"


@dataclass(frozen=True)
class BatchOption:
    key: str
    label: str
    result: dict[str, Any]


@dataclass(frozen=True)
class LoadedResult:
    path: Path
    result: dict[str, Any] | None
    source_kind: str
    error: str | None = None
    batch_options: tuple[BatchOption, ...] = ()
    selected_batch_key: str | None = None


@dataclass(frozen=True)
class JsonlRows:
    path: Path
    rows: tuple[dict[str, Any], ...]
    bad_line_count: int = 0
    error: str | None = None


@dataclass(frozen=True)
class EvalRunRequest:
    scenario_id: str
    bot_url: str
    max_turns: int
    today: str
    caller_tts_provider: str
    output_path: Path
    runs_path: Path
    latency_path: Path
    eval_runner_path: Path = DEFAULT_EVAL_RUNNER


def scenario_metadata() -> dict[str, dict[str, Any]]:
    return {
        str(scenario.get("id")): scenario
        for scenario in SCENARIOS
        if isinstance(scenario, dict) and scenario.get("id")
    }


def default_run_paths(timestamp: datetime | None = None) -> tuple[Path, Path, Path]:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run_dir = Path("eval_dashboard_runs") / stamp
    return run_dir / "results.json", DEFAULT_RUNS_PATH, run_dir / "latency.jsonl"


def read_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"No file found at {path}."
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"Could not parse {path}: {exc.msg} at line {exc.lineno}."
    except OSError as exc:
        return None, f"Could not read {path}: {exc}."
    if not isinstance(payload, dict):
        return None, f"{path} must contain a JSON object."
    return payload, None


def _single_result(payload: dict[str, Any]) -> dict[str, Any] | None:
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        return payload
    return None


def _batch_options(payload: dict[str, Any]) -> tuple[BatchOption, ...]:
    raw_bots = payload.get("bots")
    if not isinstance(raw_bots, list):
        return ()

    options: list[BatchOption] = []
    for index, bot_entry in enumerate(raw_bots, start=1):
        if not isinstance(bot_entry, dict):
            continue
        result = bot_entry.get("result")
        if not isinstance(result, dict) or _single_result(result) is None:
            continue
        summary = bot_entry.get("summary")
        summary = summary if isinstance(summary, dict) else {}
        label = (
            summary.get("bot")
            or result.get("bot")
            or result.get("agent_source")
            or f"bot-{index}"
        )
        options.append(
            BatchOption(
                key=f"{index}:{label}",
                label=str(label),
                result=result,
            )
        )
    return tuple(options)


def load_result_file(path: Path, selected_batch_key: str | None = None) -> LoadedResult:
    payload, error = read_json_file(path)
    if error:
        return LoadedResult(path=path, result=None, source_kind="error", error=error)
    assert payload is not None

    single = _single_result(payload)
    if single is not None:
        return LoadedResult(path=path, result=single, source_kind="single")

    batch_options = _batch_options(payload)
    if batch_options:
        selected = next(
            (option for option in batch_options if option.key == selected_batch_key),
            batch_options[0],
        )
        return LoadedResult(
            path=path,
            result=selected.result,
            source_kind="batch",
            batch_options=batch_options,
            selected_batch_key=selected.key,
        )

    return LoadedResult(
        path=path,
        result=None,
        source_kind="error",
        error=f"{path} does not look like an eval results.json or batch_results.json file.",
    )


def load_jsonl_rows(path: Path, *, limit: int | None = None) -> JsonlRows:
    if not path.exists():
        return JsonlRows(path=path, rows=())

    rows: list[dict[str, Any]] = []
    bad_line_count = 0
    try:
        with path.open(encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    bad_line_count += 1
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    bad_line_count += 1
    except OSError as exc:
        return JsonlRows(path=path, rows=(), error=f"Could not read {path}: {exc}.")

    if limit is not None and limit >= 0:
        rows = rows[-limit:]
    return JsonlRows(path=path, rows=tuple(rows), bad_line_count=bad_line_count)


def pass_fail_counts(result: dict[str, Any]) -> dict[str, int | float]:
    scenarios = [item for item in result.get("scenarios", []) if isinstance(item, dict)]
    scenario_count = int(result.get("scenario_count") or len(scenarios))
    passed = result.get("passed")
    if not isinstance(passed, int):
        passed = sum(1 for scenario in scenarios if scenario.get("passed") is True)
    failed = max(0, scenario_count - passed)
    pass_rate = result.get("pass_rate")
    if not isinstance(pass_rate, int | float):
        pass_rate = passed / scenario_count if scenario_count else 0.0
    return {
        "scenario_count": scenario_count,
        "passed": passed,
        "failed": failed,
        "pass_rate": float(pass_rate),
    }


def scenario_table_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in result.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        tool_calls = scenario.get("tool_calls")
        rows.append(
            {
                "status": "Pass" if scenario.get("passed") is True else "Fail",
                "id": scenario.get("id"),
                "reason": scenario.get("reason"),
                "turns": scenario.get("turns"),
                "duration_ms": scenario.get("duration_ms"),
                "p95_reply_ms": scenario.get("p95_agent_reply_ms"),
                "tool_calls": len(tool_calls) if isinstance(tool_calls, list) else 0,
            }
        )
    return rows


def failure_summaries(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": scenario.get("id"),
            "reason": scenario.get("reason") or "No judge reason recorded.",
        }
        for scenario in result.get("scenarios", [])
        if isinstance(scenario, dict) and scenario.get("passed") is not True
    ]


def build_eval_command(request: EvalRunRequest) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        str(request.eval_runner_path),
        "--scenario",
        request.scenario_id,
        "--bot-url",
        request.bot_url,
        "--max-turns",
        str(max(1, int(request.max_turns))),
        "--output",
        str(request.output_path),
        "--runs",
        str(request.runs_path),
        "--latency-path",
        str(request.latency_path),
        "--caller-tts-provider",
        request.caller_tts_provider,
    ]
    if request.today.strip():
        command.extend(["--today", request.today.strip()])
    return command


def format_timestamp(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "unknown"


def format_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.0f} ms"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _scenario_by_id(result: dict[str, Any], scenario_id: str) -> dict[str, Any] | None:
    for scenario in result.get("scenarios", []):
        if isinstance(scenario, dict) and scenario.get("id") == scenario_id:
            return scenario
    return None


def _display_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)


def _run_eval(command: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _init_session_state() -> None:
    import streamlit as st

    st.session_state.setdefault("results_path", str(DEFAULT_RESULTS_PATH))
    st.session_state.setdefault("runs_path", str(DEFAULT_RUNS_PATH))
    st.session_state.setdefault("latency_path", str(DEFAULT_LATENCY_PATH))
    if "run_output_path" not in st.session_state:
        output_path, _runs_path, latency_path = default_run_paths()
        st.session_state["run_output_path"] = str(output_path)
        st.session_state["run_latency_path"] = str(latency_path)


def _render_sidebar() -> tuple[Path, Path, Path, LoadedResult]:
    import streamlit as st

    st.sidebar.header("Artifacts")
    results_path = Path(
        st.sidebar.text_input("results.json", value=st.session_state["results_path"])
    ).expanduser()
    runs_path = Path(st.sidebar.text_input("runs.jsonl", value=st.session_state["runs_path"])).expanduser()
    latency_path = Path(
        st.sidebar.text_input("latency.jsonl", value=st.session_state["latency_path"])
    ).expanduser()

    if st.sidebar.button("Reload", use_container_width=True):
        st.session_state["results_path"] = str(results_path)
        st.session_state["runs_path"] = str(runs_path)
        st.session_state["latency_path"] = str(latency_path)
        st.rerun()

    loaded = load_result_file(results_path, st.session_state.get("selected_batch_key"))
    if loaded.batch_options:
        keys = [option.key for option in loaded.batch_options]
        labels = {option.key: option.label for option in loaded.batch_options}
        selected_key = st.sidebar.selectbox(
            "Batch bot",
            options=keys,
            index=keys.index(loaded.selected_batch_key) if loaded.selected_batch_key in keys else 0,
            format_func=lambda key: labels[key],
        )
        if selected_key != loaded.selected_batch_key:
            st.session_state["selected_batch_key"] = selected_key
            st.rerun()
        st.sidebar.caption("Rendering the selected bot's embedded single-run result.")

    _render_run_eval_panel(runs_path)
    return results_path, runs_path, latency_path, loaded


def _render_run_eval_panel(default_runs_path: Path) -> None:
    import streamlit as st

    with st.sidebar.expander("Advanced: Run eval", expanded=False):
        scenario_ids = [str(scenario["id"]) for scenario in SCENARIOS]
        scenario_id = st.selectbox("Scenario", scenario_ids, index=0)
        bot_url = st.text_input("Bot URL", value=DEFAULT_BOT_URL)
        max_turns = st.number_input("Max turns", min_value=1, max_value=50, value=8, step=1)
        today_value = st.text_input("Today", value=date.today().isoformat())
        caller_tts_provider = st.selectbox("Caller TTS provider", ["gradium", "say"], index=0)
        output_path = Path(st.text_input("Output results", value=st.session_state["run_output_path"]))
        runs_path = Path(st.text_input("Trend runs", value=str(default_runs_path)))
        latency_path = Path(st.text_input("Latency output", value=st.session_state["run_latency_path"]))
        timeout_seconds = st.number_input(
            "Process timeout seconds",
            min_value=30,
            max_value=7200,
            value=900,
            step=30,
        )
        request = EvalRunRequest(
            scenario_id=scenario_id,
            bot_url=bot_url,
            max_turns=int(max_turns),
            today=today_value,
            caller_tts_provider=caller_tts_provider,
            output_path=output_path,
            runs_path=runs_path,
            latency_path=latency_path,
        )
        command = build_eval_command(request)
        st.code(repr(command), language="python")
        if st.button("Run selected eval", use_container_width=True):
            with st.spinner("Running eval_runner.py..."):
                try:
                    completed = _run_eval(command, timeout_seconds=float(timeout_seconds))
                except subprocess.TimeoutExpired as exc:
                    st.error(f"eval_runner.py timed out after {exc.timeout} seconds.")
                    return
            if completed.stdout:
                st.text_area("stdout", completed.stdout, height=120)
            if completed.stderr:
                st.text_area("stderr", completed.stderr, height=120)
            if completed.returncode == 0:
                st.session_state["results_path"] = str(output_path)
                st.session_state["runs_path"] = str(runs_path)
                st.session_state["latency_path"] = str(latency_path)
                st.session_state.pop("selected_batch_key", None)
                st.success("Eval completed. Loaded the new result.")
                st.rerun()
            else:
                st.error(f"eval_runner.py exited with code {completed.returncode}.")


def _render_header(result: dict[str, Any]) -> None:
    import streamlit as st

    st.title("Eval Results")
    st.caption(
        "Run "
        f"{_display_value(result.get('run_id'))} | "
        f"{format_timestamp(result.get('timestamp'))} | "
        f"{_display_value(result.get('agent_source'))}"
    )
    model_cols = st.columns(3)
    model_cols[0].metric("Agent model", _display_value(result.get("agent_model")))
    model_cols[1].metric("Caller model", _display_value(result.get("caller_model")))
    model_cols[2].metric("Judge model", _display_value(result.get("judge_model")))


def _render_kpis(result: dict[str, Any]) -> None:
    import streamlit as st

    counts = pass_fail_counts(result)
    live_latency = result.get("live_latency")
    live_latency = live_latency if isinstance(live_latency, dict) else {}
    status = "Pass" if counts["failed"] == 0 and counts["scenario_count"] else "Fail"
    cols = st.columns(6)
    cols[0].metric("Status", status)
    cols[1].metric("Passed", f"{counts['passed']}/{counts['scenario_count']}")
    cols[2].metric("Pass rate", format_percent(counts["pass_rate"]))
    cols[3].metric("P95 agent reply", format_ms(result.get("p95_agent_reply_ms")))
    cols[4].metric("P95 live voice", format_ms(result.get("p95_latency_ms")))
    cols[5].metric("Live turns", _display_value(live_latency.get("completed_turns")))


def _render_overview(result: dict[str, Any]) -> None:
    import pandas as pd
    import streamlit as st

    failures = failure_summaries(result)
    if failures:
        st.subheader("Failure Summary")
        for failure in failures:
            st.error(f"{failure['id']}: {failure['reason']}")
    else:
        st.success("All loaded scenarios passed.")

    rows = scenario_table_rows(result)
    if rows:
        st.subheader("Scenario Outcomes")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No scenario rows are present in this result.")


def _filtered_scenarios(result: dict[str, Any], status_filter: str, search: str) -> list[str]:
    search = search.strip().lower()
    scenario_ids: list[str] = []
    for scenario in result.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        passed = scenario.get("passed") is True
        if status_filter == "Pass" and not passed:
            continue
        if status_filter == "Fail" and passed:
            continue
        scenario_id = str(scenario.get("id") or "")
        reason = str(scenario.get("reason") or "")
        if search and search not in scenario_id.lower() and search not in reason.lower():
            continue
        scenario_ids.append(scenario_id)
    return scenario_ids


def _render_timing_chart(scenario: dict[str, Any]) -> None:
    import altair as alt
    import pandas as pd
    import streamlit as st

    values = scenario.get("agent_reply_ms")
    if not isinstance(values, list) or not values:
        st.info("No per-turn agent reply timing was recorded.")
        return

    rows = [
        {"turn": index, "agent_reply_ms": value}
        for index, value in enumerate(values, start=1)
        if isinstance(value, int | float)
    ]
    if not rows:
        st.info("No numeric timing values were recorded.")
        return

    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("turn:O", title="Agent reply"),
            y=alt.Y("agent_reply_ms:Q", title="Milliseconds"),
            tooltip=["turn", "agent_reply_ms"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_transcript(scenario: dict[str, Any]) -> None:
    import streamlit as st

    transcript = scenario.get("transcript")
    if not isinstance(transcript, list) or not transcript:
        st.info("No transcript was recorded.")
        return

    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker") or "speaker").title()
        text = str(turn.get("text") or "")
        if speaker.lower() == "agent":
            st.chat_message("assistant").write(text)
        else:
            st.chat_message("user").write(f"**{speaker}:** {text}")


def _render_scenario_evidence(result: dict[str, Any]) -> None:
    import streamlit as st

    cols = st.columns([1, 1, 2])
    status_filter = cols[0].selectbox("Status filter", ["All", "Fail", "Pass"])
    search = cols[1].text_input("Search", value="")
    scenario_ids = _filtered_scenarios(result, status_filter, search)
    if not scenario_ids:
        st.info("No scenarios match the current filters.")
        return

    selected_id = cols[2].selectbox("Scenario", scenario_ids)
    scenario = _scenario_by_id(result, selected_id)
    if scenario is None:
        st.info("Selected scenario is not available.")
        return

    metadata = scenario_metadata().get(selected_id, {})
    verdict = "Pass" if scenario.get("passed") is True else "Fail"
    st.subheader(f"{selected_id} - {verdict}")
    st.write(str(scenario.get("reason") or "No judge reason recorded."))

    if metadata:
        meta_cols = st.columns(2)
        with meta_cols[0]:
            st.markdown("**Persona**")
            st.write(metadata.get("persona") or "n/a")
        with meta_cols[1]:
            st.markdown("**Criteria**")
            st.write(metadata.get("criteria") or "n/a")

    st.markdown("**Timing**")
    _render_timing_chart(scenario)
    st.markdown("**Transcript**")
    _render_transcript(scenario)

    tool_calls = scenario.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        with st.expander(f"Tool calls ({len(tool_calls)})", expanded=False):
            st.json(tool_calls)
    else:
        st.info("No tool calls were recorded for this scenario.")

    with st.expander("Raw scenario JSON", expanded=False):
        st.json(scenario)


def _trend_dataframe(rows: tuple[dict[str, Any], ...]):
    import pandas as pd

    records: list[dict[str, Any]] = []
    for row in rows:
        timestamp = row.get("timestamp")
        try:
            ts_value = pd.to_datetime(float(timestamp), unit="s")
        except (TypeError, ValueError, OSError):
            ts_value = pd.NaT
        records.append(
            {
                "timestamp": ts_value,
                "pass_rate": row.get("pass_rate"),
                "p95_agent_reply_ms": row.get("p95_agent_reply_ms"),
                "p95_latency_ms": row.get("p95_latency_ms"),
                "scenario_count": row.get("scenario_count"),
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    return frame


def _render_trends(runs_path: Path) -> None:
    import altair as alt
    import pandas as pd
    import streamlit as st

    loaded = load_jsonl_rows(runs_path)
    if loaded.error:
        st.error(loaded.error)
        return
    if not loaded.rows:
        st.info(f"No trend rows found at {runs_path}. Run an eval to append runs.jsonl data.")
        return
    if loaded.bad_line_count:
        st.warning(f"Ignored {loaded.bad_line_count} malformed trend row(s).")

    frame = _trend_dataframe(loaded.rows)
    if frame.empty:
        st.info("Trend rows did not contain usable timestamps.")
        return

    pass_chart = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Run"),
            y=alt.Y("pass_rate:Q", title="Pass rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["timestamp", "pass_rate", "scenario_count"],
        )
        .properties(height=220)
    )
    st.altair_chart(pass_chart, use_container_width=True)

    timing = frame.melt(
        id_vars=["timestamp"],
        value_vars=["p95_agent_reply_ms", "p95_latency_ms"],
        var_name="metric",
        value_name="milliseconds",
    ).dropna()
    if timing.empty:
        st.info("No timing metrics are present in the trend rows.")
    else:
        timing_chart = (
            alt.Chart(timing)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Run"),
                y=alt.Y("milliseconds:Q", title="Milliseconds"),
                color=alt.Color("metric:N", title="Metric"),
                tooltip=["timestamp", "metric", "milliseconds"],
            )
            .properties(height=260)
        )
        st.altair_chart(timing_chart, use_container_width=True)

    st.dataframe(pd.DataFrame(loaded.rows), hide_index=True, use_container_width=True)


def _download_button(path: Path, *, label: str, mime: str) -> None:
    import streamlit as st

    if not path.exists():
        st.caption(f"{label}: not found")
        return
    try:
        data = path.read_bytes()
    except OSError as exc:
        st.caption(f"{label}: could not read file: {exc}")
        return
    st.download_button(
        label=f"Download {label}",
        data=data,
        file_name=path.name,
        mime=mime,
        use_container_width=True,
    )


def _render_artifacts(
    result_path: Path,
    runs_path: Path,
    latency_path: Path,
    loaded: LoadedResult,
) -> None:
    import streamlit as st

    st.subheader("Loaded Paths")
    st.write(f"Results: `{result_path}`")
    st.write(f"Runs: `{runs_path}`")
    st.write(f"Latency: `{latency_path}`")
    if loaded.source_kind == "batch":
        selected = next(
            (option for option in loaded.batch_options if option.key == loaded.selected_batch_key),
            None,
        )
        st.write(f"Batch selection: `{selected.label if selected else loaded.selected_batch_key}`")

    cols = st.columns(3)
    with cols[0]:
        _download_button(result_path, label="results.json", mime="application/json")
    with cols[1]:
        _download_button(runs_path, label="runs.jsonl", mime="application/jsonl")
    with cols[2]:
        _download_button(latency_path, label="latency.jsonl", mime="application/jsonl")

    if loaded.result is not None:
        with st.expander("Raw loaded result JSON", expanded=False):
            st.json(loaded.result)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Eval Results", layout="wide")
    _init_session_state()
    result_path, runs_path, latency_path, loaded = _render_sidebar()

    if loaded.error or loaded.result is None:
        st.title("Eval Results")
        st.error(loaded.error or "No result loaded.")
        st.info("Load a results.json file or run a single eval from the advanced sidebar panel.")
        return

    result = loaded.result
    _render_header(result)
    _render_kpis(result)

    overview_tab, evidence_tab, trends_tab, artifacts_tab = st.tabs(
        ["Overview", "Scenario Evidence", "Trends", "Artifacts"]
    )
    with overview_tab:
        _render_overview(result)
    with evidence_tab:
        _render_scenario_evidence(result)
    with trends_tab:
        _render_trends(runs_path)
    with artifacts_tab:
        _render_artifacts(result_path, runs_path, latency_path, loaded)


if __name__ == "__main__":
    main()
