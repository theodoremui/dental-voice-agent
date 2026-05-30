from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import batch_eval_runner


def _args(tmp_path: Path, bots: list[Path], **overrides):
    values = {
        "bots": [str(bot) for bot in bots],
        "labels": None,
        "ports": [9100 + index for index in range(len(bots))],
        "host": "127.0.0.1",
        "start_port": 7860,
        "artifacts_dir": str(tmp_path / "artifacts"),
        "uv": "uv",
        "eval_runner": "eval_runner.py",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_specs_uses_unique_ports_and_artifact_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_eval_runner, "_port_is_free", lambda host, port: True)
    bot0 = tmp_path / "bot0.py"
    bot1 = tmp_path / "bot1.py"
    bot0.write_text("async def bot(runner_args): pass\n", encoding="utf-8")
    bot1.write_text("async def bot(runner_args): pass\n", encoding="utf-8")

    specs = batch_eval_runner._build_specs(_args(tmp_path, [bot0, bot1]), batch_id="batch")

    assert [spec.label for spec in specs] == ["bot0", "bot1"]
    assert [spec.port for spec in specs] == [9100, 9101]
    assert specs[0].bot_url == "http://127.0.0.1:9100"
    assert specs[0].result_path == tmp_path / "artifacts" / "batch" / "01-bot0" / "results.json"
    assert specs[1].latency_path == tmp_path / "artifacts" / "batch" / "02-bot1" / "latency.jsonl"


def test_build_specs_disambiguates_duplicate_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_eval_runner, "_port_is_free", lambda host, port: True)
    first = tmp_path / "a" / "bot.py"
    second = tmp_path / "b" / "bot.py"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("async def bot(runner_args): pass\n", encoding="utf-8")
    second.write_text("async def bot(runner_args): pass\n", encoding="utf-8")

    specs = batch_eval_runner._build_specs(_args(tmp_path, [first, second]), batch_id="batch")

    assert [spec.label for spec in specs] == ["bot", "bot-2"]


def test_explicit_ports_must_be_free(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_eval_runner, "_port_is_free", lambda host, port: False)
    bot = tmp_path / "bot0.py"
    bot.write_text("async def bot(runner_args): pass\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="already in use"):
        batch_eval_runner._build_specs(_args(tmp_path, [bot]), batch_id="batch")


def test_commands_pin_bot_url_and_isolated_output_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_eval_runner, "_port_is_free", lambda host, port: True)
    bot = tmp_path / "bot0.py"
    bot.write_text("async def bot(runner_args): pass\n", encoding="utf-8")
    args = _args(tmp_path, [bot])
    spec = batch_eval_runner._build_specs(args, batch_id="batch")[0]

    assert batch_eval_runner._bot_command(args, spec) == [
        "uv",
        "run",
        str(bot.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        "9100",
        "-t",
        "webrtc",
    ]
    eval_command = batch_eval_runner._eval_command(args, spec, ["--scenario", "office_hours"])

    assert eval_command[:4] == ["uv", "run", "python", "eval_runner.py"]
    assert "--scenario" in eval_command
    assert eval_command[-8:] == [
        "--bot-url",
        "http://127.0.0.1:9100",
        "--output",
        str(spec.result_path),
        "--runs",
        str(spec.eval_runs_path),
        "--latency-path",
        str(spec.latency_path),
    ]


def test_result_metadata_and_summary_capture_specific_bot(monkeypatch, tmp_path):
    monkeypatch.setattr(batch_eval_runner, "_port_is_free", lambda host, port: True)
    bot = tmp_path / "bot0.py"
    bot.write_text("async def bot(runner_args): pass\n", encoding="utf-8")
    spec = batch_eval_runner._build_specs(_args(tmp_path, [bot]), batch_id="batch")[0]
    result = {
        "run_id": "abc123",
        "timestamp": 1.5,
        "mode": "voice",
        "agent_source": "running Pipecat WebRTC voice bot at http://localhost:7860",
        "scenario_count": 2,
        "passed": 1,
        "pass_rate": 0.5,
        "p95_agent_reply_ms": 1234,
        "p95_latency_ms": 2345,
        "live_latency": {
            "completed_turns": 3,
            "p95_ttfa_ms": 456,
            "p95_ttla_ms": 2345,
        },
        "scenarios": [
            {"id": "office_hours", "passed": True},
            {"id": "emergency_swelling", "passed": False},
        ],
    }

    enriched = batch_eval_runner._inject_agent_metadata(result, spec)
    summary = batch_eval_runner._summary_from_result(
        spec,
        enriched,
        batch_id="batch",
        batch_timestamp=1.0,
        status="completed",
        duration_ms=5000,
    )

    assert enriched["evaluated_agent"]["bot"] == "bot0"
    assert enriched["evaluated_agent"]["bot_url"] == "http://127.0.0.1:9100"
    assert enriched["agent_source"].startswith("bot0 (")
    assert summary["bot"] == "bot0"
    assert summary["bot_path"] == str(bot.resolve())
    assert summary["pass_rate"] == 0.5
    assert summary["live_p95_ttfa_ms"] == 456
    assert summary["failure_ids"] == ["emergency_swelling"]
