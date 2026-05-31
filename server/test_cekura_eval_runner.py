import json
from pathlib import Path

from cekura_eval_runner import (
    SOURCE_TAG,
    build_evaluator_payload,
    build_metric_payload,
    build_observability_payload,
    ingest_local_results,
    scenario_name,
    sync_evaluators,
    transcript_to_cekura,
)
from scenarios import SCENARIOS


class FakeCekuraClient:
    def __init__(self, evaluators=None):
        self.evaluators = list(evaluators or [])
        self.created_evaluators = []
        self.ingested_calls = []

    def list_evaluators(self, **kwargs):
        return self.evaluators

    def create_evaluator(self, payload):
        self.created_evaluators.append(payload)
        return {"id": 1000 + len(self.created_evaluators), "name": payload["name"]}

    def ingest_call(self, payload):
        self.ingested_calls.append(payload)
        return {"id": 2000 + len(self.ingested_calls), "call_id": payload["call_id"]}


def test_build_evaluator_payload_maps_existing_scenario_fields():
    scenario = SCENARIOS[0]

    payload = build_evaluator_payload(
        scenario,
        personality_id=42,
        agent_id=2142,
        metric_ids=[501],
        name_prefix="BSD",
    )

    assert payload["name"] == scenario_name(scenario, name_prefix="BSD")
    assert payload["agent"] == 2142
    assert payload["personality"] == 42
    assert payload["metrics"] == [501]
    assert payload["scenario_type"] == "instruction"
    assert scenario["persona"] in payload["instructions"]
    assert scenario["criteria"] in payload["instructions"]
    assert payload["expected_outcome_prompt"] == scenario["criteria"]
    assert SOURCE_TAG in payload["tags"]
    assert f"scenario:{scenario['id']}" in payload["tags"]


def test_build_metric_payload_uses_single_shared_llm_judge_metric():
    payload = build_metric_payload(
        project_id=99,
        assistant_id="asst_bright_smile",
        agent_ids=[2142],
    )

    assert payload["project"] == 99
    assert payload["assistant_id"] == "asst_bright_smile"
    assert payload["agents"] == [2142]
    assert payload["type"] == "llm_judge"
    assert payload["eval_type"] == "binary_workflow_adherence"
    assert payload["simulation_enabled"] is True
    assert payload["observability_enabled"] is True
    assert "{{transcript}}" in payload["prompt"]
    assert "{{dynamic_variables.criteria}}" in payload["prompt"]
    assert "{{evaluator.instructions}}" in payload["prompt"]


def test_sync_evaluators_skips_existing_scenario_and_creates_missing():
    scenarios = SCENARIOS[:2]
    existing = {
        "id": 10,
        "name": scenario_name(scenarios[0]),
        "tags": [SOURCE_TAG, f"scenario:{scenarios[0]['id']}"],
    }
    client = FakeCekuraClient([existing])

    result = sync_evaluators(
        client,
        scenarios,
        personality_id=42,
        agent_id=2142,
        metric_ids=[501],
    )

    assert result["existing_count"] == 1
    assert result["created_count"] == 1
    assert result["items"][0]["status"] == "exists"
    assert result["items"][1]["status"] == "created"
    assert len(client.created_evaluators) == 1
    assert client.created_evaluators[0]["expected_outcome_prompt"] == scenarios[1]["criteria"]


def test_transcript_to_cekura_uses_required_roles_and_monotonic_times():
    transcript = [
        {"speaker": "caller", "text": "I need a cleaning."},
        {"speaker": "agent", "text": "I can help with that."},
    ]

    converted = transcript_to_cekura(transcript)

    assert converted[0]["role"] == "Testing Agent"
    assert converted[1]["role"] == "Main Agent"
    assert converted[0]["start_time"] == 0.0
    assert converted[0]["end_time"] < converted[1]["start_time"]
    assert converted[1]["content"] == "I can help with that."


def test_observability_payload_preserves_local_verdict_context():
    record = {
        "scenario": {
            "id": "insurance_known_delta",
            "category": "insurance",
            "passed": True,
            "reason": "Correctly said Delta Dental is accepted.",
            "transcript": [
                {"speaker": "caller", "text": "Do you take Delta Dental?"},
                {"speaker": "agent", "text": "Yes, we accept Delta Dental."},
            ],
        },
        "bot_id": "bot1",
        "bot_name": "bot1",
        "run_id": "run-1",
        "batch_id": "batch-1",
        "timestamp_iso": "2026-05-30T22:00:00Z",
        "source": "results",
    }

    payload = build_observability_payload(
        record,
        agent_id=2142,
        metric_ids=[501, 502],
    )

    assert payload is not None
    assert payload["agent"] == 2142
    assert payload["metric_ids"] == "501,502"
    assert payload["transcript_type"] == "cekura"
    assert payload["transcript_json"][0]["role"] == "Testing Agent"
    assert payload["transcript_json"][1]["role"] == "Main Agent"
    assert payload["dynamic_variables"]["criteria"] == SCENARIOS[10]["criteria"]
    assert payload["dynamic_variables"]["local_passed"] is True
    assert payload["metadata"]["source"] == "cekura_eval_runner.py"


def test_ingest_local_results_dry_run_builds_payloads_from_results_json(tmp_path: Path):
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "run_id": "run-abc",
                "timestamp_iso": "2026-05-30T22:00:00Z",
                "voice_agent": {"name": "bot1", "batch_id": "batch-abc"},
                "scenarios": [
                    {
                        "id": "goodbye_end_call",
                        "category": "call_closure",
                        "passed": False,
                        "reason": "Did not end the call.",
                        "transcript": [
                            {"speaker": "caller", "text": "I called by mistake, goodbye."},
                            {"speaker": "agent", "text": "Goodbye."},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = ingest_local_results(
        FakeCekuraClient(),
        results_path=results_path,
        assistant_id="asst_bright_smile",
        metric_ids=[501],
        dry_run=True,
    )

    assert result["would_ingest_count"] == 1
    item = result["items"][0]
    assert item["status"] == "would_ingest"
    assert item["payload"]["assistant_id"] == "asst_bright_smile"
    assert item["payload"]["dynamic_variables"]["scenario_id"] == "goodbye_end_call"
