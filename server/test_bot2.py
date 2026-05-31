import inspect
from datetime import date

from pipecat.processors.aggregators.llm_context import LLMContext

import bot2


def test_twilio_sample_rate_contract(monkeypatch):
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token")

    assert bot2.twilio_transport_overrides() == {
        "audio_in_sample_rate": 16000,
        "audio_out_sample_rate": 8000,
    }

    serializer = bot2.build_twilio_serializer(
        {
            "stream_id": "MZ123",
            "call_id": "CA123",
        }
    )
    assert serializer._params.sample_rate == 16000
    assert serializer._params.twilio_sample_rate == 8000

    stt = bot2.build_stt_service()
    assert stt._init_sample_rate == 16000


def test_bot2_uses_default_turn_strategy_without_incomplete_filtering():
    params = bot2.build_user_aggregator_params(
        vad_analyzer=bot2.build_vad_analyzer(twilio=False, audio_in_sample_rate=16000)
    )

    assert params.user_turn_strategies is None
    assert params.filter_incomplete_user_turns is False
    source = inspect.getsource(bot2)
    assert "FilterIncompleteUserTurnStrategies" not in source
    assert "filter_incomplete_user_turns=True" not in source.replace(" ", "")


def test_twilio_vad_defaults_and_env_overrides(monkeypatch):
    defaults = bot2.build_twilio_vad_params()
    assert defaults.confidence == 0.55
    assert defaults.start_secs == 0.12
    assert defaults.stop_secs == 0.25
    assert defaults.min_volume == 0.35

    monkeypatch.setenv("VOICE_VAD_CONFIDENCE", "0.6")
    monkeypatch.setenv("VOICE_VAD_START_SECS", "0.15")
    monkeypatch.setenv("VOICE_VAD_STOP_SECS", "0.3")
    monkeypatch.setenv("VOICE_VAD_MIN_VOLUME", "0.4")

    overridden = bot2.build_twilio_vad_params()
    assert overridden.confidence == 0.6
    assert overridden.start_secs == 0.15
    assert overridden.stop_secs == 0.3
    assert overridden.min_volume == 0.4


def test_webrtc_vad_coalesces_short_eval_pauses_and_is_separate_from_twilio(monkeypatch):
    defaults = bot2.build_webrtc_vad_params()
    assert defaults.confidence == 0.55
    assert defaults.start_secs == 0.12
    assert defaults.stop_secs == 0.55
    assert defaults.min_volume == 0.35

    twilio_defaults = bot2.build_twilio_vad_params()
    assert twilio_defaults.stop_secs == 0.25

    analyzer = bot2.build_vad_analyzer(twilio=False, audio_in_sample_rate=16000)
    assert analyzer.params.stop_secs == 0.55

    monkeypatch.setenv("VOICE_WEBRTC_VAD_STOP_SECS", "0.65")
    monkeypatch.setenv("VOICE_VAD_STOP_SECS", "0.3")

    overridden = bot2.build_webrtc_vad_params()
    assert overridden.stop_secs == 0.65
    assert bot2.build_twilio_vad_params().stop_secs == 0.3


def test_bot2_llm_default_token_ceiling_allows_tool_turns():
    assert bot2.DEFAULT_NEMOTRON_LLM_TEMPERATURE == 0.2
    assert bot2.DEFAULT_NEMOTRON_LLM_MAX_TOKENS == 240


def test_appointment_fast_path_books_exact_request():
    context = LLMContext()
    context.add_message(
        {
            "role": "user",
            "content": (
                "Hi, this is Priya Shah. I need a cavity follow-up on June 4, "
                "2026 at 1 PM."
            ),
        }
    )
    processor = bot2.AppointmentFastPathProcessor(today=date(2026, 5, 30))

    response = processor._response_for_context(context)

    assert response is not None
    assert "Booked for Thursday, June 4 at one PM" in response
    assert "confirmation BSD" in response


def test_appointment_fast_path_confirms_default_afternoon_slot():
    context = LLMContext()
    context.add_message(
        {
            "role": "user",
            "content": (
                "Hi, this is Maria Lopez. I'd like to schedule a routine cleaning "
                "for next Tuesday afternoon."
            ),
        }
    )
    processor = bot2.AppointmentFastPathProcessor(today=date(2026, 5, 30))

    response = processor._response_for_context(context)

    assert response is not None
    assert "Booked for Tuesday, June 2 at two PM" in response
    assert "confirmation BSD" in response


def test_appointment_fast_path_merges_bare_name_followup():
    processor = bot2.AppointmentFastPathProcessor(today=date(2026, 5, 30))
    context = LLMContext()
    context.add_message(
        {
            "role": "user",
            "content": "I need an appointment next Monday at 4 PM for a chipped filling.",
        }
    )

    assert processor._response_for_context(context) == "May I have your name?"

    context.add_message({"role": "user", "content": "Jordan Reed."})
    response = processor._response_for_context(context)

    assert response is not None
    assert "Booked for Monday, June 1 at four PM" in response
