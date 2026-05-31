from tools import TOOL_IMPLS, _spoken_tool_response, build_system_instruction


def test_system_instruction_guides_availability_before_vague_time():
    instruction = build_system_instruction()

    assert "vague time of day" in instruction
    assert "call check_availability" in instruction
    assert "already gave name, date, time, and reason" in instruction


def test_mock_availability_includes_common_afternoon_slots():
    result = TOOL_IMPLS["check_availability"]({"date": "2026-06-02"})

    assert result["date"] == "2026-06-02"
    assert "2:00 PM" in result["open_slots"]
    assert "2:30 PM" in result["open_slots"]


def test_spoken_tool_responses_avoid_post_tool_llm_hop():
    availability = _spoken_tool_response(
        "check_availability",
        {"date": "2026-06-02"},
        {"date": "2026-06-02", "open_slots": ["1:00 PM", "2:00 PM", "2:30 PM"]},
    )
    booking = _spoken_tool_response(
        "book_appointment",
        {},
        {
            "confirmation_id": "BSD1002",
            "date": "2026-06-02",
            "time": "2:30 PM",
        },
    )

    assert availability == (
        "two PM on Tuesday, June 2 is open. Should I book it?"
    )
    assert booking == (
        "You're all set for Tuesday, June 2 at two thirty PM. Your confirmation is BSD1002."
    )
