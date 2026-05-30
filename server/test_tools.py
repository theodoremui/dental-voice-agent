import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools  # noqa: E402


def setup_function():
    tools._BOOKINGS.clear()
    tools._next_id[0] = 1000


def test_book_reschedule_and_cancel_appointment():
    booked = tools.TOOL_IMPLS["book_appointment"](
        {
            "name": "Casey Lee",
            "date": "2026-06-02",
            "time": "1:00 PM",
            "reason": "cleaning",
        }
    )

    assert booked["status"] == "booked"
    assert booked["confirmation_id"] == "BSD1001"

    rescheduled = tools.TOOL_IMPLS["reschedule_appointment"](
        {
            "confirmation_id": booked["confirmation_id"],
            "date": "2026-06-03",
            "time": "2:30 PM",
        }
    )

    assert rescheduled["status"] == "rescheduled"
    assert rescheduled["date"] == "2026-06-03"
    assert rescheduled["time"] == "2:30 PM"

    canceled = tools.TOOL_IMPLS["cancel_appointment"](
        {"confirmation_id": booked["confirmation_id"]}
    )

    assert canceled["status"] == "canceled"
    assert canceled["confirmation_id"] == booked["confirmation_id"]
    assert tools.TOOL_IMPLS["cancel_appointment"](
        {"confirmation_id": booked["confirmation_id"]}
    ) == {"status": "not_found", "confirmation_id": booked["confirmation_id"]}


def test_unknown_confirmation_id_returns_not_found():
    assert tools.TOOL_IMPLS["reschedule_appointment"](
        {"confirmation_id": "BSD9999", "date": "2026-06-02", "time": "1:00 PM"}
    ) == {"status": "not_found", "confirmation_id": "BSD9999"}

    assert tools.TOOL_IMPLS["cancel_appointment"]({"confirmation_id": "BSD9999"}) == {
        "status": "not_found",
        "confirmation_id": "BSD9999",
    }


def test_insurance_known_list_only():
    assert tools.TOOL_IMPLS["check_insurance"]({"provider": "Delta Dental"})["accepted"] is True
    assert tools.TOOL_IMPLS["check_insurance"]({"provider": "MetLife"})["accepted"] is True
    assert tools.TOOL_IMPLS["check_insurance"]({"provider": "Aetna"})["accepted"] is True
    assert tools.TOOL_IMPLS["check_insurance"]({"provider": "Cigna"}) == {
        "provider": "Cigna",
        "accepted": False,
        "known_list_only": True,
    }


def test_pipecat_schema_exposes_all_tool_names():
    schema = tools.build_pipecat_tools_schema()
    names = [tool.name for tool in schema.standard_tools]
    assert names == list(tools.TOOL_IMPLS)


def test_register_pipecat_functions_registers_and_invokes_handlers():
    class FakeLLM:
        def __init__(self):
            self.handlers = {}

        def register_function(self, name, handler):
            self.handlers[name] = handler

    class FakeParams:
        arguments = {"date": "2026-06-02"}

        def __init__(self):
            self.results = []

        async def result_callback(self, result, properties=None):
            self.results.append((result, properties))

    llm = FakeLLM()
    tools.register_pipecat_functions(llm)

    assert set(llm.handlers) == set(tools.TOOL_IMPLS)

    params = FakeParams()
    asyncio.run(llm.handlers["check_availability"](params))
    assert params.results == [
        ({"date": "2026-06-02", "open_slots": ["1:00 PM", "2:30 PM", "4:00 PM"]}, None)
    ]
