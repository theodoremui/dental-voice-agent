from datetime import date

from pipecat.processors.aggregators.llm_context import LLMContext

import bot3


def _context_with(*texts: str) -> LLMContext:
    context = LLMContext()
    for text in texts:
        context.add_message({"role": "user", "content": text})
    return context


def test_bot3_remembers_name_given_before_booking_request():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with("I am Devin Carter.")

    assert processor._response_for_context(context) is None

    context.add_message(
        {
            "role": "user",
            "content": (
                "I am a new patient and want a first visit this Friday at two thirty PM "
                "because of tooth sensitivity."
            ),
        }
    )
    response = processor._response_for_context(context)

    assert response is not None
    assert "Booked for Friday, June 5 at two thirty PM" in response
    assert "Confirmation BSD" in response


def test_bot3_accumulates_fragmented_booking_details_without_reasking_known_context():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with("Hi Arya.")

    assert processor._response_for_context(context) is None

    context.add_message(
        {"role": "user", "content": "I'd like to schedule a cavity follow up for June fourth"}
    )
    assert processor._response_for_context(context) == "May I have your name?"

    context.add_message({"role": "user", "content": "twenty twenty six at one PM"})
    assert processor._response_for_context(context) == "May I have your name?"

    context.add_message({"role": "user", "content": "Priya Shah. Could you book it?"})
    response = processor._response_for_context(context)

    assert response is not None
    assert "Booked for Thursday, June 4 at one PM" in response


def test_bot3_updates_memory_on_corrections_before_booking():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with(
        "This is Jamie Lee.",
        "I need an appointment next Tuesday at one PM.",
    )

    assert processor._response_for_context(context) == "What is the reason for the visit?"

    context.add_message(
        {"role": "user", "content": "Actually my name is Jamie Li and two thirty PM is better."}
    )
    assert processor._response_for_context(context) == "What is the reason for the visit?"

    context.add_message({"role": "user", "content": "A routine cleaning."})
    response = processor._response_for_context(context)

    assert response is not None
    assert "Booked for Tuesday, June 2 at two thirty PM" in response


def test_bot3_defaults_vague_afternoon_to_preferred_open_slot():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with(
        "This is Maria Lopez. I need a routine cleaning next Tuesday afternoon."
    )

    response = processor._response_for_context(context)

    assert response is not None
    assert "Booked for Tuesday, June 2 at two PM" in response


def test_bot3_offers_slots_when_caller_asks_for_options():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with(
        "This is Maria Lopez. What afternoon options are open next Tuesday for a cleaning?"
    )

    response = processor._response_for_context(context)

    assert response is not None
    assert "I have one PM, two PM, two thirty PM, or four PM" in response

    context.add_message({"role": "user", "content": "Two thirty PM works."})
    response = processor._response_for_context(context)

    assert response is not None
    assert "Booked for Tuesday, June 2 at two thirty PM" in response


def test_bot3_answers_policy_and_insurance_without_llm():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))

    assert (
        processor._response_for_context(_context_with("Do you accept Guardian insurance?"))
        == "The office will confirm Guardian coverage for you."
    )

    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    assert (
        processor._response_for_context(_context_with("Can you pull up my chart from my phone number?"))
        == "I cannot identify you or pull up records from caller ID alone."
    )
