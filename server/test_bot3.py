from datetime import date

from pipecat.processors.aggregators.llm_context import LLMContext

import bot3
import tools


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
    assert "You're all set for Friday, June 5 at two thirty PM" in response
    assert "Your confirmation is BSD" in response


def test_bot3_accumulates_fragmented_booking_details_without_reasking_known_context():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with("Hi Arya.")

    assert processor._response_for_context(context) is None

    context.add_message(
        {"role": "user", "content": "I'd like to schedule a cavity follow up for June fourth"}
    )
    assert processor._response_for_context(context) == "I can help with that. May I have your name?"

    context.add_message({"role": "user", "content": "twenty twenty six at one PM"})
    assert processor._response_for_context(context) == "I can help with that. May I have your name?"

    context.add_message({"role": "user", "content": "Priya Shah. Could you book it?"})
    response = processor._response_for_context(context)

    assert response is not None
    assert "You're all set for Thursday, June 4 at one PM" in response


def test_bot3_updates_memory_on_corrections_before_booking():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with(
        "This is Jamie Lee.",
        "I need an appointment next Tuesday at one PM.",
    )

    assert processor._response_for_context(context) == "What brings you in for the visit?"

    context.add_message(
        {"role": "user", "content": "Actually my name is Jamie Li and two thirty PM is better."}
    )
    assert processor._response_for_context(context) == "What brings you in for the visit?"

    context.add_message({"role": "user", "content": "A routine cleaning."})
    response = processor._response_for_context(context)

    assert response is not None
    assert "You're all set for Tuesday, June 2 at two thirty PM" in response


def test_bot3_defaults_vague_afternoon_to_preferred_open_slot():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with(
        "This is Maria Lopez. I need a routine cleaning next Tuesday afternoon."
    )

    response = processor._response_for_context(context)

    assert response is not None
    assert "You're all set for Tuesday, June 2 at two PM" in response


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
    assert "You're all set for Tuesday, June 2 at two thirty PM" in response


def test_bot3_understands_am_times_and_does_not_repeat_time_question():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with("This is Maria Lopez. I need an appointment on Monday June first.")

    assert processor._response_for_context(context) == "What brings you in for the visit?"

    context.add_message({"role": "user", "content": "It's for dental cleaning place."})
    assert processor._response_for_context(context) == "What time works best for you?"

    context.add_message({"role": "user", "content": "nine o'clock, please."})
    response = processor._response_for_context(context)

    assert response is not None
    assert "What time works best for you?" not in response
    assert "I'm sorry, I do not see nine AM available on Monday, June 1" in response
    assert "I have one PM, two PM, two thirty PM, or four PM" in response


def test_bot3_handles_repeated_am_time_and_books_when_caller_accepts_open_slot():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with(
        "This is Maria Lopez. I need a dental cleaning on Monday June first."
    )

    assert processor._response_for_context(context) == "What time works best for you?"

    context.add_message({"role": "user", "content": "I would like nine a.m. in the morning."})
    response = processor._response_for_context(context)
    assert response is not None
    assert "I'm sorry, I do not see nine AM available on Monday, June 1" in response

    context.add_message({"role": "user", "content": "ten a.m. Monday. June first"})
    response = processor._response_for_context(context)
    assert response is not None
    assert "I'm sorry, I do not see ten AM available on Monday, June 1" in response

    context.add_message({"role": "user", "content": "One PM works."})
    response = processor._response_for_context(context)

    assert response is not None
    assert "You're all set for Monday, June 1 at one PM" in response


def test_bot3_does_not_book_again_after_thanks_or_bye():
    tools.reset_mock_backend()
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with(
        "This is Maria Lopez. I need a dental cleaning on Monday June first."
    )

    assert processor._response_for_context(context) == "What time works best for you?"

    context.add_message({"role": "user", "content": "How about nine am Monday?"})
    response = processor._response_for_context(context)
    assert response is not None
    assert "I'm sorry, I do not see nine AM available on Monday, June 1" in response

    context.add_message({"role": "user", "content": "How about two pm?"})
    response = processor._response_for_context(context)
    assert response is not None
    assert "You're all set for Monday, June 1 at two PM. Your confirmation is BSD1002." == response

    context.add_message({"role": "user", "content": "That's awesome. Thank you so much."})
    assert processor._response_for_context(context) == "You're very welcome."

    context.add_message({"role": "user", "content": "Okay, bye now."})
    assert (
        processor._response_for_context(context)
        == "Thanks for calling Bright Smile Dental. Have a good day."
    )

    created_bookings = [
        confirmation_id for confirmation_id in tools._BOOKINGS if confirmation_id != "BSD1001"
    ]
    assert created_bookings == ["BSD1002"]


def test_bot3_morning_without_specific_open_slot_offers_available_slots():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    context = _context_with(
        "This is Maria Lopez. I need a dental cleaning on Monday June first in the morning."
    )

    response = processor._response_for_context(context)

    assert response is not None
    assert "I'm sorry, I do not see morning slots on Monday, June 1" in response
    assert "I have one PM, two PM, two thirty PM, or four PM" in response


def test_bot3_answers_policy_and_insurance_without_llm():
    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))

    assert (
        processor._response_for_context(_context_with("Do you accept Guardian insurance?"))
        == "The office can confirm Guardian coverage for you."
    )

    processor = bot3.MemoryFirstFrontDeskProcessor(today=date(2026, 5, 30))
    assert (
        processor._response_for_context(_context_with("Can you pull up my chart from my phone number?"))
        == "I'm not able to identify you or pull up records from caller ID alone."
    )
