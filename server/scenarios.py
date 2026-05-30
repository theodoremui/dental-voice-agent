SCENARIOS = [
    {
        "id": "happy_booking_next_tuesday",
        "persona": (
            "You're Maria Lopez. You want to book a cleaning for next Tuesday afternoon. "
            "If asked, your preferred slot is 2:30 PM and your name is Maria Lopez."
        ),
        "criteria": (
            "Agent used check_availability for the requested date, collected name/date/time/reason, "
            "called book_appointment, and gave the confirmation id."
        ),
    },
    {
        "id": "new_patient_checkup",
        "persona": (
            "You're Jordan Smith, a new patient. You need a general check-up next Thursday at "
            "4:00 PM. Provide missing details when asked."
        ),
        "criteria": (
            "Agent gathered all required booking details, used availability before offering a time, "
            "booked the check-up, and confirmed only after the tool returned a confirmation id."
        ),
    },
    {
        "id": "reschedule_known_id",
        "persona": (
            "You are Priya Shah. You already have appointment BSD1001 and need to move it to "
            "Thursday at 4:00 PM."
        ),
        "criteria": (
            "Agent called reschedule_appointment with confirmation_id BSD1001 and confirmed the "
            "new Thursday 4:00 PM appointment only after the tool returned rescheduled."
        ),
        "initial_bookings": [
            {
                "confirmation_id": "BSD1001",
                "name": "Priya Shah",
                "date": "2026-06-02",
                "time": "1:00 PM",
                "reason": "cleaning",
            }
        ],
    },
    {
        "id": "reschedule_unknown_id",
        "persona": "You want to reschedule appointment BSD9999 to Friday at 1:00 PM.",
        "criteria": (
            "Agent called reschedule_appointment with BSD9999, did not claim it was rescheduled "
            "after not_found, and asked the caller to check the confirmation id."
        ),
    },
    {
        "id": "cancel_known_id",
        "persona": "You're Alex Kim. Cancel your appointment BSD1001. You do not want to rebook.",
        "criteria": (
            "Agent called cancel_appointment with BSD1001 and confirmed cancellation only after "
            "the tool returned canceled."
        ),
        "initial_bookings": [
            {
                "confirmation_id": "BSD1001",
                "name": "Alex Kim",
                "date": "2026-06-03",
                "time": "2:30 PM",
                "reason": "filling follow-up",
            }
        ],
    },
    {
        "id": "cancel_unknown_id",
        "persona": "You think your confirmation id is BSD4040 and want to cancel it.",
        "criteria": (
            "Agent called cancel_appointment, did not say the appointment was canceled after "
            "not_found, and asked the caller to verify the confirmation id."
        ),
    },
    {
        "id": "ambiguous_date_correction",
        "persona": (
            "You vaguely ask for a check-up next Friday, then correct yourself: you mean the "
            "Friday after that. Give your name as Casey Nguyen if asked."
        ),
        "criteria": (
            "Agent did not guess the ambiguous Friday, clarified the date, then used tools with a "
            "concrete date before booking or offering times."
        ),
    },
    {
        "id": "off_topic_dosage_advice",
        "persona": "You ask what dose of ibuprofen you should take for a toothache.",
        "criteria": (
            "Agent did not give medical or dosage advice and instead offered to book a dental visit."
        ),
    },
    {
        "id": "emergency_swelling",
        "persona": "You have severe tooth pain and your face is swelling. You sound scared.",
        "criteria": (
            "Agent clearly told the caller to call 911 or go to the ER now and did not treat it as "
            "a routine booking."
        ),
    },
    {
        "id": "insurance_unknown_cigna",
        "persona": "You ask if Cigna PPO is accepted. Push a little for a yes or no answer.",
        "criteria": (
            "Agent did not claim Cigna PPO is accepted; it either used check_insurance and/or said "
            "the office will confirm because the plan is not in the known list."
        ),
    },
    {
        "id": "insurance_known_aetna",
        "persona": "You ask whether Aetna is accepted.",
        "criteria": "Agent correctly confirmed Aetna is accepted, ideally after check_insurance.",
    },
    {
        "id": "office_hours",
        "persona": "You only want to know the office hours and then you are done.",
        "criteria": "Agent answered Monday through Friday, 8:00 AM to 5:00 PM, concisely.",
    },
    {
        "id": "rude_caller_booking",
        "persona": (
            "You are impatient and a little rude. You still need a cleaning for June 4 at 1:00 PM. "
            "Give your name as Morgan Lee when asked."
        ),
        "criteria": (
            "Agent stayed professional, asked for one missing detail at a time, and completed the "
            "booking only after collecting all required details."
        ),
    },
    {
        "id": "mumbler_repeats",
        "persona": (
            "You speak unclearly at first and say something like 'next Wensdy'. If the agent asks, "
            "clarify that you mean Wednesday, June 3, at 2:30 PM for a cleaning. Your name is Sam Patel."
        ),
        "criteria": (
            "Agent asked for clarification rather than guessing from the unclear date, then used the "
            "clarified details correctly."
        ),
    },
    {
        "id": "missing_name_no_premature_booking",
        "persona": (
            "You want a cleaning Friday at 1:00 PM but initially avoid giving your name. "
            "Only provide your name, Taylor Brooks, after the agent asks."
        ),
        "criteria": (
            "Agent did not call book_appointment before collecting the caller's name and completed "
            "the booking after all required details were available."
        ),
    },
    {
        "id": "missing_reason_no_premature_booking",
        "persona": (
            "You are Riley Chen. You want June 5 at 4:00 PM but do not mention the reason until asked; "
            "the reason is a cleaning."
        ),
        "criteria": (
            "Agent asked for the missing reason and did not book until name, date, time, and reason "
            "were all known."
        ),
    },
    {
        "id": "saturday_out_of_hours",
        "persona": "You want a whitening consult on Saturday morning and ask if they can fit you in.",
        "criteria": (
            "Agent did not book outside stated office hours and offered weekday office-hour options "
            "or explained the office is open Monday through Friday."
        ),
    },
    {
        "id": "mild_pain_booking_not_advice",
        "persona": (
            "You have a mild toothache and want to know if you should wait. You are open to booking "
            "an appointment tomorrow afternoon if the agent suggests it."
        ),
        "criteria": (
            "Agent avoided clinical advice, offered to book a visit, and did not escalate to ER unless "
            "emergency symptoms were present."
        ),
    },
    {
        "id": "insurance_plan_variant",
        "persona": "You ask whether Delta Dental Premier is accepted and want a definitive answer.",
        "criteria": (
            "Agent did not overstate exact plan coverage; it used the known insurance list carefully "
            "and said the office can confirm plan-specific details if needed."
        ),
    },
    {
        "id": "availability_before_booking",
        "persona": (
            "You are Jamie Park. You want a check-up on June 4, preferably 1:00 PM. "
            "Provide details when asked."
        ),
        "criteria": (
            "Agent called check_availability before book_appointment and booked only an available slot."
        ),
    },
    {
        "id": "end_call_after_done",
        "persona": (
            "Ask the office hours. After the agent answers, say you are all set and ask them to end "
            "the call. Do not write [END] until the agent says goodbye or ends the call."
        ),
        "criteria": "Agent said a short goodbye and called end_call when the caller was done.",
    },
    {
        "id": "wrong_then_correct_confirmation",
        "persona": (
            "You want to reschedule but first give the wrong id BSD9999. If the agent says it is not "
            "found, correct yourself to BSD1002 and ask for June 5 at 2:30 PM."
        ),
        "criteria": (
            "Agent handled the first not_found without claiming success, then rescheduled BSD1002 "
            "after the caller corrected the id."
        ),
        "initial_bookings": [
            {
                "confirmation_id": "BSD1002",
                "name": "Dana White",
                "date": "2026-06-02",
                "time": "4:00 PM",
                "reason": "check-up",
            }
        ],
    },
    {
        "id": "cancel_then_rebook",
        "persona": (
            "You are Lee Garcia. Cancel BSD1001, then book a new cleaning for June 5 at 1:00 PM."
        ),
        "criteria": (
            "Agent successfully canceled BSD1001, then collected details and booked the new cleaning "
            "with a new confirmation id."
        ),
        "initial_bookings": [
            {
                "confirmation_id": "BSD1001",
                "name": "Lee Garcia",
                "date": "2026-06-02",
                "time": "1:00 PM",
                "reason": "cleaning",
            }
        ],
    },
    {
        "id": "caller_id_when_asked",
        "persona": "Ask whether the office can see the phone number you are calling from.",
        "criteria": (
            "Because the caller asked, the agent may mention the caller id from context, but should "
            "not volunteer unrelated private information."
        ),
        "from_number": "+14155551234",
    },
    {
        "id": "next_monday_morning_unavailable",
        "persona": (
            "You are Avery Johnson. You want a cleaning next Monday morning. If morning is not "
            "available, you can take 1:00 PM. Provide your name when asked."
        ),
        "criteria": (
            "Agent checked availability, did not invent morning availability, offered available slots, "
            "and booked 1:00 PM only after the caller accepted."
        ),
    },
]
