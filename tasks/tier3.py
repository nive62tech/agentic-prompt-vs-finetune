"""
Tier 3 — Error recovery.

Two task types:

1. type="error": the natural first tool call returns an error (city not
   found, no flights, unsupported currency, item not found, invalid ID,
   no calendar data). Success = the model (a) still attempts the correct
   first call, (b) does NOT proceed to call a downstream tool with
   fabricated/guessed data once it hits the error, and (c) its final
   answer surfaces the problem to the user rather than claiming success.

2. type="ambiguous": the tool call SUCCEEDS but returns a result that
   contradicts an implicit assumption (e.g. two calendar events at the
   same time). Success = the model notices and flags the contradiction
   in its final answer, rather than reporting the raw result uncritically.

`trigger_tool` / `trigger_args`: the call expected to surface the
    error/ambiguity.
`must_not_call`: tool names the model should NOT call afterward (calling
    them typically means it's proceeding on fabricated data).
`recovery_keywords`: if the model's final text contains ANY of these
    (case-insensitive substring match), that counts as successful
    recovery/flagging. Keep this permissive — it's a coarse signal, not
    a strict parser; tune it once you see real model outputs.
"""

TIER3_TASKS = [
    {"id": "t3_001", "type": "error",
     "prompt": "What's the weather in Atlantis?",
     "trigger_tool": "get_weather", "trigger_args": {"city": "Atlantis"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "couldn't find", "don't have", "no data", "unable", "no weather", "doesn't exist", "not available"]},

    {"id": "t3_002", "type": "error",
     "prompt": "Find a flight from Cairo to Sydney and book it.",
     "trigger_tool": "search_flights", "trigger_args": {"origin": "Cairo", "destination": "Sydney"},
     "must_not_call": ["book_flight"],
     "recovery_keywords": ["no flights", "not found", "unavailable", "couldn't find", "no route", "unable"]},

    {"id": "t3_003", "type": "error",
     "prompt": "Convert 100 USD to XAU.",
     "trigger_tool": "convert_currency", "trigger_args": {"amount": 100, "base": "USD", "target": "XAU"},
     "must_not_call": [],
     "recovery_keywords": ["unsupported", "not supported", "don't support", "can't convert", "not available", "unable"]},

    {"id": "t3_004", "type": "error",
     "prompt": "What's the price of a smartwatch?",
     "trigger_tool": "lookup_price", "trigger_args": {"item": "smartwatch"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "don't have", "not available", "not in", "unable", "no price"]},

    {"id": "t3_005", "type": "error",
     "prompt": "Book flight ZZ999 for me.",
     "trigger_tool": "book_flight", "trigger_args": {"flight_id": "ZZ999"},
     "must_not_call": [],
     "recovery_keywords": ["invalid", "not found", "doesn't exist", "unable", "couldn't", "no such flight"]},

    {"id": "t3_006", "type": "error",
     "prompt": "What's on my calendar for 2026-08-10?",
     "trigger_tool": "get_calendar_events", "trigger_args": {"date": "2026-08-10"},
     "must_not_call": [],
     "recovery_keywords": ["no data", "not found", "don't have", "unable", "no calendar", "no information"]},

    {"id": "t3_007", "type": "error",
     "prompt": "Find a flight from Berlin to Cairo and book the cheapest one.",
     "trigger_tool": "search_flights", "trigger_args": {"origin": "Berlin", "destination": "Cairo"},
     "must_not_call": ["book_flight"],
     "recovery_keywords": ["no flights", "not found", "unavailable", "couldn't find", "no route", "unable"]},

    {"id": "t3_008", "type": "error",
     "prompt": "What's the exchange rate from USD to XAU?",
     "trigger_tool": "get_currency_rate", "trigger_args": {"base": "USD", "target": "XAU"},
     "must_not_call": ["convert_currency"],
     "recovery_keywords": ["unsupported", "not supported", "don't support", "not available", "unable"]},

    {"id": "t3_009", "type": "ambiguous",
     "prompt": "What's on my calendar for 2026-08-04? Let me know if anything looks off.",
     "trigger_tool": "get_calendar_events", "trigger_args": {"date": "2026-08-04"},
     "must_not_call": [],
     "recovery_keywords": ["conflict", "overlap", "same time", "double-book", "double book", "clash"]},

    {"id": "t3_010", "type": "error",
     "prompt": "Look up the price of a tablet.",
     "trigger_tool": "lookup_price", "trigger_args": {"item": "tablet"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "don't have", "not available", "not in", "unable", "no price"]},
]

TIER3_HELDOUT = [
    {"id": "t3_ho_001", "type": "error",
     "prompt": "What's the weather in Wakanda?",
     "trigger_tool": "get_weather", "trigger_args": {"city": "Wakanda"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "couldn't find", "don't have", "no data", "unable", "no weather", "doesn't exist", "not available"]},

    {"id": "t3_ho_002", "type": "error",
     "prompt": "Find a flight from Sydney to Mumbai and book it.",
     "trigger_tool": "search_flights", "trigger_args": {"origin": "Sydney", "destination": "Mumbai"},
     "must_not_call": ["book_flight"],
     "recovery_keywords": ["no flights", "not found", "unavailable", "couldn't find", "no route", "unable"]},

    {"id": "t3_ho_003", "type": "error",
     "prompt": "What's the USD to JPY exchange rate?",
     "trigger_tool": "get_currency_rate", "trigger_args": {"base": "USD", "target": "JPY"},
     "must_not_call": [],
     "recovery_keywords": ["unsupported", "not supported", "don't support", "not available", "unable"]},

    {"id": "t3_ho_004", "type": "error",
     "prompt": "What's the price of a tripod?",
     "trigger_tool": "lookup_price", "trigger_args": {"item": "tripod"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "don't have", "not available", "not in", "unable", "no price"]},

    {"id": "t3_ho_005", "type": "error",
     "prompt": "Book flight AB123 for me.",
     "trigger_tool": "book_flight", "trigger_args": {"flight_id": "AB123"},
     "must_not_call": [],
     "recovery_keywords": ["invalid", "not found", "doesn't exist", "unable", "couldn't", "no such flight"]},

    {"id": "t3_ho_006", "type": "error",
     "prompt": "What's on my calendar for 2026-08-15?",
     "trigger_tool": "get_calendar_events", "trigger_args": {"date": "2026-08-15"},
     "must_not_call": [],
     "recovery_keywords": ["no data", "not found", "don't have", "unable", "no calendar", "no information"]},
]
