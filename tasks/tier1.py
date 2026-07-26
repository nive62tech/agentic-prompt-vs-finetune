"""
Tier 1 — Single tool call.

Each task is a natural-language request resolvable with exactly ONE
correctly-formed tool call. Success = correct tool name + correct arguments.

`expected_tool`: the tool name that should be called.
`expected_args`: dict of argument values that should appear in the call
                 (grader does case-insensitive string comparison on values).
"""

TIER1_TASKS = [
    {"id": "t1_001", "prompt": "What's the weather like in Tokyo right now?",
     "expected_tool": "get_weather", "expected_args": {"city": "Tokyo"}},

    {"id": "t1_002", "prompt": "Is it raining in London?",
     "expected_tool": "get_weather", "expected_args": {"city": "London"}},

    {"id": "t1_003", "prompt": "Tell me the temperature in New York.",
     "expected_tool": "get_weather", "expected_args": {"city": "New York"}},

    {"id": "t1_004", "prompt": "Check the weather for Mumbai.",
     "expected_tool": "get_weather", "expected_args": {"city": "Mumbai"}},

    {"id": "t1_005", "prompt": "How hot is it in Dubai today?",
     "expected_tool": "get_weather", "expected_args": {"city": "Dubai"}},

    {"id": "t1_006", "prompt": "What's the price of a laptop?",
     "expected_tool": "lookup_price", "expected_args": {"item": "laptop"}},

    {"id": "t1_007", "prompt": "How much do headphones cost?",
     "expected_tool": "lookup_price", "expected_args": {"item": "headphones"}},

    {"id": "t1_008", "prompt": "What's the price on a monitor?",
     "expected_tool": "lookup_price", "expected_args": {"item": "monitor"}},

    {"id": "t1_009", "prompt": "Check the price of a keyboard for me.",
     "expected_tool": "lookup_price", "expected_args": {"item": "keyboard"}},

    {"id": "t1_010", "prompt": "What's my exchange rate from USD to EUR?",
     "expected_tool": "get_currency_rate", "expected_args": {"base": "USD", "target": "EUR"}},

    {"id": "t1_011", "prompt": "What is the GBP to USD exchange rate?",
     "expected_tool": "get_currency_rate", "expected_args": {"base": "GBP", "target": "USD"}},

    {"id": "t1_012", "prompt": "Convert 100 USD to EUR.",
     "expected_tool": "convert_currency", "expected_args": {"amount": 100, "base": "USD", "target": "EUR"}},

    {"id": "t1_013", "prompt": "How much is 50 USD in INR?",
     "expected_tool": "convert_currency", "expected_args": {"amount": 50, "base": "USD", "target": "INR"}},

    {"id": "t1_014", "prompt": "What's on my calendar for 2026-08-01?",
     "expected_tool": "get_calendar_events", "expected_args": {"date": "2026-08-01"}},

    {"id": "t1_015", "prompt": "Do I have anything scheduled on 2026-08-03?",
     "expected_tool": "get_calendar_events", "expected_args": {"date": "2026-08-03"}},

    {"id": "t1_016", "prompt": "Search for flights from Tokyo to London.",
     "expected_tool": "search_flights", "expected_args": {"origin": "Tokyo", "destination": "London"}},
]

# Held-out generalization set — same structure, different entities/values.
# Used ONLY for evaluation, never for DSPy optimization or fine-tuning.
TIER1_HELDOUT = [
    {"id": "t1_ho_001", "prompt": "What's the current weather in Paris?",
     "expected_tool": "get_weather", "expected_args": {"city": "Paris"}},

    {"id": "t1_ho_002", "prompt": "Is Berlin cold right now?",
     "expected_tool": "get_weather", "expected_args": {"city": "Berlin"}},

    {"id": "t1_ho_003", "prompt": "Weather in Sydney please.",
     "expected_tool": "get_weather", "expected_args": {"city": "Sydney"}},

    {"id": "t1_ho_004", "prompt": "What's the price of a mouse?",
     "expected_tool": "lookup_price", "expected_args": {"item": "mouse"}},

    {"id": "t1_ho_005", "prompt": "What's the INR to USD rate?",
     "expected_tool": "get_currency_rate", "expected_args": {"base": "INR", "target": "USD"}},

    {"id": "t1_ho_006", "prompt": "Convert 200 GBP to USD.",
     "expected_tool": "convert_currency", "expected_args": {"amount": 200, "base": "GBP", "target": "USD"}},

    {"id": "t1_ho_007", "prompt": "What do I have going on 2026-08-04?",
     "expected_tool": "get_calendar_events", "expected_args": {"date": "2026-08-04"}},

    {"id": "t1_ho_008", "prompt": "Find flights from Mumbai to Dubai.",
     "expected_tool": "search_flights", "expected_args": {"origin": "Mumbai", "destination": "Dubai"}},
]
