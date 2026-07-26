"""
Tier 2 — Multi-tool chains (2-4 sequential tool calls, later calls may
depend on earlier outputs).

`expected_sequence`: ordered list of {"tool": ..., "args": {...}} the model
                      SHOULD call. For calls whose arguments depend on a
                      prior tool's output (e.g. booking a flight_id returned
                      by search_flights), args uses "<from:tool_name>" as a
                      placeholder the grader resolves dynamically against
                      the actual simulator output.
`order_sensitive`: if False, the two+ calls can happen in either order
                    (e.g. checking weather in two independent cities).
"""

TIER2_TASKS = [
    {"id": "t2_001",
     "prompt": "Find a flight from Tokyo to London and book it.",
     "expected_sequence": [
         {"tool": "search_flights", "args": {"origin": "Tokyo", "destination": "London"}},
         {"tool": "book_flight", "args": {"flight_id": "<from:search_flights>"}},
     ],
     "order_sensitive": True},

    {"id": "t2_002",
     "prompt": "Book the cheapest flight from Mumbai to Dubai.",
     "expected_sequence": [
         {"tool": "search_flights", "args": {"origin": "Mumbai", "destination": "Dubai"}},
         {"tool": "book_flight", "args": {"flight_id": "<from:search_flights>"}},
     ],
     "order_sensitive": True},

    {"id": "t2_003",
     "prompt": "What's the weather in Paris and New York? Tell me which is warmer.",
     "expected_sequence": [
         {"tool": "get_weather", "args": {"city": "Paris"}},
         {"tool": "get_weather", "args": {"city": "New York"}},
     ],
     "order_sensitive": False},

    {"id": "t2_004",
     "prompt": "Convert 100 USD to EUR, then convert that EUR amount to GBP.",
     "expected_sequence": [
         {"tool": "convert_currency", "args": {"amount": 100, "base": "USD", "target": "EUR"}},
         {"tool": "convert_currency", "args": {"amount": "<from:convert_currency:converted>", "base": "EUR", "target": "GBP"}},
     ],
     "order_sensitive": True},

    {"id": "t2_005",
     "prompt": "Check my calendar for 2026-08-01 and also look up the price of a laptop.",
     "expected_sequence": [
         {"tool": "get_calendar_events", "args": {"date": "2026-08-01"}},
         {"tool": "lookup_price", "args": {"item": "laptop"}},
     ],
     "order_sensitive": False},

    {"id": "t2_006",
     "prompt": "What's the USD to INR exchange rate, and how much is 200 USD in INR?",
     "expected_sequence": [
         {"tool": "get_currency_rate", "args": {"base": "USD", "target": "INR"}},
         {"tool": "convert_currency", "args": {"amount": 200, "base": "USD", "target": "INR"}},
     ],
     "order_sensitive": False},

    {"id": "t2_007",
     "prompt": "Find a flight from New York to Paris and book it.",
     "expected_sequence": [
         {"tool": "search_flights", "args": {"origin": "New York", "destination": "Paris"}},
         {"tool": "book_flight", "args": {"flight_id": "<from:search_flights>"}},
     ],
     "order_sensitive": True},

    {"id": "t2_008",
     "prompt": "Compare the weather in Dubai and Singapore.",
     "expected_sequence": [
         {"tool": "get_weather", "args": {"city": "Dubai"}},
         {"tool": "get_weather", "args": {"city": "Singapore"}},
     ],
     "order_sensitive": False},

    {"id": "t2_009",
     "prompt": "Look up the price of a keyboard and a mouse.",
     "expected_sequence": [
         {"tool": "lookup_price", "args": {"item": "keyboard"}},
         {"tool": "lookup_price", "args": {"item": "mouse"}},
     ],
     "order_sensitive": False},

    {"id": "t2_010",
     "prompt": "Find a flight from Dubai to Singapore and book it.",
     "expected_sequence": [
         {"tool": "search_flights", "args": {"origin": "Dubai", "destination": "Singapore"}},
         {"tool": "book_flight", "args": {"flight_id": "<from:search_flights>"}},
     ],
     "order_sensitive": True},

    {"id": "t2_011",
     "prompt": "Check the weather in Tokyo and Berlin, and tell me my calendar for 2026-08-01.",
     "expected_sequence": [
         {"tool": "get_weather", "args": {"city": "Tokyo"}},
         {"tool": "get_weather", "args": {"city": "Berlin"}},
         {"tool": "get_calendar_events", "args": {"date": "2026-08-01"}},
     ],
     "order_sensitive": False},

    {"id": "t2_012",
     "prompt": "What's the GBP to USD rate, then convert 300 GBP to USD.",
     "expected_sequence": [
         {"tool": "get_currency_rate", "args": {"base": "GBP", "target": "USD"}},
         {"tool": "convert_currency", "args": {"amount": 300, "base": "GBP", "target": "USD"}},
     ],
     "order_sensitive": False},
]

TIER2_HELDOUT = [
    {"id": "t2_ho_001",
     "prompt": "Find a flight from London to Tokyo and book it.",
     "expected_sequence": [
         {"tool": "search_flights", "args": {"origin": "London", "destination": "Tokyo"}},
         {"tool": "book_flight", "args": {"flight_id": "<from:search_flights>"}},
     ],
     "order_sensitive": True},

    {"id": "t2_ho_002",
     "prompt": "What's the weather in Cairo and Sydney? Which is nicer?",
     "expected_sequence": [
         {"tool": "get_weather", "args": {"city": "Cairo"}},
         {"tool": "get_weather", "args": {"city": "Sydney"}},
     ],
     "order_sensitive": False},

    {"id": "t2_ho_003",
     "prompt": "Convert 500 EUR to USD, then convert that USD amount to INR.",
     "expected_sequence": [
         {"tool": "convert_currency", "args": {"amount": 500, "base": "EUR", "target": "USD"}},
         {"tool": "convert_currency", "args": {"amount": "<from:convert_currency:converted>", "base": "USD", "target": "INR"}},
     ],
     "order_sensitive": True},

    {"id": "t2_ho_004",
     "prompt": "Check my calendar for 2026-08-03 and look up the price of a monitor.",
     "expected_sequence": [
         {"tool": "get_calendar_events", "args": {"date": "2026-08-03"}},
         {"tool": "lookup_price", "args": {"item": "monitor"}},
     ],
     "order_sensitive": False},

    {"id": "t2_ho_005",
     "prompt": "Find a flight from Paris to New York and book it.",
     "expected_sequence": [
         {"tool": "search_flights", "args": {"origin": "Paris", "destination": "New York"}},
         {"tool": "book_flight", "args": {"flight_id": "<from:search_flights>"}},
     ],
     "order_sensitive": True},

    {"id": "t2_ho_006",
     "prompt": "What's the USD to GBP rate, and what's the weather in London?",
     "expected_sequence": [
         {"tool": "get_currency_rate", "args": {"base": "USD", "target": "GBP"}},
         {"tool": "get_weather", "args": {"city": "London"}},
     ],
     "order_sensitive": False},
]
