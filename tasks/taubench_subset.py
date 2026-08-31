"""
tau-bench-inspired task suite. See envs/tools_taubench.py for the scope
note: these are hand-ported task patterns using real tau-bench tool
signatures, not the official benchmark run.

Two task types, matching the same Tier 2 / Tier 3 grading logic already
used throughout this project:
  - CHAIN tasks: two independent, unordered tool calls (same pattern as
    the original Tier 2 weather-compare / price-pair tasks). Deliberately
    avoids any cross-call value dependency (e.g. "book the flight found by
    search"), since that would need placeholder-resolution logic specific
    to envs/tools.py's search_flights function that doesn't generalize to
    this module's different tool names/return shapes.
  - ERROR tasks: one tool call that returns a genuine error from the
    simulator; success means the model reports the problem rather than
    proceeding as if nothing were wrong. Graded identically to the
    project's existing Tier 3 tasks.
"""

TAUBENCH_CHAIN_TASKS = [
    {"id": "tb_c_001",
     "prompt": "Check the reservation details for ABC123 and also look up the price of item50.",
     "expected_sequence": [
         {"tool": "get_reservation_details", "args": {"reservation_id": "ABC123"}},
         {"tool": "get_product_details", "args": {"item_id": "item50"}},
     ],
     "order_sensitive": False},

    {"id": "tb_c_002",
     "prompt": "Get the details of order ORD1001 and also check reservation DEF456.",
     "expected_sequence": [
         {"tool": "get_order_details", "args": {"order_id": "ORD1001"}},
         {"tool": "get_reservation_details", "args": {"reservation_id": "DEF456"}},
     ],
     "order_sensitive": False},

    {"id": "tb_c_003",
     "prompt": "Search flights from JFK to LAX and also check the details of order ORD1003.",
     "expected_sequence": [
         {"tool": "search_direct_flight", "args": {"origin": "JFK", "destination": "LAX"}},
         {"tool": "get_order_details", "args": {"order_id": "ORD1003"}},
     ],
     "order_sensitive": False},

    {"id": "tb_c_004",
     "prompt": "Look up product item77 and check reservation GHI789.",
     "expected_sequence": [
         {"tool": "get_product_details", "args": {"item_id": "item77"}},
         {"tool": "get_reservation_details", "args": {"reservation_id": "GHI789"}},
     ],
     "order_sensitive": False},
]

TAUBENCH_CHAIN_HELDOUT = [
    {"id": "tb_c_ho_001",
     "prompt": "Check reservation ABC123 and look up product item12.",
     "expected_sequence": [
         {"tool": "get_reservation_details", "args": {"reservation_id": "ABC123"}},
         {"tool": "get_product_details", "args": {"item_id": "item12"}},
     ],
     "order_sensitive": False},

    {"id": "tb_c_ho_002",
     "prompt": "Get order ORD1002 details and search flights from LAX to ORD.",
     "expected_sequence": [
         {"tool": "get_order_details", "args": {"order_id": "ORD1002"}},
         {"tool": "search_direct_flight", "args": {"origin": "LAX", "destination": "ORD"}},
     ],
     "order_sensitive": False},

    {"id": "tb_c_ho_003",
     "prompt": "Search flights from ORD to JFK and check reservation DEF456.",
     "expected_sequence": [
         {"tool": "search_direct_flight", "args": {"origin": "ORD", "destination": "JFK"}},
         {"tool": "get_reservation_details", "args": {"reservation_id": "DEF456"}},
     ],
     "order_sensitive": False},
]

TAUBENCH_ERROR_TASKS = [
    {"id": "tb_e_001", "type": "error",
     "prompt": "Get the details of reservation XYZ999.",
     "trigger_tool": "get_reservation_details", "trigger_args": {"reservation_id": "XYZ999"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "couldn't find", "doesn't exist", "unable", "no reservation", "no record", "not a valid"]},

    {"id": "tb_e_002", "type": "error",
     "prompt": "Search for direct flights from MIA to SEA.",
     "trigger_tool": "search_direct_flight", "trigger_args": {"origin": "MIA", "destination": "SEA"},
     "must_not_call": ["book_reservation"],
     "recovery_keywords": ["no direct flights", "not found", "no route", "unable", "no flights", "does not"]},

    {"id": "tb_e_003", "type": "error",
     "prompt": "Exchange the item in order ORD1002 for item88.",
     "trigger_tool": "exchange_delivered_order_items", "trigger_args": {"order_id": "ORD1002", "new_item_id": "item88"},
     "must_not_call": [],
     "recovery_keywords": ["not eligible", "cannot exchange", "not delivered", "unable", "still pending", "not able"]},

    {"id": "tb_e_004", "type": "error",
     "prompt": "Get the details of order ORD9999.",
     "trigger_tool": "get_order_details", "trigger_args": {"order_id": "ORD9999"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "couldn't find", "doesn't exist", "unable", "no order", "no record"]},

    {"id": "tb_e_005", "type": "error",
     "prompt": "Book a flight for user004 with flight number HAT999.",
     "trigger_tool": "book_reservation", "trigger_args": {"user_id": "user004", "flight_number": "HAT999"},
     "must_not_call": [],
     "recovery_keywords": ["invalid", "not found", "doesn't exist", "unable", "no such flight", "not a valid"]},
]

TAUBENCH_ERROR_HELDOUT = [
    {"id": "tb_e_ho_001", "type": "error",
     "prompt": "Cancel reservation ZZZ000.",
     "trigger_tool": "cancel_reservation", "trigger_args": {"reservation_id": "ZZZ000"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "couldn't find", "doesn't exist", "unable", "no reservation", "no record"]},

    {"id": "tb_e_ho_002", "type": "error",
     "prompt": "Look up product item99.",
     "trigger_tool": "get_product_details", "trigger_args": {"item_id": "item99"},
     "must_not_call": [],
     "recovery_keywords": ["not found", "don't have", "not available", "unable", "no product", "no record"]},
]
