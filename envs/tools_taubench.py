"""
tau-bench-INSPIRED deterministic tool simulators.

IMPORTANT SCOPE NOTE: this is NOT the official tau-bench harness. Running
the real tau-bench requires its own package, a separate user-simulator LLM
playing the customer's side of a dual-control conversation, policy-document
compliance checking, and database-state-diff scoring — a fundamentally
different and much heavier evaluation setup than the rest of this project.

Instead, this module hand-ports a small set of tau-bench's real airline and
retail domain tool signatures (names, arguments, general behavior — see
https://github.com/sierra-research/tau-bench) into the same deterministic-
simulator style used throughout this project (envs/tools.py), so the
existing task-suite pattern can produce a genuine, if modest, external-
domain generalization check. Every result using this module should be
reported as "tau-bench-inspired subset", never as "tau-bench results".
"""

# ---------------------------------------------------------------------------
# Airline domain (fixed data)
# ---------------------------------------------------------------------------

FLIGHTS = {
    ("jfk", "lax"): [{"flight_number": "HAT001", "price": 250}],
    ("lax", "ord"): [{"flight_number": "HAT002", "price": 180}],
    ("ord", "jfk"): [{"flight_number": "HAT003", "price": 210}],
    ("jfk", "mia"): [{"flight_number": "HAT004", "price": 150}],
    # deliberately missing route, e.g. ("mia","sea") -> triggers
    # "no direct flights" error for error-recovery tasks.
}

RESERVATIONS = {
    "ABC123": {"user_id": "user001", "flight_number": "HAT001", "status": "confirmed"},
    "DEF456": {"user_id": "user002", "flight_number": "HAT002", "status": "confirmed"},
    "GHI789": {"user_id": "user003", "flight_number": "HAT004", "status": "confirmed"},
    # deliberately missing IDs, e.g. "XYZ999" -> triggers "not found" error.
}

# ---------------------------------------------------------------------------
# Retail domain (fixed data)
# ---------------------------------------------------------------------------

ORDERS = {
    "ORD1001": {"user_id": "user001", "item_id": "item50", "status": "delivered"},
    "ORD1002": {"user_id": "user002", "item_id": "item77", "status": "pending"},
    "ORD1003": {"user_id": "user003", "item_id": "item12", "status": "delivered"},
    # deliberately missing IDs, e.g. "ORD9999" -> triggers "not found" error.
}

PRODUCTS = {
    "item50": {"name": "Wireless Mouse", "price": 25},
    "item77": {"name": "Keyboard", "price": 60},
    "item12": {"name": "USB-C Cable", "price": 10},
    "item88": {"name": "Webcam", "price": 45},
    # deliberately missing IDs, e.g. "item99" -> triggers "not found" error.
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_direct_flight(origin: str, destination: str) -> dict:
    key = (origin.strip().lower(), destination.strip().lower())
    flights = FLIGHTS.get(key, [])
    if not flights:
        return {"error": f"no direct flights found: {origin}->{destination}"}
    return {"origin": origin, "destination": destination, "flights": flights}


def get_reservation_details(reservation_id: str) -> dict:
    r = RESERVATIONS.get(reservation_id)
    if r is None:
        return {"error": f"reservation not found: {reservation_id}"}
    return {"reservation_id": reservation_id, **r}


def book_reservation(user_id: str, flight_number: str) -> dict:
    valid_flights = {f["flight_number"] for flights in FLIGHTS.values() for f in flights}
    if flight_number not in valid_flights:
        return {"error": f"invalid flight_number: {flight_number}"}
    return {"user_id": user_id, "flight_number": flight_number, "status": "confirmed"}


def cancel_reservation(reservation_id: str) -> dict:
    if reservation_id not in RESERVATIONS:
        return {"error": f"reservation not found: {reservation_id}"}
    return {"reservation_id": reservation_id, "status": "cancelled"}


def get_order_details(order_id: str) -> dict:
    o = ORDERS.get(order_id)
    if o is None:
        return {"error": f"order not found: {order_id}"}
    return {"order_id": order_id, **o}


def get_product_details(item_id: str) -> dict:
    p = PRODUCTS.get(item_id)
    if p is None:
        return {"error": f"product not found: {item_id}"}
    return {"item_id": item_id, **p}


def exchange_delivered_order_items(order_id: str, new_item_id: str) -> dict:
    o = ORDERS.get(order_id)
    if o is None:
        return {"error": f"order not found: {order_id}"}
    if o["status"] != "delivered":
        return {"error": f"order not eligible for exchange, status={o['status']}"}
    if new_item_id not in PRODUCTS:
        return {"error": f"product not found: {new_item_id}"}
    return {"order_id": order_id, "new_item_id": new_item_id, "status": "exchanged"}


TOOL_REGISTRY = {
    "search_direct_flight": search_direct_flight,
    "get_reservation_details": get_reservation_details,
    "book_reservation": book_reservation,
    "cancel_reservation": cancel_reservation,
    "get_order_details": get_order_details,
    "get_product_details": get_product_details,
    "exchange_delivered_order_items": exchange_delivered_order_items,
}

TOOL_SCHEMAS = [
    {"name": "search_direct_flight", "description": "Search direct flights between two airports.",
     "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}}, "required": ["origin", "destination"]}},
    {"name": "get_reservation_details", "description": "Get details of a flight reservation by ID.",
     "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}}, "required": ["reservation_id"]}},
    {"name": "book_reservation", "description": "Book a flight reservation for a user.",
     "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}, "flight_number": {"type": "string"}}, "required": ["user_id", "flight_number"]}},
    {"name": "cancel_reservation", "description": "Cancel a flight reservation by ID.",
     "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}}, "required": ["reservation_id"]}},
    {"name": "get_order_details", "description": "Get details of a retail order by ID.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "get_product_details", "description": "Get details of a product by item ID.",
     "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}}, "required": ["item_id"]}},
    {"name": "exchange_delivered_order_items", "description": "Exchange an item in a delivered order for a different item.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "new_item_id": {"type": "string"}}, "required": ["order_id", "new_item_id"]}},
]


def call_tool(name: str, args: dict) -> dict:
    if name not in TOOL_REGISTRY:
        return {"error": f"unknown tool: {name}"}
    try:
        return TOOL_REGISTRY[name](**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
