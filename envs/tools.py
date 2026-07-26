"""
Deterministic tool simulators for the agentic tool-use task suite.

Every tool here is a pure Python function with fixed, hard-coded data —
no network calls, no randomness. This means every experiment run is free,
reproducible, and exactly gradable, and lets us deliberately inject errors
for Tier 3 tasks (see WEATHER_DATA / FLIGHT_DATA "poison" entries below).

TOOL_SCHEMAS is the JSON-schema-style tool list to hand to the model
(e.g. via tokenizer.apply_chat_template(..., tools=TOOL_SCHEMAS, ...)).
"""

# ---------------------------------------------------------------------------
# Fixed data
# ---------------------------------------------------------------------------

WEATHER_DATA = {
    "tokyo":     {"temp_c": 27, "condition": "clear"},
    "london":    {"temp_c": 16, "condition": "rainy"},
    "new york":  {"temp_c": 22, "condition": "cloudy"},
    "mumbai":    {"temp_c": 31, "condition": "humid"},
    "dubai":     {"temp_c": 40, "condition": "clear"},
    "paris":     {"temp_c": 19, "condition": "cloudy"},
    "berlin":    {"temp_c": 14, "condition": "rainy"},
    "sydney":    {"temp_c": 18, "condition": "clear"},
    "singapore": {"temp_c": 30, "condition": "humid"},
    "cairo":     {"temp_c": 35, "condition": "clear"},
    # deliberately NOT included: "atlantis" -> used to trigger a
    # "city not found" error for Tier 3 error-recovery tasks.
}

CURRENCY_RATES = {
    # (base, target) -> rate. Only a small fixed set is supported.
    ("usd", "eur"): 0.92,
    ("usd", "gbp"): 0.79,
    ("usd", "inr"): 83.1,
    ("eur", "usd"): 1.09,
    ("gbp", "usd"): 1.27,
    ("inr", "usd"): 0.012,
    # deliberately unsupported pair, e.g. ("usd", "xau") -> used to
    # trigger an "unsupported currency pair" error for Tier 3 tasks.
}

FLIGHT_DATA = {
    ("tokyo", "london"):    [{"flight_id": "TL101", "price_usd": 950, "duration_hr": 14}],
    ("london", "tokyo"):    [{"flight_id": "LT202", "price_usd": 980, "duration_hr": 14}],
    ("new york", "paris"):  [{"flight_id": "NP303", "price_usd": 620, "duration_hr": 8}],
    ("paris", "new york"):  [{"flight_id": "PN404", "price_usd": 640, "duration_hr": 9}],
    ("dubai", "singapore"): [{"flight_id": "DS505", "price_usd": 410, "duration_hr": 7}],
    ("mumbai", "dubai"):    [{"flight_id": "MD606", "price_usd": 210, "duration_hr": 3}],
    # deliberately missing route, e.g. ("cairo", "sydney") -> used to
    # trigger a "no flights found" error for Tier 3 tasks.
}

PRICE_CATALOG = {
    "laptop": 899.0,
    "headphones": 149.0,
    "monitor": 259.0,
    "keyboard": 59.0,
    "mouse": 29.0,
    # deliberately missing item, e.g. "smartwatch" -> triggers
    # "item not found" error for Tier 3 tasks.
}

CALENDAR_DATA = {
    "2026-08-01": ["Team sync 10am", "Dentist 2pm"],
    "2026-08-02": [],
    "2026-08-03": ["Flight to Paris 6am"],
    "2026-08-04": ["Client call 11am", "Board meeting 11am"],  # deliberate scheduling conflict, same time
}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    key = city.strip().lower()
    if key not in WEATHER_DATA:
        return {"error": f"city not found: {city}"}
    return {"city": city, **WEATHER_DATA[key]}


def get_currency_rate(base: str, target: str) -> dict:
    key = (base.strip().lower(), target.strip().lower())
    if key not in CURRENCY_RATES:
        return {"error": f"unsupported currency pair: {base}->{target}"}
    return {"base": base, "target": target, "rate": CURRENCY_RATES[key]}


def convert_currency(amount: float, base: str, target: str) -> dict:
    rate_result = get_currency_rate(base, target)
    if "error" in rate_result:
        return rate_result
    return {"amount": amount, "base": base, "target": target,
            "converted": round(amount * rate_result["rate"], 2)}


def search_flights(origin: str, destination: str) -> dict:
    key = (origin.strip().lower(), destination.strip().lower())
    flights = FLIGHT_DATA.get(key, [])
    if not flights:
        return {"error": f"no flights found: {origin}->{destination}"}
    return {"origin": origin, "destination": destination, "flights": flights}


def book_flight(flight_id: str) -> dict:
    all_ids = {f["flight_id"] for flights in FLIGHT_DATA.values() for f in flights}
    if flight_id not in all_ids:
        return {"error": f"invalid flight_id: {flight_id}"}
    return {"flight_id": flight_id, "status": "booked"}


def lookup_price(item: str) -> dict:
    key = item.strip().lower()
    if key not in PRICE_CATALOG:
        return {"error": f"item not found: {item}"}
    return {"item": item, "price_usd": PRICE_CATALOG[key]}


def get_calendar_events(date: str) -> dict:
    if date not in CALENDAR_DATA:
        return {"error": f"no calendar data for date: {date}"}
    return {"date": date, "events": CALENDAR_DATA[date]}


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "get_currency_rate": get_currency_rate,
    "convert_currency": convert_currency,
    "search_flights": search_flights,
    "book_flight": book_flight,
    "lookup_price": lookup_price,
    "get_calendar_events": get_calendar_events,
}


# ---------------------------------------------------------------------------
# Tool schemas (pass this list to the model as its available tools)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_currency_rate",
        "description": "Get the exchange rate between two currencies (3-letter codes).",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["base", "target"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "base": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["amount", "base", "target"],
        },
    },
    {
        "name": "search_flights",
        "description": "Search flights between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a flight by its flight_id.",
        "parameters": {
            "type": "object",
            "properties": {"flight_id": {"type": "string"}},
            "required": ["flight_id"],
        },
    },
    {
        "name": "lookup_price",
        "description": "Look up the price of an item in the catalog.",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
        },
    },
    {
        "name": "get_calendar_events",
        "description": "Get calendar events for a given date (YYYY-MM-DD).",
        "parameters": {
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
        },
    },
]


def call_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call by name. Used by the grader / experiment harness."""
    if name not in TOOL_REGISTRY:
        return {"error": f"unknown tool: {name}"}
    try:
        return TOOL_REGISTRY[name](**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
