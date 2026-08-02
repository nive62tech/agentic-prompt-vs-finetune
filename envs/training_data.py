"""
Training-data generator for DSPy optimization and QLoRA fine-tuning.

Why this exists: tasks/tier1.py has only 16 tasks (all reserved for
evaluation) + 8 held-out. That's not enough to draw N=10/50 TRAINING
examples from without reusing eval data — which would invalidate the
whole comparison. This module generates a much larger pool of Tier 1
examples from the same simulators/templates, then samples N of them
for training while guaranteeing zero overlap with tasks/tier1.py's
eval and held-out prompts.

Only Tier 1 is covered here. Tier 2/3 training data can follow the
same pattern later (template x entity combos) if you extend the
comparison to those tiers — flagged as a known scope limit for now,
not an oversight.
"""

import random
import itertools
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.tools import WEATHER_DATA, PRICE_CATALOG, CURRENCY_RATES, CALENDAR_DATA, FLIGHT_DATA
from tasks.tier1 import TIER1_TASKS, TIER1_HELDOUT
from tasks.tier2 import TIER2_TASKS, TIER2_HELDOUT
from tasks.tier3 import TIER3_TASKS, TIER3_HELDOUT


def _title(city_key: str) -> str:
    return " ".join(w.capitalize() for w in city_key.split())


WEATHER_TEMPLATES = [
    "What's the weather in {city}?",
    "Is it raining in {city}?",
    "How hot is it in {city} today?",
    "Check the weather for {city}.",
    "Tell me the temperature in {city}.",
]

PRICE_TEMPLATES = [
    "What's the price of a {item}?",
    "How much does a {item} cost?",
    "Check the price on a {item}.",
    "Look up the {item} price for me.",
]

RATE_TEMPLATES = [
    "What's the exchange rate from {base} to {target}?",
    "What is the {base} to {target} rate?",
]

CONVERT_TEMPLATES = [
    "Convert {amount} {base} to {target}.",
    "How much is {amount} {base} in {target}?",
]

CALENDAR_TEMPLATES = [
    "What's on my calendar for {date}?",
    "Do I have anything scheduled on {date}?",
]

FLIGHT_TEMPLATES = [
    "Search for flights from {origin} to {destination}.",
    "Find flights from {origin} to {destination}.",
]

AMOUNTS = [50, 100, 250]


def generate_tier1_pool() -> list:
    pool = []

    for city_key in WEATHER_DATA:
        city = _title(city_key)
        for tmpl in WEATHER_TEMPLATES:
            pool.append({
                "prompt": tmpl.format(city=city),
                "expected_tool": "get_weather",
                "expected_args": {"city": city},
            })

    for item in PRICE_CATALOG:
        for tmpl in PRICE_TEMPLATES:
            pool.append({
                "prompt": tmpl.format(item=item),
                "expected_tool": "lookup_price",
                "expected_args": {"item": item},
            })

    for (base, target) in CURRENCY_RATES:
        for tmpl in RATE_TEMPLATES:
            pool.append({
                "prompt": tmpl.format(base=base.upper(), target=target.upper()),
                "expected_tool": "get_currency_rate",
                "expected_args": {"base": base.upper(), "target": target.upper()},
            })
        for tmpl in CONVERT_TEMPLATES:
            for amount in AMOUNTS:
                pool.append({
                    "prompt": tmpl.format(amount=amount, base=base.upper(), target=target.upper()),
                    "expected_tool": "convert_currency",
                    "expected_args": {"amount": amount, "base": base.upper(), "target": target.upper()},
                })

    for date in CALENDAR_DATA:
        for tmpl in CALENDAR_TEMPLATES:
            pool.append({
                "prompt": tmpl.format(date=date),
                "expected_tool": "get_calendar_events",
                "expected_args": {"date": date},
            })

    for (origin_key, dest_key) in FLIGHT_DATA:
        origin, dest = _title(origin_key), _title(dest_key)
        for tmpl in FLIGHT_TEMPLATES:
            pool.append({
                "prompt": tmpl.format(origin=origin, destination=dest),
                "expected_tool": "search_flights",
                "expected_args": {"origin": origin, "destination": dest},
            })

    return pool


def sample_tier1_training(n: int, seed: int = 42) -> list:
    """
    Samples n training examples, guaranteed to have zero overlap with
    tasks/tier1.py's TIER1_TASKS (eval) or TIER1_HELDOUT (generalization).
    """
    reserved_prompts = {t["prompt"] for t in TIER1_TASKS} | {t["prompt"] for t in TIER1_HELDOUT}
    pool = [ex for ex in generate_tier1_pool() if ex["prompt"] not in reserved_prompts]

    # dedupe the pool itself (some template/entity combos can coincide)
    seen = set()
    deduped = []
    for ex in pool:
        if ex["prompt"] not in seen:
            seen.add(ex["prompt"])
            deduped.append(ex)

    if n > len(deduped):
        raise ValueError(f"Requested {n} training examples but only {len(deduped)} available in the pool.")

    rng = random.Random(seed)
    return rng.sample(deduped, n)


def generate_tier2_pool() -> list:
    """
    Tier 2 training pool, built ONLY from patterns that have enough
    non-overlapping combinations left after excluding tasks/tier2.py's
    eval + held-out prompts.

    NOTE: the flight search-then-book pattern is deliberately excluded.
    FLIGHT_DATA only has 6 routes, and TIER2_TASKS + TIER2_HELDOUT
    together already use all 6 — there is no non-overlapping data left
    to generate flight-chain training examples from without leaking
    eval data. This is a real constraint of the small, free, deterministic
    simulated world, not an oversight.
    """
    pool = []
    cities = [_title(c) for c in WEATHER_DATA]
    items = list(PRICE_CATALOG)
    dates = list(CALENDAR_DATA)

    # Pattern A: weather-compare (2 unordered get_weather calls)
    for c1, c2 in itertools.combinations(cities, 2):
        pool.append({
            "prompt": f"What's the weather in {c1} and {c2}? Which is nicer?",
            "expected_sequence": [
                {"tool": "get_weather", "args": {"city": c1}},
                {"tool": "get_weather", "args": {"city": c2}},
            ],
            "order_sensitive": False,
        })

    # Pattern B: price-pair (2 unordered lookup_price calls)
    for i1, i2 in itertools.combinations(items, 2):
        pool.append({
            "prompt": f"Look up the price of a {i1} and a {i2}.",
            "expected_sequence": [
                {"tool": "lookup_price", "args": {"item": i1}},
                {"tool": "lookup_price", "args": {"item": i2}},
            ],
            "order_sensitive": False,
        })

    # Pattern C: calendar+price (unordered)
    for date, item in itertools.product(dates, items):
        pool.append({
            "prompt": f"Check my calendar for {date} and look up the price of a {item}.",
            "expected_sequence": [
                {"tool": "get_calendar_events", "args": {"date": date}},
                {"tool": "lookup_price", "args": {"item": item}},
            ],
            "order_sensitive": False,
        })

    # Pattern D: currency rate then convert (order-sensitive, same pair+amount)
    for (base, target) in CURRENCY_RATES:
        for amount in AMOUNTS:
            pool.append({
                "prompt": f"What's the {base.upper()} to {target.upper()} rate, and how much is {amount} {base.upper()} in {target.upper()}?",
                "expected_sequence": [
                    {"tool": "get_currency_rate", "args": {"base": base.upper(), "target": target.upper()}},
                    {"tool": "convert_currency", "args": {"amount": amount, "base": base.upper(), "target": target.upper()}},
                ],
                "order_sensitive": False,
            })

    return pool


def sample_tier2_training(n: int, seed: int = 42) -> list:
    """Samples n Tier 2 training examples, zero overlap with tasks/tier2.py's
    TIER2_TASKS (eval) or TIER2_HELDOUT (generalization)."""
    reserved_prompts = {t["prompt"] for t in TIER2_TASKS} | {t["prompt"] for t in TIER2_HELDOUT}
    pool = [ex for ex in generate_tier2_pool() if ex["prompt"] not in reserved_prompts]

    seen = set()
    deduped = []
    for ex in pool:
        if ex["prompt"] not in seen:
            seen.add(ex["prompt"])
            deduped.append(ex)

    if n > len(deduped):
        raise ValueError(f"Requested {n} Tier 2 training examples but only {len(deduped)} available.")

    rng = random.Random(seed)
    return rng.sample(deduped, n)


def generate_tier3_pool() -> list:
    """
    Tier 3 training pool, built from "error" type tasks only (not
    "ambiguous" type — the calendar-conflict pattern only has one valid
    instance in CALENDAR_DATA, already used in TIER3_TASKS, so there is
    no room to generate more without modifying the simulator itself).

    Uses nonexistent identifiers (fake city/item/route/currency/flight-ID/
    date values not present in the simulator's data) to naturally trigger
    "not found" style errors. This gives an effectively large combination
    space with no leakage risk, unlike Tier 2's constrained valid-entity
    space.
    """
    pool = []

    FAKE_CITIES = ["Wakanda", "Narnia", "Gotham", "Zion", "Asgard", "Elysium",
                   "Metropolis", "Themyscira", "Hobbiton", "Rivendell",
                   "Oz", "Neverland", "Shangri-La", "El Dorado", "Utopia"]
    for city in FAKE_CITIES:
        pool.append({
            "prompt": f"What's the weather in {city}?",
            "type": "error",
            "trigger_tool": "get_weather", "trigger_args": {"city": city},
            "must_not_call": [],
            "recovery_keywords": ["not found", "couldn't find", "don't have", "no data",
                                   "unable", "doesn't exist", "not available", "not a valid"],
        })

    FAKE_ITEMS = ["smartwatch", "tripod", "webcam", "printer", "scanner",
                  "router", "speaker", "microphone", "drone", "projector"]
    for item in FAKE_ITEMS:
        pool.append({
            "prompt": f"What's the price of a {item}?",
            "type": "error",
            "trigger_tool": "lookup_price", "trigger_args": {"item": item},
            "must_not_call": [],
            "recovery_keywords": ["not found", "don't have", "not available", "not in",
                                   "unable", "no price", "does not have"],
        })

    real_cities = [_title(c) for c in WEATHER_DATA]
    real_routes = {(o, d) for (o, d) in FLIGHT_DATA}
    fake_routes = []
    for o in real_cities:
        for d in real_cities:
            if o != d and (o.lower(), d.lower()) not in real_routes:
                fake_routes.append((o, d))
    rng_routes = random.Random(7)
    rng_routes.shuffle(fake_routes)
    for (origin, dest) in fake_routes[:15]:
        pool.append({
            "prompt": f"Find a flight from {origin} to {dest} and book it.",
            "type": "error",
            "trigger_tool": "search_flights", "trigger_args": {"origin": origin, "destination": dest},
            "must_not_call": ["book_flight"],
            "recovery_keywords": ["no flights", "not found", "unavailable", "couldn't find",
                                   "no route", "unable", "does not", "no available"],
        })

    FAKE_FLIGHT_IDS = ["XY456", "QR789", "JK321", "MN654", "PL987", "ST246", "UV135"]
    for fid in FAKE_FLIGHT_IDS:
        pool.append({
            "prompt": f"Book flight {fid} for me.",
            "type": "error",
            "trigger_tool": "book_flight", "trigger_args": {"flight_id": fid},
            "must_not_call": [],
            "recovery_keywords": ["invalid", "not found", "doesn't exist", "unable",
                                   "couldn't", "no such flight", "not a valid"],
        })

    real_pairs = set(CURRENCY_RATES.keys())
    currencies = ["USD", "EUR", "GBP", "INR", "JPY", "AUD", "CAD", "CHF"]
    fake_pairs = []
    for b in currencies:
        for t in currencies:
            if b != t and (b.lower(), t.lower()) not in real_pairs:
                fake_pairs.append((b, t))
    rng_pairs = random.Random(11)
    rng_pairs.shuffle(fake_pairs)
    for (base, target) in fake_pairs[:12]:
        pool.append({
            "prompt": f"What's the exchange rate from {base} to {target}?",
            "type": "error",
            "trigger_tool": "get_currency_rate", "trigger_args": {"base": base, "target": target},
            "must_not_call": ["convert_currency"],
            "recovery_keywords": ["unsupported", "not supported", "don't support",
                                   "does not support", "not available", "unable", "not a valid"],
        })

    FAKE_DATES = ["2026-09-01", "2026-09-05", "2026-09-10", "2026-09-15",
                  "2026-09-20", "2026-09-25", "2026-10-01"]
    for date in FAKE_DATES:
        pool.append({
            "prompt": f"What's on my calendar for {date}?",
            "type": "error",
            "trigger_tool": "get_calendar_events", "trigger_args": {"date": date},
            "must_not_call": [],
            "recovery_keywords": ["no data", "not found", "don't have", "unable",
                                   "no calendar", "no information", "does not have"],
        })

    return pool


def sample_tier3_training(n: int, seed: int = 42) -> list:
    """Samples n Tier 3 training examples, zero overlap with tasks/tier3.py's
    TIER3_TASKS (eval) or TIER3_HELDOUT (generalization)."""
    reserved_prompts = {t["prompt"] for t in TIER3_TASKS} | {t["prompt"] for t in TIER3_HELDOUT}
    pool = [ex for ex in generate_tier3_pool() if ex["prompt"] not in reserved_prompts]

    seen = set()
    deduped = []
    for ex in pool:
        if ex["prompt"] not in seen:
            seen.add(ex["prompt"])
            deduped.append(ex)

    if n > len(deduped):
        raise ValueError(f"Requested {n} Tier 3 training examples but only {len(deduped)} available.")

    rng = random.Random(seed)
    return rng.sample(deduped, n)


if __name__ == "__main__":
    pool = generate_tier1_pool()
    print(f"Total generated pool size: {len(pool)}")

    reserved = {t["prompt"] for t in TIER1_TASKS} | {t["prompt"] for t in TIER1_HELDOUT}
    overlap = [ex for ex in pool if ex["prompt"] in reserved]
    print(f"Overlap with eval/held-out (should be small/handled by filtering): {len(overlap)}")

    train_10 = sample_tier1_training(10)
    train_50 = sample_tier1_training(50)
    print(f"\nSampled {len(train_10)} for N=10 regime, {len(train_50)} for N=50 regime.")

    # confirm zero leakage into eval/held-out
    train_50_prompts = {ex["prompt"] for ex in train_50}
    leak = train_50_prompts & reserved
    print(f"Leakage check (should be 0): {len(leak)}")

    print("\nSample of 3 generated training examples:")
    for ex in train_10[:3]:
        print(" ", ex)

    print("\n" + "=" * 60)
    print("TIER 2")
    print("=" * 60)
    t2_pool = generate_tier2_pool()
    print(f"Total generated Tier 2 pool size: {len(t2_pool)}")
    t2_train_10 = sample_tier2_training(10)
    print(f"Sampled {len(t2_train_10)} for N=10 regime.")
    t2_reserved = {t["prompt"] for t in TIER2_TASKS} | {t["prompt"] for t in TIER2_HELDOUT}
    t2_leak = {ex["prompt"] for ex in t2_train_10} & t2_reserved
    print(f"Leakage check (should be 0): {len(t2_leak)}")
    print("\nSample of 3 generated Tier 2 training examples:")
    for ex in t2_train_10[:3]:
        print(" ", ex)

    print("\n" + "=" * 60)
    print("TIER 3")
    print("=" * 60)
    t3_pool = generate_tier3_pool()
    print(f"Total generated Tier 3 pool size: {len(t3_pool)}")
    t3_train_10 = sample_tier3_training(10)
    print(f"Sampled {len(t3_train_10)} for N=10 regime.")
    t3_reserved = {t["prompt"] for t in TIER3_TASKS} | {t["prompt"] for t in TIER3_HELDOUT}
    t3_leak = {ex["prompt"] for ex in t3_train_10} & t3_reserved
    print(f"Leakage check (should be 0): {len(t3_leak)}")
    print("\nSample of 3 generated Tier 3 training examples:")
    for ex in t3_train_10[:3]:
        print(" ", ex)
