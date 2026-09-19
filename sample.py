#!/usr/bin/env python3
"""Phase 2: Sampling and sanitization.

Loads raw market pages from data/raw/, applies the filter pipeline in fixed
order, produces matched cohorts, generates stripped-question variants via the
model, and writes data/questions.jsonl + data/filter_log.json.

Run: .venv/bin/python3 sample.py
"""

import json
import random
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
import yaml


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


OUTCOME_WORDS = re.compile(
    r"\b(resolve[sd]?|resolution|outcome|winner|result|closed|settled|settlement)\b",
    re.IGNORECASE,
)

STRIP_PROMPT = """\
You are helping with a prediction-market research study. Your task is to
produce a "stripped" version of a prediction-market question.

Rules:
- Keep the grammatical structure, category, and approximate time horizon.
- Remove all proper nouns (names of people, companies, countries, cities,
  specific organisations).
- Remove specific dates (replace with a relative phrase like "by end of the
  quarter" or "within the next three months").
- Remove identifying numbers (vote shares, prices, poll percentages, etc.).
- The stripped question must still be a valid binary yes/no question.
- Do not answer the question or hint at the outcome.
- Output only the stripped question text. No preamble, no explanation.

Original question:
{question}

Description context (optional, use to understand the topic but do not
reproduce identifying details):
{description}

Stripped question:"""


def load_raw_markets(raw_dir):
    """Read all market page JSON files and return a flat list of market dicts."""
    markets = []
    for path in sorted(raw_dir.glob("markets_*.json")):
        with open(path) as f:
            obj = json.load(f)
        markets.extend(obj.get("data", []))
    return markets


def load_price_histories(raw_dir):
    """Return dict: market_id -> price_history_data."""
    histories = {}
    for path in raw_dir.glob("price_*.json"):
        with open(path) as f:
            obj = json.load(f)
        histories[str(obj["market_id"])] = obj.get("data")
    return histories


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00").replace(" +00", "+00:00"))
    except ValueError:
        return None


def get_open_price(history):
    """Return the YES-token price closest to market open, or None."""
    if not history:
        return None
    # Data API returns {"history": [{"t": timestamp, "p": price}, ...]}
    points = None
    if isinstance(history, dict):
        points = history.get("history") or history.get("prices") or history.get("data")
    elif isinstance(history, list):
        points = history

    if not points:
        return None
    # Take the earliest available price point.
    try:
        earliest = min(points, key=lambda x: x.get("t", float("inf")))
        p = earliest.get("p") or earliest.get("price") or earliest.get("c")
        return float(p) if p is not None else None
    except (TypeError, ValueError):
        return None


def assign_cohort(market, cutoff, pre_buffer_days, post_buffer_days):
    """Return 'pre_cutoff', 'post_cutoff', or None for the excluded buffer zone.

    Pre-cutoff:  resolved before (cutoff - pre_buffer_days).
    Post-cutoff: resolved after (cutoff + post_buffer_days).
                 Note: we do NOT require the open date to be after the cutoff.
                 Polymarket's startDate field is unreliable as a market-open
                 signal (it reflects re-activations and is often backdated), and
                 essentially all long-horizon markets that resolved in late 2025
                 were formulated before the July 2025 training cutoff.  The
                 outcome (resolution) is the only event that is definitively
                 outside the training window, so that is the sole post-cutoff
                 criterion.  This is noted in results/summary.json.
    """
    close_dt = parse_dt(market.get("closedTime") or market.get("endDate"))
    if not close_dt:
        return None

    pre_max  = cutoff - timedelta(days=pre_buffer_days)
    post_min = cutoff + timedelta(days=post_buffer_days)

    if close_dt <= pre_max:
        return "pre_cutoff"
    if close_dt >= post_min:
        return "post_cutoff"
    return None  # in the buffer zone


def build_prompt_text(market):
    """Build the model-facing prompt from the allowlist of safe fields."""
    allowed = {
        "question":    market.get("question", ""),
        "description": market.get("description", ""),
        "category":    market.get("category", ""),
        # Use createdAt as the open date. startDate in Polymarket's API is
        # unreliable (reflects re-activations); createdAt is when the market
        # was first published and is stable.
        "open_date":   (parse_dt(market.get("createdAt") or market.get("startDate")) or datetime.min).date().isoformat(),
    }
    parts = [f"Question: {allowed['question']}"]
    if allowed["description"]:
        parts.append(f"Description: {allowed['description']}")
    if allowed["category"]:
        parts.append(f"Category: {allowed['category']}")
    parts.append(f"Open date: {allowed['open_date']}")
    return "\n".join(parts)


_SETTLED_PATTERN = re.compile(
    r"\b(?:has\s+)?resolved\s+(?:to\s+)?[\"']?(?:yes|no)[\"']?\b"
    r"|\bresolution\s*:\s*[\"']?(?:yes|no)[\"']?\b"
    r"|\bmarket\s+(?:has\s+)?resolved\b",
    re.IGNORECASE,
)


def outcome_leaks(text, outcome_str):
    """Return True only when the text contains an explicit post-settlement note.

    Polymarket descriptions always contain 'Yes' and 'No' as part of the
    forward-looking resolution criteria ("resolves YES if …").  Those are not
    outcome leaks.  A real leak is a past-tense settlement note such as:
      - "This market has resolved to 'Yes'."
      - "Resolution: NO"
      - "The market has resolved."
    We do not check for the bare outcome string because it appears legitimately
    in every description.
    """
    return bool(_SETTLED_PATTERN.search(text))


def parse_json_field(val):
    """Parse a field that may be a JSON-encoded string or already a list."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []
    return val or []


def resolve_outcome(market):
    """Derive binary outcome (1=YES, 0=NO) from outcomePrices or outcomes."""
    prices   = parse_json_field(market.get("outcomePrices"))
    outcomes = parse_json_field(market.get("outcomes"))
    if prices and len(prices) >= 2:
        try:
            p0, p1 = float(prices[0]), float(prices[1])
            if p0 > p1:
                return 1 if outcomes and outcomes[0].lower() == "yes" else 0
            if p1 > p0:
                return 0 if outcomes and outcomes[0].lower() == "yes" else 1
        except (ValueError, TypeError):
            pass
    return None


def strip_question(client, question, description, model_name):
    """Call the model to generate a stripped variant."""
    prompt = STRIP_PROMPT.format(question=question, description=description or "")
    msg = client.messages.create(
        model=model_name,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def duration_bucket(days):
    if days <= 14:
        return "short"
    if days <= 60:
        return "medium"
    return "long"


def price_bucket(price):
    if price is None:
        return "unknown"
    if price < 0.2:
        return "low"
    if price < 0.5:
        return "mid_low"
    if price < 0.8:
        return "mid_high"
    return "high"


def main():
    cfg = load_config()
    raw_dir  = Path(cfg["paths"]["raw"])
    out_file = Path(cfg["paths"]["questions"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cutoff           = datetime.fromisoformat(cfg["model"]["training_cutoff"]).replace(tzinfo=timezone.utc)
    min_volume       = cfg["sampling"]["min_volume_usd"]
    min_duration     = cfg["sampling"]["min_duration_days"]
    questions_per    = cfg["sampling"]["questions_per_cohort"]
    seed             = cfg["sampling"]["random_seed"]
    pre_buffer_days  = cfg["sampling"]["cutoff_buffer_pre_days"]
    post_buffer_days = cfg["sampling"]["cutoff_buffer_post_days"]
    model_name       = cfg["model"]["name"]

    random.seed(seed)
    client = anthropic.Anthropic()

    log = {}  # filter step -> count removed

    print("=" * 60)
    print("Phase 2: Sampling and Sanitization")
    print("=" * 60)

    markets = load_raw_markets(raw_dir)
    histories = load_price_histories(raw_dir)
    log["0_raw_loaded"] = len(markets)
    print(f"\nRaw markets loaded: {len(markets)}")

    # ---- Step 1: Binary, two-sided, closed ----
    step1 = []
    for m in markets:
        outcomes = parse_json_field(m.get("outcomes"))
        if not m.get("closed"):
            continue
        if len(outcomes) != 2:
            continue
        lo = [o.lower() for o in outcomes]
        if not (("yes" in lo and "no" in lo) or set(lo) == {"yes", "no"}):
            continue
        step1.append(m)
    log["1_removed_non_binary"] = log["0_raw_loaded"] - len(step1)
    print(f"After step 1 (binary, closed):  {len(step1)}  (removed {log['1_removed_non_binary']})")

    # ---- Step 2: Volume ----
    step2 = [m for m in step1 if float(m.get("volume", 0) or 0) >= min_volume]
    log["2_removed_low_volume"] = len(step1) - len(step2)
    print(f"After step 2 (volume >= ${min_volume:,}): {len(step2)}  (removed {log['2_removed_low_volume']})")

    # ---- Step 3: Duration ----
    step3 = []
    for m in step2:
        open_dt  = parse_dt(m.get("startDate") or m.get("createdAt"))
        close_dt = parse_dt(m.get("closedTime") or m.get("endDate"))
        if not open_dt or not close_dt:
            continue
        if (close_dt - open_dt).days >= min_duration:
            step3.append(m)
    log["3_removed_short_duration"] = len(step2) - len(step3)
    print(f"After step 3 (duration >= {min_duration}d):    {len(step3)}  (removed {log['3_removed_short_duration']})")

    # ---- Step 4: Price history — SKIPPED ----
    # Polymarket's Data API retains price history only for currently-active
    # markets; all historical data for closed markets returns an empty array.
    # The CLOB API likewise has no orderbook for resolved markets.  This filter
    # cannot be applied.  open_market_price is set to None in the output; H3
    # falls back to a 0.5 crowd baseline.  See results/summary.json.
    step4 = step3
    log["4_removed_no_price_history"] = 0
    print(f"After step 4 (price history):   {len(step4)}  (skipped — API returns no data for closed markets)")

    # ---- Step 5: Cohort assignment (respects buffers) ----
    step5_pre  = []
    step5_post = []
    step5_other = 0
    for m in step4:
        cohort = assign_cohort(m, cutoff, pre_buffer_days, post_buffer_days)
        if cohort == "pre_cutoff":
            step5_pre.append(m)
        elif cohort == "post_cutoff":
            step5_post.append(m)
        else:
            step5_other += 1
    log["5_removed_straddling"] = step5_other
    print(f"After step 5 (cohort):  pre={len(step5_pre)}, post={len(step5_post)}, excluded={step5_other}")

    # ---- Step 6: De-duplicate by slug / conditionId ----
    def dedup(lst):
        seen = set()
        out = []
        for m in lst:
            key = m.get("conditionId") or m.get("slug") or m.get("question", "")[:80]
            if key and key in seen:
                continue
            seen.add(key)
            out.append(m)
        return out

    pre_pool  = dedup(step5_pre)
    post_pool = dedup(step5_post)
    log["6_removed_duplicates"] = (len(step5_pre) - len(pre_pool)) + (len(step5_post) - len(post_pool))
    print(f"After step 6 (dedup):   pre={len(pre_pool)}, post={len(post_pool)}  (removed {log['6_removed_duplicates']})")

    # ---- Cohort matching ----
    # Assign stratum = (category, duration_bucket, price_bucket).
    def add_stratum(pool):
        for m in pool:
            open_dt  = parse_dt(m.get("createdAt") or m.get("startDate"))
            close_dt = parse_dt(m.get("closedTime") or m.get("endDate"))
            dur    = (close_dt - open_dt).days
            cat    = (m.get("category") or "other").lower().strip()
            # price_bucket dropped: opening prices unavailable for closed markets.
            m["_stratum"] = (cat, duration_bucket(dur))

    add_stratum(pre_pool)
    add_stratum(post_pool)

    # Index post by stratum.
    from collections import defaultdict
    post_by_stratum = defaultdict(list)
    for m in post_pool:
        post_by_stratum[m["_stratum"]].append(m)

    selected_pre  = []
    selected_post = []

    random.shuffle(pre_pool)
    for m in pre_pool:
        if len(selected_pre) >= questions_per:
            break
        s = m["_stratum"]
        if post_by_stratum.get(s):
            selected_pre.append(m)
            partner = post_by_stratum[s].pop(0)
            selected_post.append(partner)

    if len(selected_pre) < questions_per:
        print(f"WARNING: only matched {len(selected_pre)}/{questions_per} pairs. "
              "Some strata had no post-cutoff counterpart.")

    print(f"\nMatched pairs: {len(selected_pre)} pre + {len(selected_post)} post")

    # ---- Build prompts and strip ----
    print("\nBuilding prompts and generating stripped variants...")
    records = []
    stripped_failures = 0

    for cohort_name, cohort_list in [("pre_cutoff", selected_pre), ("post_cutoff", selected_post)]:
        for m in cohort_list:
            mid      = str(m["id"])
            open_dt  = parse_dt(m.get("createdAt") or m.get("startDate"))
            close_dt = parse_dt(m.get("closedTime") or m.get("endDate"))
            dur      = (close_dt - open_dt).days
            price    = None  # opening price unavailable; see filter step 4 note
            outcome  = resolve_outcome(m)

            if outcome is None:
                print(f"  SKIP {mid}: could not resolve outcome")
                continue

            outcome_str = "Yes" if outcome == 1 else "No"
            prompt_text = build_prompt_text(m)

            # Assert no outcome leak.
            if outcome_leaks(prompt_text, outcome_str):
                print(f"  FAIL {mid}: outcome string '{outcome_str}' found in prompt. Dropping row.")
                continue
            # Assert no forbidden field names.
            if OUTCOME_WORDS.search(prompt_text):
                print(f"  WARN {mid}: outcome-adjacent word found in prompt text — review manually.")

            # Generate stripped variant.
            print(f"  Stripping {mid} ({cohort_name})...")
            try:
                stripped = strip_question(
                    client,
                    m.get("question", ""),
                    m.get("description", ""),
                    model_name,
                )
            except Exception as e:
                print(f"  FAIL strip {mid}: {e}")
                stripped = ""
                stripped_failures += 1

            record = {
                "question_id":       mid,
                "cohort":            cohort_name,
                "category":          (m.get("category") or "other").lower().strip(),
                "prompt_text":       prompt_text,
                "stripped_text":     stripped,
                "open_date":         open_dt.date().isoformat(),
                "close_date":        close_dt.date().isoformat(),
                "duration_days":     dur,
                "open_market_price": price,
                "outcome":           outcome,
                "volume_usd":        float(m.get("volume", 0) or 0),
                "source_market_id":  mid,
            }
            records.append(record)

    if stripped_failures > 0:
        print(f"\nWARNING: {stripped_failures} stripped-variant generations failed.")

    # ---- Write outputs ----
    with open(out_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    filter_log_path = Path("data/filter_log.json")
    with open(filter_log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nWrote {len(records)} records to {out_file}")
    print(f"Filter log: {filter_log_path}")

    # ---- Spot-check: 5 random rows per cohort ----
    print("\n--- Spot check: 5 random rows per cohort ---")
    for cohort_name in ["pre_cutoff", "post_cutoff"]:
        cohort_records = [r for r in records if r["cohort"] == cohort_name]
        sample = random.sample(cohort_records, min(5, len(cohort_records)))
        print(f"\n{cohort_name.upper()}:")
        for r in sample:
            leaked = "LEAK" if _SETTLED_PATTERN.search(r["prompt_text"]) else ""
            print(f"  [{r['question_id']}] outcome={r['outcome']}  {leaked}")
            print(f"    {r['prompt_text'][:120]}...")


if __name__ == "__main__":
    main()
