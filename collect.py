#!/usr/bin/env python3
"""Phase 1: Collect resolved binary markets from Polymarket.

Paginates the Gamma API for closed markets in two date windows (pre-cutoff and
post-cutoff), saves every raw response unmodified to data/raw/, then fetches
price history for each candidate market from the Data API.

Run: .venv/bin/python3 collect.py
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import yaml


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


GAMMA = "https://gamma-api.polymarket.com"
DATA  = "https://data-api.polymarket.com"


def fetch_with_backoff(url, params=None, max_retries=5):
    """GET with exponential backoff. Raises on repeated failure."""
    delay = 2
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                print(f"  Rate limited. Waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"FATAL: {url} failed after {max_retries} attempts: {e}")
                raise
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def paginate_markets(raw_dir, label, extra_params, target):
    """Fetch pages of closed markets, writing each page to raw_dir.

    Returns a flat list of all market dicts collected (up to target).
    """
    limit = 100
    offset = 0
    page = 0
    all_markets = []

    while len(all_markets) < target:
        params = {"closed": "true", "limit": limit, "offset": offset, **extra_params}
        print(f"  {label} page {page} (offset={offset})...")
        data = fetch_with_backoff(f"{GAMMA}/markets", params)

        # Write raw page unmodified with request provenance.
        page_file = raw_dir / f"markets_{label}_p{page:04d}.json"
        with open(page_file, "w") as f:
            json.dump(
                {
                    "url": f"{GAMMA}/markets",
                    "params": params,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "data": data,
                },
                f,
                indent=2,
            )

        if not data:
            break
        all_markets.extend(data)
        print(f"    -> {len(data)} markets (running total: {len(all_markets)})")

        if len(data) < limit:
            break  # last page

        offset += limit
        page += 1
        time.sleep(0.4)

    return all_markets


def fetch_price_history(raw_dir, market_id, token_id, open_dt):
    """Fetch the first 24 h of price history after market open.

    Skips if the cache file already exists.  Returns the parsed JSON or None.
    """
    cache_file = raw_dir / f"price_{market_id}_{str(token_id)[:12]}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)["data"]

    start_ts = int(open_dt.timestamp())
    end_ts   = int((open_dt + timedelta(hours=24)).timestamp())

    params = {
        "token_id": token_id,
        "start":    start_ts,
        "end":      end_ts,
        "interval": "max",
    }
    try:
        result = fetch_with_backoff(f"{DATA}/v2/prices-history", params)
    except Exception as e:
        print(f"    price history failed for {market_id}: {e}")
        return None

    with open(cache_file, "w") as f:
        json.dump(
            {
                "market_id":    market_id,
                "token_id":     token_id,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "data":         result,
            },
            f,
            indent=2,
        )
    return result


def parse_dt(s):
    """Parse an ISO-8601 string to a UTC-aware datetime.  Returns None on failure."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00").replace(" +00", "+00:00"))
    except ValueError:
        return None


def main():
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    cutoff      = datetime.fromisoformat(cfg["model"]["training_cutoff"]).replace(tzinfo=timezone.utc)
    pre_buffer  = timedelta(days=cfg["sampling"]["cutoff_buffer_pre_days"])
    post_buffer = timedelta(days=cfg["sampling"]["cutoff_buffer_post_days"])
    target_pool = cfg["sampling"]["target_pool_size"]

    pre_max_date  = (cutoff - pre_buffer).date().isoformat()    # 2025-04-02
    post_min_date = (cutoff + post_buffer).date().isoformat()   # 2025-08-30

    print("=" * 60)
    print("Phase 1: Data Collection")
    print(f"  Training cutoff:      {cutoff.date()}")
    print(f"  Pre-cutoff window:    closed before {pre_max_date}")
    print(f"  Post-cutoff window:   closed after  {post_min_date}")
    print("=" * 60)

    # --- Pre-cutoff markets ---
    print("\nCollecting PRE-CUTOFF markets...")
    pre_markets = paginate_markets(
        raw_dir, "pre",
        extra_params={"end_date_min": "2023-01-01", "end_date_max": pre_max_date},
        target=target_pool,
    )

    # --- Post-cutoff markets ---
    print("\nCollecting POST-CUTOFF markets...")
    post_markets = paginate_markets(
        raw_dir, "post",
        extra_params={"end_date_min": post_min_date},
        target=target_pool,
    )

    all_markets = pre_markets + post_markets
    print(f"\nRaw markets collected: {len(pre_markets)} pre + {len(post_markets)} post = {len(all_markets)} total")

    # --- Date range check ---
    dates_pre  = [parse_dt(m.get("closedTime") or m.get("endDate")) for m in pre_markets]
    dates_post = [parse_dt(m.get("closedTime") or m.get("endDate")) for m in post_markets]
    dates_pre  = [d for d in dates_pre  if d]
    dates_post = [d for d in dates_post if d]

    if dates_pre:
        print(f"  Pre-cutoff date range:  {min(dates_pre).date()} – {max(dates_pre).date()}")
    if dates_post:
        print(f"  Post-cutoff date range: {min(dates_post).date()} – {max(dates_post).date()}")

    # Verify both sides of the cutoff are covered before proceeding.
    ok = True
    if not dates_pre or max(dates_pre).date().isoformat() > pre_max_date:
        print(f"WARNING: pre-cutoff max date exceeds {pre_max_date}. Inspect raw data.")
        ok = False
    if not dates_post or min(dates_post).date().isoformat() < post_min_date:
        print(f"WARNING: post-cutoff min date precedes {post_min_date}. Inspect raw data.")
        ok = False
    if ok:
        print("  Date range check: OK — both cohorts covered.")

    # --- Price history: NOT FETCHED ---
    # Polymarket's Data API returns an empty array for all closed/resolved
    # markets.  The CLOB API similarly has no orderbook for resolved markets.
    # Opening prices are therefore unavailable; see sample.py step 4 note.

    # --- Summary ---
    raw_files = list(raw_dir.glob("*.json"))
    print(f"\nSummary:")
    print(f"  data/raw/ files:     {len(raw_files)}")
    print(f"  Pre-cutoff markets:  {len(pre_markets)}")
    print(f"  Post-cutoff markets: {len(post_markets)}")
    print(f"  NOTE: Opening prices unavailable from Polymarket API for closed markets.")


if __name__ == "__main__":
    main()
