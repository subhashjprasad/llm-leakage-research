#!/usr/bin/env python3
"""Phase 3: Elicitation.

Loads data/questions.jsonl, calls the configured model runs_per_question times
for each (question, variant) pair, and writes data/forecasts.jsonl.

Keyed cache: already-completed (question_id, variant, run_index) triples are
skipped so a crash is resumable.  Refusals and parse failures are written to
data/elicit_errors.jsonl, never silently dropped.

Run: .venv/bin/python3 elicit.py
"""

import json
import re
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import yaml


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


ELICIT_PROMPT = """\
You are a careful forecaster. You will be given a prediction-market question
along with its description and the date on which the market opened.

Your task is to estimate the probability that the answer is YES, using only
information that would have been available on the open date. Do not search the
web or use any external tools. Give a probability even if you are uncertain.

{question_text}

Respond with a JSON object containing a single field "probability" whose value
is a number between 0 and 1 (inclusive). Example: {{"probability": 0.35}}

Respond with the JSON object only. No explanation, no preamble."""


def load_questions(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_done(forecasts_path):
    """Return set of (question_id, variant, run_index) already written."""
    done = set()
    if not forecasts_path.exists():
        return done
    with open(forecasts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r["question_id"], r["variant"], r["run_index"]))
    return done


def call_model(client, model_name, temperature, question_text):
    """Return (probability: float, raw_response: str) or raise."""
    prompt = ELICIT_PROMPT.format(question_text=question_text)
    # temperature removed: SDK 1.6.0 / Claude 4.x API no longer accepts it.
    # Haiku 4.5 also does not support the effort parameter.
    # Run-to-run variance comes from the model's internal sampling.
    msg = client.messages.create(
        model=model_name,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    # Collect text from all text blocks (skip thinking blocks if present).
    raw_parts = [block.text for block in msg.content if block.type == "text"]
    raw = "\n".join(raw_parts).strip()

    # Strip markdown code fences if present.
    clean = re.sub(r"^```(?:json)?\s*", "", raw)
    clean = re.sub(r"\s*```$", "", clean).strip()

    # Parse exactly one JSON object with a "probability" key.
    parsed = json.loads(clean)
    prob = float(parsed["probability"])
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"probability {prob} out of [0, 1]")
    return prob, raw


def main():
    cfg          = load_config()
    questions_path  = Path(cfg["paths"]["questions"])
    forecasts_path  = Path(cfg["paths"]["forecasts"])
    errors_path     = Path("data/elicit_errors.jsonl")
    forecasts_path.parent.mkdir(parents=True, exist_ok=True)

    model_name   = cfg["model"]["name"]
    temperature  = cfg["model"]["temperature"]
    runs         = cfg["model"]["runs_per_question"]

    client = anthropic.Anthropic()

    questions = load_questions(questions_path)
    done      = load_done(forecasts_path)

    print("=" * 60)
    print("Phase 3: Elicitation")
    print(f"  Model:              {model_name}")
    print(f"  Temperature:        {temperature}")
    print(f"  Runs per question:  {runs}")
    print(f"  Questions loaded:   {len(questions)}")
    print(f"  Already cached:     {len(done)}")
    print("=" * 60)

    # Variants: full text and stripped text.
    variants = [
        ("full",    "prompt_text"),
        ("stripped", "stripped_text"),
    ]

    total_runs    = len(questions) * len(variants) * runs
    remaining     = total_runs - len(done)
    print(f"\nTotal runs planned: {total_runs}  Remaining: {remaining}\n")

    successes = refusals = parse_failures = 0

    forecast_fh = open(forecasts_path, "a")
    error_fh    = open(errors_path, "a")

    try:
        for q in questions:
            qid = q["question_id"]
            for variant_name, text_field in variants:
                question_text = q.get(text_field, "")
                if not question_text:
                    continue
                for run_idx in range(runs):
                    key = (qid, variant_name, run_idx)
                    if key in done:
                        continue

                    print(f"  {qid} / {variant_name} / run {run_idx}...", end=" ", flush=True)
                    try:
                        prob, raw = call_model(client, model_name, temperature, question_text)
                        record = {
                            "question_id":  qid,
                            "variant":      variant_name,
                            "run_index":    run_idx,
                            "probability":  prob,
                            "raw_response": raw,
                            "model":        model_name,
                            "timestamp":    datetime.now(timezone.utc).isoformat(),
                        }
                        forecast_fh.write(json.dumps(record) + "\n")
                        forecast_fh.flush()
                        done.add(key)
                        successes += 1
                        print(f"p={prob:.3f}")

                    except anthropic.APIStatusError as e:
                        # Refusals come back as content-policy errors.
                        err_type = "refusal"
                        raw_err  = str(e)
                        print(f"REFUSAL: {raw_err[:80]}")
                        refusals += 1
                        error_fh.write(json.dumps({
                            "question_id": qid,
                            "variant":     variant_name,
                            "run_index":   run_idx,
                            "error_type":  err_type,
                            "detail":      raw_err,
                            "timestamp":   datetime.now(timezone.utc).isoformat(),
                        }) + "\n")
                        error_fh.flush()

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        # Model returned something we couldn't parse.
                        print(f"PARSE FAIL: {e}")
                        parse_failures += 1
                        error_fh.write(json.dumps({
                            "question_id": qid,
                            "variant":     variant_name,
                            "run_index":   run_idx,
                            "error_type":  "parse_failure",
                            "detail":      str(e),
                            "timestamp":   datetime.now(timezone.utc).isoformat(),
                        }) + "\n")
                        error_fh.flush()

                    time.sleep(0.3)
    finally:
        forecast_fh.close()
        error_fh.close()

    print(f"\nDone.")
    print(f"  Successful calls:  {successes}")
    print(f"  Refusals:          {refusals}")
    print(f"  Parse failures:    {parse_failures}")
    print(f"  Forecasts file:    {forecasts_path}")
    if refusals or parse_failures:
        print(f"  Error log:         {errors_path}")


if __name__ == "__main__":
    main()
