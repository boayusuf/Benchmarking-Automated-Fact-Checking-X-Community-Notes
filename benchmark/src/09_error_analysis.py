"""
Step 9: Stratified random sample of disagreement cases for manual error coding.

Updated 2026-04-29: bumped PER_CATEGORY 20 -> 34 per the dissertation review.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
RESULTS = PROJECT_ROOT / "data" / "results"
RANDOM_STATE = 42
PER_CATEGORY = 34
CATEGORIES = ("POLITICAL", "HEALTH", "SCIENTIFIC")


def map_strict(v) -> int | None:
    if pd.isna(v): return None
    v = str(v).strip().upper()
    if v == "TRUE":  return 1
    if v == "FALSE": return 0
    return None


def cn_to_int(c) -> int:
    return 1 if c == "NOT_MISLEADING" else 0


def has_disagreement(row) -> bool:
    truth = cn_to_int(row["cn_verdict"])
    for col in ("gpt4o_verdict", "tavily_gpt4o_verdict", "google_verdict"):
        m = map_strict(row.get(col))
        if m is not None and m != truth:
            return True
    return False


def main() -> int:
    df = pd.read_csv(RESULTS / "master_results.csv")
    print(f"  loaded master_results: {len(df)} rows", flush=True)

    df["any_disagreement"] = df.apply(has_disagreement, axis=1)
    disagree = df[df["any_disagreement"]].copy()
    print(f"  rows with at least one AFC disagreement: {len(disagree)}", flush=True)

    samples = []
    for cat in CATEGORIES:
        pool = disagree[disagree["category"] == cat]
        n = min(PER_CATEGORY, len(pool))
        if n < PER_CATEGORY:
            print(f"  WARNING: only {n} disagreements in {cat} (target {PER_CATEGORY})", flush=True)
        samples.append(pool.sample(n=n, random_state=RANDOM_STATE))
    sample = pd.concat(samples).reset_index(drop=True)

    out = sample[[
        "noteId", "tweetId", "category", "summary", "extracted_claim", "cn_verdict",
        "gpt4o_verdict", "gpt4o_confidence", "gpt4o_reasoning",
        "tavily_gpt4o_verdict", "tavily_gpt4o_confidence", "tavily_gpt4o_reasoning",
        "google_verdict", "google_source", "google_raw_rating",
    ]].copy()
    out["error_category"] = ""
    out["notes"] = ""

    out_path = RESULTS / "error_analysis_cases.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  saved {len(out)} stratified cases -> {out_path}", flush=True)
    print(out.groupby(["category", "cn_verdict"]).size(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
