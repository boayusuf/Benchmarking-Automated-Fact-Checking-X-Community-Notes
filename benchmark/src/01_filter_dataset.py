"""
Step 1: Filter the Community Notes dataset.

Produces two CSVs in data/filtered/:
- misleading_notes.csv  — English, helpful, MISINFORMED_OR_POTENTIALLY_MISLEADING, summary > 30 chars, no spam
- not_misleading_notes.csv — random sample of 300 English, helpful, NOT_MISLEADING notes (for Cohen's Kappa)

No API needed. Pure pandas + langdetect.
"""

from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42
RANDOM_STATE = 42

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
RAW_CSV = Path(r"C:\Users\yusuf\Documents\CN-Dataset\community_notes_clean.csv")
FILTERED_DIR = PROJECT_ROOT / "data" / "filtered"
FILTERED_DIR.mkdir(parents=True, exist_ok=True)

SPAM_TOKENS = ("spam", "phishing", "advertisement", "click here")


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def detect_language_column(series: pd.Series) -> pd.Series:
    tqdm.pandas(desc="langdetect")
    return series.progress_apply(is_english)


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: input CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1

    print(f"Reading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"  loaded rows: {len(df):,}")
    print(f"  columns: {list(df.columns)}")

    # ----- MISLEADING set -----
    print("\n--- Filtering MISLEADING notes ---")
    m = df[df["classification"] == "MISINFORMED_OR_POTENTIALLY_MISLEADING"]
    print(f"after classification == MISINFORMED_OR_POTENTIALLY_MISLEADING: {len(m):,}")

    m = m[m["currentStatus"] == "CURRENTLY_RATED_HELPFUL"]
    print(f"after currentStatus == CURRENTLY_RATED_HELPFUL:                {len(m):,}")

    m = m[m["summary"].notna()]
    m = m[m["summary"].astype(str).str.len() > 30]
    print(f"after summary length > 30 chars:                                {len(m):,}")

    summary_lower = m["summary"].astype(str).str.lower()
    spam_mask = pd.Series(False, index=m.index)
    for token in SPAM_TOKENS:
        spam_mask |= summary_lower.str.contains(token, na=False, regex=False)
    m = m[~spam_mask]
    print(f"after removing spam tokens {SPAM_TOKENS}: {len(m):,}")

    print("Detecting language (this is the slow step) ...")
    english_mask = detect_language_column(m["summary"].astype(str))
    m = m[english_mask.values]
    print(f"after language == English:                                      {len(m):,}")

    misleading_out = FILTERED_DIR / "misleading_notes.csv"
    m.to_csv(misleading_out, index=False)
    print(f"-> saved {len(m):,} misleading rows to {misleading_out}")

    # ----- NOT_MISLEADING set (sample of 300) -----
    print("\n--- Filtering NOT_MISLEADING notes (for Kappa) ---")
    n = df[df["classification"] == "NOT_MISLEADING"]
    print(f"after classification == NOT_MISLEADING:                         {len(n):,}")

    n = n[n["currentStatus"] == "CURRENTLY_RATED_HELPFUL"]
    print(f"after currentStatus == CURRENTLY_RATED_HELPFUL:                 {len(n):,}")

    n = n[n["summary"].notna()]
    n = n[n["summary"].astype(str).str.len() > 30]
    print(f"after summary length > 30 chars:                                {len(n):,}")

    print("Detecting language ...")
    english_mask_n = detect_language_column(n["summary"].astype(str))
    n = n[english_mask_n.values]
    print(f"after language == English:                                      {len(n):,}")

    sample_size = min(300, len(n))
    n_sample = n.sample(n=sample_size, random_state=RANDOM_STATE)
    not_misleading_out = FILTERED_DIR / "not_misleading_notes.csv"
    n_sample.to_csv(not_misleading_out, index=False)
    print(f"-> saved {len(n_sample):,} not-misleading rows to {not_misleading_out}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
