"""
Replace the 300-row `not_misleading_notes.csv` with a 2000-row sample so step 8
metrics have stable per-category Kappa for HEALTH/SCIENTIFIC.

Re-runs the not-misleading half of step 1 only — does NOT touch
`misleading_notes.csv`. Same seed (42) and same filters as `01_filter_dataset.py`.
"""

from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42
RANDOM_STATE = 42
NEW_SAMPLE_SIZE = 100000  # was 2000; bumped 2026-04-29 to take ALL English NM notes

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
RAW_CSV = Path(r"C:\Users\yusuf\Documents\CN-Dataset\community_notes_clean.csv")
FILTERED_DIR = PROJECT_ROOT / "data" / "filtered"
OUT_PATH = FILTERED_DIR / "not_misleading_notes.csv"


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: input CSV not found at {RAW_CSV}", file=sys.stderr)
        return 1

    print(f"Reading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"  loaded rows: {len(df):,}")

    print("\n--- Filtering NOT_MISLEADING notes (target n=2000) ---")
    n = df[df["classification"] == "NOT_MISLEADING"]
    print(f"after classification == NOT_MISLEADING:           {len(n):,}")

    # Spec said HELPFUL, but in the actual CN data only 2 NOT_MISLEADING notes are
    # HELPFUL — the original 300-row sample was built with NOT_HELPFUL too.
    n = n[n["currentStatus"] == "CURRENTLY_RATED_NOT_HELPFUL"]
    print(f"after currentStatus == CURRENTLY_RATED_NOT_HELPFUL: {len(n):,}")

    n = n[n["summary"].notna()]
    n = n[n["summary"].astype(str).str.len() > 30]
    print(f"after summary length > 30 chars:                  {len(n):,}")

    print("Detecting language (slow step) ...")
    tqdm.pandas(desc="langdetect")
    english_mask = n["summary"].astype(str).progress_apply(is_english)
    n = n[english_mask.values]
    print(f"after language == English:                        {len(n):,}")

    sample_size = min(NEW_SAMPLE_SIZE, len(n))
    n_sample = n.sample(n=sample_size, random_state=RANDOM_STATE)
    n_sample.to_csv(OUT_PATH, index=False)
    print(f"\n-> saved {len(n_sample):,} not-misleading rows to {OUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
