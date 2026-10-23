"""
Step 2: Classify Community Notes into POLITICAL / HEALTH / SCIENTIFIC / OTHER
using OpenAI gpt-4o-mini.

- Initial random sample of 5000 from misleading_notes.csv (seed=42).
- Early-stop once each of POLITICAL, HEALTH, SCIENTIFIC has >= 400 classified.
- Top up with another 3000-note random sample if any target category falls short.
- Then classify all 300 not-misleading notes the same way.
- Final output: data/filtered/classified_claims.csv with up to 400/cat misleading
  and up to 100/cat not-misleading (target cats only).
- Progress is saved every CHECKPOINT_EVERY notes; reruns resume automatically.
"""

from pathlib import Path
import os
import sys
import time

# Force UTF-8 so non-ASCII summaries can't crash a print().
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("ERROR: OPENAI_API_KEY missing from .env", file=sys.stderr)
    raise SystemExit(1)

client = OpenAI(api_key=API_KEY)

MODEL_NAME = "gpt-4o-mini"
RANDOM_STATE = 42
INITIAL_SAMPLE = 5000
TOP_UP_SAMPLE = 3000
TARGET_PER_CAT = 400
NOT_MISLEADING_PER_CAT = 100000  # was 100; bumped 2026-04-29 to keep ALL classified NM target-cat rows
TARGET_CATS = ("POLITICAL", "HEALTH", "SCIENTIFIC")
VALID_CATS = TARGET_CATS + ("OTHER",)
SLEEP_BETWEEN = 0.2     # seconds — gpt-4o-mini Tier 1 = 500 RPM, this stays ~150 RPM
CHECKPOINT_EVERY = 25
QUOTA_BAIL_THRESHOLD = 10

PROMPT = """Classify the following Community Note into exactly ONE category based on the claim it addresses. Respond with ONLY the category name, nothing else.

Categories:
- POLITICAL: Claims about political figures, policies, elections, government actions, legislation, political parties, wars, geopolitics
- HEALTH: Claims about medical treatments, diseases, vaccines, drugs, public health measures, nutrition, fitness
- SCIENTIFIC: Claims about scientific research, climate, technology, space, environment, natural phenomena, AI, engineering
- OTHER: Does not fit any above category

Community Note: "{summary}"
"""

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def classify_with_retry(summary: str, max_attempts: int = 5) -> str:
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": PROMPT.format(summary=summary)}],
                temperature=0,
                max_tokens=10,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            for cat in VALID_CATS:
                if cat in text:
                    return cat
            return "INVALID"
        except RETRYABLE as e:
            wait = (2 ** attempt) * 5
            print(f"\n  retry {attempt + 1}/{max_attempts} after {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  non-retryable error ({type(e).__name__}): {e}", flush=True)
            return "ERROR"
    preview = summary[:80].encode("ascii", errors="replace").decode("ascii")
    print(f"\n  exhausted retries on summary: {preview}...", flush=True)
    return "ERROR"


def classify_dataset(
    df: pd.DataFrame,
    progress_path: Path,
    target_per_cat: int | None = None,
    early_stop_cats: tuple[str, ...] | None = None,
    desc: str = "classifying",
) -> tuple[pd.DataFrame, dict]:
    if progress_path.exists():
        done = pd.read_csv(progress_path, dtype={"noteId": str, "tweetId": str})
        done_ids = set(done["noteId"].astype(str))
        results = done.to_dict("records")
        print(f"  resuming from {len(done_ids)} already-classified notes", flush=True)
    else:
        done_ids = set()
        results = []

    cat_counts = {c: 0 for c in VALID_CATS}
    cat_counts["INVALID"] = 0
    cat_counts["ERROR"] = 0
    for r in results:
        c = r.get("category")
        if c in cat_counts:
            cat_counts[c] += 1

    total = max(len(df), len(results))
    pbar = tqdm(total=total, desc=desc, initial=len(results), file=sys.stdout)
    n_since_save = 0
    consecutive_failures = 0
    quota_exhausted = False

    for _, row in df.iterrows():
        nid = str(row["noteId"])
        if nid in done_ids:
            continue
        if early_stop_cats and target_per_cat:
            if all(cat_counts[c] >= target_per_cat for c in early_stop_cats):
                print(f"\n  early stop: all of {early_stop_cats} reached {target_per_cat}", flush=True)
                break

        cat = classify_with_retry(str(row["summary"]))
        consecutive_failures = consecutive_failures + 1 if cat == "ERROR" else 0

        results.append(
            {
                "noteId": nid,
                "tweetId": str(row.get("tweetId", "")),
                "summary": row["summary"],
                "classification": row["classification"],
                "currentStatus": row["currentStatus"],
                "category": cat,
            }
        )
        done_ids.add(nid)
        if cat in cat_counts:
            cat_counts[cat] += 1
        n_since_save += 1
        pbar.update(1)
        pbar.set_postfix({c: cat_counts[c] for c in TARGET_CATS})

        if n_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            n_since_save = 0
            print(
                f"\n  checkpoint: {len(results)} done | "
                + " ".join(f"{c}={cat_counts[c]}" for c in TARGET_CATS),
                flush=True,
            )

        if consecutive_failures >= QUOTA_BAIL_THRESHOLD:
            print(
                f"\n  BAIL: {QUOTA_BAIL_THRESHOLD} consecutive retry-exhausted calls. Saving and exiting.",
                flush=True,
            )
            quota_exhausted = True
            break

        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(results).to_csv(progress_path, index=False)
    pbar.close()
    if quota_exhausted:
        raise SystemExit(2)
    return pd.DataFrame(results), cat_counts


def main() -> int:
    filtered = PROJECT_ROOT / "data" / "filtered"

    print("=== MISLEADING set ===", flush=True)
    m_full = pd.read_csv(filtered / "misleading_notes.csv", dtype={"noteId": str, "tweetId": str})
    print(f"  total misleading rows: {len(m_full):,}", flush=True)

    sample = m_full.sample(n=min(INITIAL_SAMPLE, len(m_full)), random_state=RANDOM_STATE)
    print(f"  initial sample size: {len(sample)}", flush=True)

    progress_m = filtered / "classified_progress_misleading.csv"
    df_m, counts_m = classify_dataset(
        sample, progress_m,
        target_per_cat=TARGET_PER_CAT,
        early_stop_cats=TARGET_CATS,
        desc="misleading",
    )
    print(f"\n  pass 1 counts: {counts_m}", flush=True)

    while any(counts_m[c] < TARGET_PER_CAT for c in TARGET_CATS):
        already = set(df_m["noteId"].astype(str))
        remaining = m_full[~m_full["noteId"].astype(str).isin(already)]
        if remaining.empty:
            print("  ! ran out of notes to top up from", flush=True)
            break
        topup = remaining.sample(n=min(TOP_UP_SAMPLE, len(remaining)), random_state=RANDOM_STATE)
        print(f"\n  topping up with {len(topup)} more notes", flush=True)
        sample_full = pd.concat([sample, topup])
        df_m, counts_m = classify_dataset(
            sample_full, progress_m,
            target_per_cat=TARGET_PER_CAT,
            early_stop_cats=TARGET_CATS,
            desc="misleading-topup",
        )
        sample = sample_full
        print(f"\n  counts after top-up: {counts_m}", flush=True)

    print("\n=== NOT_MISLEADING set (300) ===", flush=True)
    n_full = pd.read_csv(filtered / "not_misleading_notes.csv", dtype={"noteId": str, "tweetId": str})
    progress_n = filtered / "classified_progress_not_misleading.csv"
    df_n, counts_n = classify_dataset(n_full, progress_n, desc="not-misleading")
    print(f"\n  not-misleading counts: {counts_n}", flush=True)

    final_m_parts = []
    for c in TARGET_CATS:
        sub = df_m[df_m["category"] == c]
        n_take = min(TARGET_PER_CAT, len(sub))
        if n_take < TARGET_PER_CAT:
            print(f"  WARNING: only {n_take} {c} misleading notes (target {TARGET_PER_CAT})", flush=True)
        final_m_parts.append(sub.sample(n=n_take, random_state=RANDOM_STATE))
    final_m = pd.concat(final_m_parts)

    final_n_parts = []
    for c in TARGET_CATS:
        sub = df_n[df_n["category"] == c]
        n_take = min(NOT_MISLEADING_PER_CAT, len(sub))
        final_n_parts.append(sub.sample(n=n_take, random_state=RANDOM_STATE))
    final_n = pd.concat(final_n_parts)

    final = pd.concat([final_m, final_n], ignore_index=True)
    cols = ["noteId", "tweetId", "summary", "classification", "currentStatus", "category"]
    out_path = filtered / "classified_claims.csv"
    final[cols].to_csv(out_path, index=False)
    print(f"\n  saved {len(final)} rows -> {out_path}", flush=True)
    print("\nFinal distribution:", flush=True)
    print(final.groupby(["classification", "category"]).size(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
