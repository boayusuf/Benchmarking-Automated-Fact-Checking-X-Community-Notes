"""
Step 3: Extract the specific factual claim from each classified note using
OpenAI gpt-4o-mini.

- Reads `data/filtered/classified_claims.csv` (output of step 2).
- Different prompt for MISINFORMED_OR_POTENTIALLY_MISLEADING vs NOT_MISLEADING.
- Drops rows where the model returns "UNCLEAR".
- Saves intermediate progress every CHECKPOINT_EVERY rows so reruns resume.
- Final output: `data/filtered/extracted_claims.csv` with columns
  noteId, tweetId, summary, category, cn_verdict, extracted_claim.
"""

from pathlib import Path
import os
import sys
import time

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
SLEEP_BETWEEN = 0.2
CHECKPOINT_EVERY = 25
QUOTA_BAIL_THRESHOLD = 10
MAX_OUTPUT_TOKENS = 200

PROMPT_MISLEADING = """A Community Note is correcting a misleading post on social media (X/Twitter). The note explains why the original post is wrong.

Your task: extract the misleading claim — the specific factual statement made by the original post that the note is refuting. This is the claim a fact-checker would label as FALSE or MISLEADING.

Do NOT extract the correction or the truth from the note. Extract the misleading claim that the original post made.

Return ONLY the misleading claim as a single declarative sentence. If you cannot determine what the original post's misleading claim was from the note, respond with "UNCLEAR".

Community Note: "{summary}"
"""

PROMPT_NOT_MISLEADING = """A Community Note is confirming that a post on social media (X/Twitter) is accurate. The note explains why the original post's claim is correct.

Your task: extract the factual claim — the specific statement made by the original post that the note is affirming.

Return ONLY the claim as a single declarative sentence. If you cannot determine what the original post claimed from the note, respond with "UNCLEAR".

Community Note: "{summary}"
"""

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def extract_with_retry(summary: str, classification: str, max_attempts: int = 5) -> str:
    if classification == "MISINFORMED_OR_POTENTIALLY_MISLEADING":
        prompt = PROMPT_MISLEADING.format(summary=summary)
    else:
        prompt = PROMPT_NOT_MISLEADING.format(summary=summary)

    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            return (resp.choices[0].message.content or "").strip()
        except RETRYABLE as e:
            wait = (2 ** attempt) * 5
            print(f"\n  retry {attempt + 1}/{max_attempts} after {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  non-retryable error ({type(e).__name__}): {e}", flush=True)
            return "ERROR"
    return "ERROR"


def main() -> int:
    filtered = PROJECT_ROOT / "data" / "filtered"
    src_path = filtered / "classified_claims.csv"
    progress_path = filtered / "extracted_progress.csv"
    out_path = filtered / "extracted_claims.csv"

    df = pd.read_csv(src_path, dtype={"noteId": str, "tweetId": str})
    print(f"  loaded {len(df)} classified notes", flush=True)

    if progress_path.exists():
        done = pd.read_csv(progress_path, dtype={"noteId": str, "tweetId": str})
        done_ids = set(done["noteId"].astype(str))
        results = done.to_dict("records")
        print(f"  resuming from {len(done_ids)} already-extracted notes", flush=True)
    else:
        done_ids = set()
        results = []

    pbar = tqdm(total=len(df), desc="extracting", initial=len(results), file=sys.stdout)
    n_since_save = 0
    consecutive_failures = 0
    quota_exhausted = False

    for _, row in df.iterrows():
        nid = str(row["noteId"])
        if nid in done_ids:
            continue

        claim = extract_with_retry(str(row["summary"]), str(row["classification"]))
        consecutive_failures = consecutive_failures + 1 if claim == "ERROR" else 0

        cn_verdict = (
            "MISLEADING"
            if row["classification"] == "MISINFORMED_OR_POTENTIALLY_MISLEADING"
            else "NOT_MISLEADING"
        )
        results.append(
            {
                "noteId": nid,
                "tweetId": str(row.get("tweetId", "")),
                "summary": row["summary"],
                "category": row["category"],
                "cn_verdict": cn_verdict,
                "extracted_claim": claim,
            }
        )
        done_ids.add(nid)
        n_since_save += 1
        pbar.update(1)

        if n_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            n_since_save = 0
            n_unclear = sum(1 for r in results if r["extracted_claim"] == "UNCLEAR")
            n_error = sum(1 for r in results if r["extracted_claim"] == "ERROR")
            print(
                f"\n  checkpoint: {len(results)} done | UNCLEAR={n_unclear} ERROR={n_error}",
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

    extracted = pd.DataFrame(results)
    n_unclear = (extracted["extracted_claim"] == "UNCLEAR").sum()
    n_error = (extracted["extracted_claim"] == "ERROR").sum()
    final = extracted[~extracted["extracted_claim"].isin(["UNCLEAR", "ERROR"])].copy()

    cols = ["noteId", "tweetId", "summary", "category", "cn_verdict", "extracted_claim"]
    final[cols].to_csv(out_path, index=False)
    print(f"\n  extracted: {len(extracted)} | UNCLEAR: {n_unclear} | ERROR: {n_error}", flush=True)
    print(f"  saved {len(final)} rows -> {out_path}", flush=True)
    print("\nFinal distribution:", flush=True)
    print(final.groupby(["cn_verdict", "category"]).size(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
