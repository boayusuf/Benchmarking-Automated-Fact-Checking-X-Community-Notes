"""
Step 4: AFC System A — GPT-4o without evidence retrieval.

For each extracted claim, ask GPT-4o for a verdict (TRUE/FALSE/MIXED) plus
confidence and reasoning, using only the model's parametric knowledge.

- Reads `data/filtered/extracted_claims.csv` (1226 claims).
- Uses gpt-4o, temperature=0, JSON response_format for reliable parsing.
- Checkpoints every CHECKPOINT_EVERY rows; reruns resume.
- Final output: `data/results/afc_gpt4o_verdicts.csv` with columns
  noteId, extracted_claim, cn_verdict, gpt4o_verdict, gpt4o_confidence,
  gpt4o_reasoning.
"""

from pathlib import Path
import json
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

MODEL_NAME = "gpt-4o"
SLEEP_BETWEEN = 0.3
CHECKPOINT_EVERY = 25
QUOTA_BAIL_THRESHOLD = 10
MAX_OUTPUT_TOKENS = 300
VALID_VERDICTS = ("TRUE", "FALSE", "MIXED")

PROMPT = """You are a fact-checking system. Evaluate the following claim and determine its veracity based on your knowledge.

Claim: "{claim}"

Respond in this exact JSON format only, no other text:
{{"verdict": "TRUE" or "FALSE" or "MIXED", "confidence": 0.0 to 1.0, "reasoning": "2-3 sentence explanation"}}

Rules:
- TRUE: The claim is factually accurate
- FALSE: The claim is factually inaccurate or misleading
- MIXED: Contains both true and false elements, or is true but misleading without context
- Be honest about uncertainty in your confidence score
"""

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def fact_check_with_retry(claim: str, max_attempts: int = 5) -> dict:
    """Returns dict with verdict, confidence, reasoning. On error returns sentinel values."""
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": PROMPT.format(claim=claim)}],
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
            text = (resp.choices[0].message.content or "").strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {"verdict": "INVALID", "confidence": 0.0, "reasoning": f"unparseable: {text[:200]}"}

            verdict = str(parsed.get("verdict", "")).strip().upper()
            if verdict not in VALID_VERDICTS:
                return {"verdict": "INVALID", "confidence": 0.0, "reasoning": f"bad verdict: {verdict}"}

            try:
                confidence = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            reasoning = str(parsed.get("reasoning", "")).strip()

            return {"verdict": verdict, "confidence": confidence, "reasoning": reasoning}
        except RETRYABLE as e:
            wait = (2 ** attempt) * 5
            print(f"\n  retry {attempt + 1}/{max_attempts} after {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  non-retryable error ({type(e).__name__}): {e}", flush=True)
            return {"verdict": "ERROR", "confidence": 0.0, "reasoning": str(e)[:200]}
    return {"verdict": "ERROR", "confidence": 0.0, "reasoning": "exhausted retries"}


def main() -> int:
    src_path = PROJECT_ROOT / "data" / "filtered" / "extracted_claims.csv"
    out_path = PROJECT_ROOT / "data" / "results" / "afc_gpt4o_verdicts.csv"
    progress_path = PROJECT_ROOT / "data" / "results" / "afc_gpt4o_progress.csv"

    df = pd.read_csv(src_path, dtype={"noteId": str, "tweetId": str})
    print(f"  loaded {len(df)} claims to fact-check", flush=True)

    if progress_path.exists():
        done = pd.read_csv(progress_path, dtype={"noteId": str})
        done_ids = set(done["noteId"].astype(str))
        results = done.to_dict("records")
        print(f"  resuming from {len(done_ids)} already-checked claims", flush=True)
    else:
        done_ids = set()
        results = []

    pbar = tqdm(total=len(df), desc="fact-checking", initial=len(results), file=sys.stdout)
    n_since_save = 0
    consecutive_failures = 0
    quota_exhausted = False

    for _, row in df.iterrows():
        nid = str(row["noteId"])
        if nid in done_ids:
            continue

        result = fact_check_with_retry(str(row["extracted_claim"]))
        consecutive_failures = consecutive_failures + 1 if result["verdict"] == "ERROR" else 0

        results.append({
            "noteId": nid,
            "extracted_claim": row["extracted_claim"],
            "cn_verdict": row["cn_verdict"],
            "category": row["category"],
            "gpt4o_verdict": result["verdict"],
            "gpt4o_confidence": result["confidence"],
            "gpt4o_reasoning": result["reasoning"],
        })
        done_ids.add(nid)
        n_since_save += 1
        pbar.update(1)

        if n_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            n_since_save = 0
            verdicts = pd.Series([r["gpt4o_verdict"] for r in results])
            counts = verdicts.value_counts().to_dict()
            print(f"\n  checkpoint: {len(results)} done | " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

        if consecutive_failures >= QUOTA_BAIL_THRESHOLD:
            print(f"\n  BAIL: {QUOTA_BAIL_THRESHOLD} consecutive retry-exhausted calls. Saving and exiting.", flush=True)
            quota_exhausted = True
            break

        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(results).to_csv(progress_path, index=False)
    pbar.close()
    if quota_exhausted:
        raise SystemExit(2)

    final = pd.DataFrame(results)
    cols = ["noteId", "extracted_claim", "cn_verdict", "category",
            "gpt4o_verdict", "gpt4o_confidence", "gpt4o_reasoning"]
    final[cols].to_csv(out_path, index=False)
    print(f"\n  saved {len(final)} rows -> {out_path}", flush=True)
    print("\nVerdict distribution:", flush=True)
    print(final.groupby(["cn_verdict", "gpt4o_verdict"]).size(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
