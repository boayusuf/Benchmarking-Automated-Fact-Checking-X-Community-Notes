"""
Step 4c (System E, replacing the abandoned Gemini run): AFC verdicts using
Llama 3.3 70B Versatile via Groq, no evidence retrieval.

Model note: Yusuf asked for "Llama 3 70B". Groq deprecated `llama3-70b-8192`;
its current production 70B Llama model is `llama-3.3-70b-versatile`. Same
70B-parameter Llama family.

Mirrors 04_afc_gpt4o.py:
  - Same prompt
  - Same JSON response schema (TRUE/FALSE/MIXED + confidence + reasoning)
  - temperature=0
  - JSON mode (response_format={"type": "json_object"})
  - Reads `data/filtered/extracted_claims.csv`
  - Writes `data/results/afc_llama_verdicts.csv`

Groq free tier: 30 RPM, 1000 RPD on llama-3.3-70b-versatile (as of 2026).
Script bails cleanly on quota exhaustion; re-run resumes from progress.
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
from groq import Groq, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
load_dotenv(PROJECT_ROOT / ".env")

# Use only the fresh key 10 to finish the last ~550 calls.
KEYS = [k for k in (
    os.getenv("GROQ_API_KEY_10"),
) if k]
if not KEYS:
    fallback = os.getenv("GROQ_API_KEY")
    if fallback:
        KEYS = [fallback]
if not KEYS:
    print("ERROR: no Groq keys in .env", file=sys.stderr)
    raise SystemExit(1)
print(f"  loaded {len(KEYS)} Groq key(s)", flush=True)

CLIENTS = [Groq(api_key=k) for k in KEYS]

MODEL_NAME = "llama-3.3-70b-versatile"
# 6 keys × 30 RPM = 180 RPM combined. Sleep 0.5s -> ~120 RPM with margin.
SLEEP_BETWEEN = 2.5   # single key: 30 RPM cap; aim ~24 RPM
CHECKPOINT_EVERY = 25
QUOTA_BAIL_THRESHOLD = 5
MAX_OUTPUT_TOKENS = 400
VALID_VERDICTS = ("TRUE", "FALSE", "MIXED")
_call_idx = 0

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
    global _call_idx
    for attempt in range(max_attempts):
        key_num = _call_idx % len(CLIENTS)
        client = CLIENTS[key_num]
        _call_idx += 1
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
                conf = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))
            reasoning = str(parsed.get("reasoning", "")).strip()
            return {"verdict": verdict, "confidence": conf, "reasoning": reasoning}
        except RETRYABLE as e:
            wait = (2 ** attempt) * 5
            print(f"\n  retry {attempt + 1}/{max_attempts} after {wait}s ({type(e).__name__}, key#{key_num + 1})", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  non-retryable error ({type(e).__name__}): {e}", flush=True)
            return {"verdict": "ERROR", "confidence": 0.0, "reasoning": str(e)[:200]}
    return {"verdict": "ERROR", "confidence": 0.0, "reasoning": "exhausted retries"}


def main() -> int:
    src_path = PROJECT_ROOT / "data" / "filtered" / "extracted_claims.csv"
    out_path = PROJECT_ROOT / "data" / "results" / "afc_llama_verdicts.csv"
    progress_path = PROJECT_ROOT / "data" / "results" / "afc_llama_progress.csv"

    df = pd.read_csv(src_path, dtype={"noteId": str, "tweetId": str})
    print(f"  loaded {len(df)} claims", flush=True)

    if progress_path.exists():
        done = pd.read_csv(progress_path, dtype={"noteId": str})
        done_ids = set(done["noteId"].astype(str))
        results = done.to_dict("records")
        print(f"  resuming from {len(done_ids)} already-checked", flush=True)
    else:
        done_ids = set()
        results = []

    pbar = tqdm(total=len(df), desc="llama-fc", initial=len(results), file=sys.stdout)
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
            "llama_verdict": result["verdict"],
            "llama_confidence": result["confidence"],
            "llama_reasoning": result["reasoning"],
        })
        done_ids.add(nid)
        n_since_save += 1
        pbar.update(1)

        if n_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            n_since_save = 0
            verdicts = pd.Series([r["llama_verdict"] for r in results])
            counts = verdicts.value_counts().to_dict()
            print(f"\n  checkpoint: {len(results)} done | "
                  + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

        if consecutive_failures >= QUOTA_BAIL_THRESHOLD:
            print(f"\n  BAIL: {QUOTA_BAIL_THRESHOLD} consecutive retry-exhausted calls — "
                  f"likely Groq RPD wall. Saving progress; re-run later to resume.", flush=True)
            quota_exhausted = True
            break

        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(results).to_csv(progress_path, index=False)
    pbar.close()
    if quota_exhausted:
        raise SystemExit(2)

    final = pd.DataFrame(results)
    cols = ["noteId", "extracted_claim", "cn_verdict", "category",
            "llama_verdict", "llama_confidence", "llama_reasoning"]
    final[cols].to_csv(out_path, index=False)
    print(f"\n  saved {len(final)} rows -> {out_path}", flush=True)
    print("\nVerdict distribution:", flush=True)
    print(final.groupby(["cn_verdict", "llama_verdict"]).size(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
