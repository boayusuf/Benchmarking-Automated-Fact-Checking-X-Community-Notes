"""
Step 4b (System E): AFC verdicts using Gemini 2.5 Flash without evidence.

Mirrors 04_afc_gpt4o.py:
  - Same prompt
  - Same JSON response schema (TRUE/FALSE/MIXED + confidence + reasoning)
  - temperature=0
  - Reads `data/filtered/extracted_claims.csv`
  - Writes `data/results/afc_gemini_verdicts.csv`

Differences:
  - Uses google-generativeai SDK with `response_mime_type="application/json"`
  - Rotates between GEMINI_API_KEY_1 and GEMINI_API_KEY_2 per attempt
  - Lower QUOTA_BAIL_THRESHOLD (3) so we exit fast when Gemini RPD hits;
    re-running this script picks up where it left off (resume by noteId).
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
import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted, ServiceUnavailable, InternalServerError, DeadlineExceeded,
)

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
load_dotenv(PROJECT_ROOT / ".env")

KEYS = [k for k in (
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
) if k]
if not KEYS:
    fallback = os.getenv("GEMINI_API_KEY")
    if fallback:
        KEYS = [fallback]
if not KEYS:
    print("ERROR: no Gemini keys in .env", file=sys.stderr)
    raise SystemExit(1)
print(f"  loaded {len(KEYS)} Gemini key(s)", flush=True)

MODEL_NAME = "gemini-2.5-flash"
SLEEP_BETWEEN = 4.5          # ~13 RPM combined; safely under 10 RPM per key
CHECKPOINT_EVERY = 10
QUOTA_BAIL_THRESHOLD = 3     # exit fast when daily RPD wall hits
MAX_OUTPUT_TOKENS = 2048   # Gemini 2.5 Flash uses thinking tokens that count against this; needs headroom
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

RETRYABLE = (ResourceExhausted, ServiceUnavailable, InternalServerError, DeadlineExceeded)

_call_idx = 0


def _make_model(key: str):
    genai.configure(api_key=key)
    return genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )


def fact_check_with_retry(claim: str, max_attempts: int = 5) -> dict:
    """Round-robin keys per attempt; on 429 immediately retry with the other key."""
    global _call_idx
    for attempt in range(max_attempts):
        key_num = _call_idx % len(KEYS)
        key = KEYS[key_num]
        _call_idx += 1
        try:
            model = _make_model(key)
            resp = model.generate_content(PROMPT.format(claim=claim))
            # Be defensive: gemini-2.5-flash can return finish_reason=MAX_TOKENS
            # if the thinking budget exhausts the cap. .text raises in that case.
            try:
                text = (resp.text or "").strip()
            except Exception as e:
                fr = None
                try:
                    fr = resp.candidates[0].finish_reason
                except Exception:
                    pass
                return {"verdict": "INVALID", "confidence": 0.0,
                        "reasoning": f"empty/blocked response (finish_reason={fr})"}
            # JSON-mode should give a clean object, but be defensive
            if text.startswith("```"):
                # strip code fences if present
                text = text.strip("`")
                if text.startswith("json\n"):
                    text = text[5:]
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
    out_path = PROJECT_ROOT / "data" / "results" / "afc_gemini_verdicts.csv"
    progress_path = PROJECT_ROOT / "data" / "results" / "afc_gemini_progress.csv"

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

    pbar = tqdm(total=len(df), desc="gemini-fc", initial=len(results), file=sys.stdout)
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
            "gemini_verdict": result["verdict"],
            "gemini_confidence": result["confidence"],
            "gemini_reasoning": result["reasoning"],
        })
        done_ids.add(nid)
        n_since_save += 1
        pbar.update(1)

        if n_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            n_since_save = 0
            verdicts = pd.Series([r["gemini_verdict"] for r in results])
            counts = verdicts.value_counts().to_dict()
            print(f"\n  checkpoint: {len(results)} done | "
                  + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

        if consecutive_failures >= QUOTA_BAIL_THRESHOLD:
            print(f"\n  BAIL: {QUOTA_BAIL_THRESHOLD} consecutive retry-exhausted calls — "
                  f"likely Gemini RPD wall. Saving progress and exiting; re-run tomorrow to resume.",
                  flush=True)
            quota_exhausted = True
            break

        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(results).to_csv(progress_path, index=False)
    pbar.close()
    if quota_exhausted:
        raise SystemExit(2)

    final = pd.DataFrame(results)
    cols = ["noteId", "extracted_claim", "cn_verdict", "category",
            "gemini_verdict", "gemini_confidence", "gemini_reasoning"]
    final[cols].to_csv(out_path, index=False)
    print(f"\n  saved {len(final)} rows -> {out_path}", flush=True)
    print("\nVerdict distribution:", flush=True)
    print(final.groupby(["cn_verdict", "gemini_verdict"]).size(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
