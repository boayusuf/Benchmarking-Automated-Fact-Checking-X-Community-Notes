"""
Step 5: AFC System B — Tavily evidence retrieval + GPT-4o verdicts (RAG).

For each extracted claim:
  1. Search Tavily (rotating between two API keys) for top-5 web snippets.
  2. Send claim + evidence to GPT-4o for a TRUE/FALSE/MIXED verdict in JSON.

- Reads `data/filtered/extracted_claims.csv` (1226 claims).
- Tavily key rotation: round-robin per call; on 429 the retry uses the other key.
- Checkpoints every CHECKPOINT_EVERY rows; reruns resume.
- Final output: `data/results/afc_tavily_verdicts.csv` with columns
  noteId, extracted_claim, cn_verdict, category,
  tavily_evidence_summary, tavily_gpt4o_verdict, tavily_gpt4o_confidence,
  tavily_gpt4o_reasoning.
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
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY missing", file=sys.stderr); raise SystemExit(1)

TAVILY_KEYS = [k for k in (
    os.getenv("TAVILY_API_KEY_1"),
    os.getenv("TAVILY_API_KEY_2"),
    os.getenv("TAVILY_API_KEY_3"),
) if k]
if not TAVILY_KEYS:
    print("ERROR: no TAVILY_API_KEY_1 / _2 in .env", file=sys.stderr); raise SystemExit(1)
print(f"  loaded {len(TAVILY_KEYS)} Tavily key(s)", flush=True)

client = OpenAI(api_key=OPENAI_KEY)

MODEL_NAME = "gpt-4o"
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 5
TAVILY_TIMEOUT = 30
SLEEP_BETWEEN = 0.3
CHECKPOINT_EVERY = 25
QUOTA_BAIL_THRESHOLD = 10
MAX_OUTPUT_TOKENS = 400
EVIDENCE_CHAR_CAP = 4000   # truncate combined evidence to keep prompt cost bounded
VALID_VERDICTS = ("TRUE", "FALSE", "MIXED")

PROMPT = """You are a fact-checking system. Evaluate the following claim using the provided evidence.

Claim: "{claim}"

Evidence from web sources:
{evidence}

Respond in this exact JSON format only, no other text:
{{"verdict": "TRUE" or "FALSE" or "MIXED", "confidence": 0.0 to 1.0, "reasoning": "2-3 sentence explanation based on the evidence"}}

Rules:
- TRUE: The evidence supports the claim as factually accurate
- FALSE: The evidence contradicts the claim or shows it is misleading
- MIXED: Evidence is conflicting or claim is partially true
- If evidence is insufficient, note this in reasoning and lower confidence
"""

OPENAI_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
TAVILY_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_tavily_idx = 0


def tavily_search(query: str, max_attempts: int = 5) -> tuple[str, str]:
    """Returns (combined_evidence, status). status in {'OK', 'EMPTY', 'ERROR'}."""
    global _tavily_idx
    for attempt in range(max_attempts):
        key_num = _tavily_idx % len(TAVILY_KEYS)
        key = TAVILY_KEYS[key_num]
        _tavily_idx += 1
        try:
            r = requests.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"query": query, "max_results": TAVILY_MAX_RESULTS, "search_depth": "basic"},
                timeout=TAVILY_TIMEOUT,
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                if not results:
                    return "", "EMPTY"
                texts = [str(res.get("content", "")).strip() for res in results]
                texts = [t for t in texts if t]
                combined = "\n".join(texts)
                if len(combined) > EVIDENCE_CHAR_CAP:
                    combined = combined[:EVIDENCE_CHAR_CAP] + "..."
                return combined, "OK"
            if r.status_code in TAVILY_RETRYABLE_STATUS:
                wait = (2 ** attempt) * 5
                print(f"\n  tavily HTTP {r.status_code} (key#{key_num + 1}); retry {attempt + 1}/{max_attempts} after {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"\n  tavily HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return "", "ERROR"
        except (requests.Timeout, requests.ConnectionError) as e:
            wait = (2 ** attempt) * 5
            print(f"\n  tavily {type(e).__name__} (key#{key_num + 1}); retry {attempt + 1}/{max_attempts} after {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  tavily non-retryable ({type(e).__name__}): {e}", flush=True)
            return "", "ERROR"
    return "", "ERROR"


def fact_check_with_retry(claim: str, evidence: str, max_attempts: int = 5) -> dict:
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": PROMPT.format(claim=claim, evidence=evidence)}],
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
        except OPENAI_RETRYABLE as e:
            wait = (2 ** attempt) * 5
            print(f"\n  openai retry {attempt + 1}/{max_attempts} after {wait}s ({type(e).__name__})", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  openai non-retryable ({type(e).__name__}): {e}", flush=True)
            return {"verdict": "ERROR", "confidence": 0.0, "reasoning": str(e)[:200]}
    return {"verdict": "ERROR", "confidence": 0.0, "reasoning": "exhausted retries"}


def main() -> int:
    src_path = PROJECT_ROOT / "data" / "filtered" / "extracted_claims.csv"
    out_path = PROJECT_ROOT / "data" / "results" / "afc_tavily_verdicts.csv"
    progress_path = PROJECT_ROOT / "data" / "results" / "afc_tavily_progress.csv"

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

    pbar = tqdm(total=len(df), desc="rag-checking", initial=len(results), file=sys.stdout)
    n_since_save = 0
    consecutive_failures = 0
    quota_exhausted = False

    for _, row in df.iterrows():
        nid = str(row["noteId"])
        if nid in done_ids:
            continue

        claim = str(row["extracted_claim"])
        evidence, ev_status = tavily_search(claim)
        if not evidence:
            evidence = "(no evidence retrieved from web search)"

        result = fact_check_with_retry(claim, evidence)

        is_failure = result["verdict"] == "ERROR" or ev_status == "ERROR"
        consecutive_failures = consecutive_failures + 1 if is_failure else 0

        evidence_summary = evidence if len(evidence) <= 1500 else evidence[:1500] + "..."

        results.append({
            "noteId": nid,
            "extracted_claim": claim,
            "cn_verdict": row["cn_verdict"],
            "category": row["category"],
            "tavily_evidence_summary": evidence_summary,
            "tavily_status": ev_status,
            "tavily_gpt4o_verdict": result["verdict"],
            "tavily_gpt4o_confidence": result["confidence"],
            "tavily_gpt4o_reasoning": result["reasoning"],
        })
        done_ids.add(nid)
        n_since_save += 1
        pbar.update(1)

        if n_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            n_since_save = 0
            verdicts = pd.Series([r["tavily_gpt4o_verdict"] for r in results])
            counts = verdicts.value_counts().to_dict()
            print(f"\n  checkpoint: {len(results)} done | " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

        if consecutive_failures >= QUOTA_BAIL_THRESHOLD:
            print(f"\n  BAIL: {QUOTA_BAIL_THRESHOLD} consecutive failures. Saving and exiting.", flush=True)
            quota_exhausted = True
            break

        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(results).to_csv(progress_path, index=False)
    pbar.close()
    if quota_exhausted:
        raise SystemExit(2)

    final = pd.DataFrame(results)
    cols = ["noteId", "extracted_claim", "cn_verdict", "category",
            "tavily_evidence_summary", "tavily_status",
            "tavily_gpt4o_verdict", "tavily_gpt4o_confidence", "tavily_gpt4o_reasoning"]
    final[cols].to_csv(out_path, index=False)
    print(f"\n  saved {len(final)} rows -> {out_path}", flush=True)
    print("\nVerdict distribution:", flush=True)
    print(final.groupby(["cn_verdict", "tavily_gpt4o_verdict"]).size(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
