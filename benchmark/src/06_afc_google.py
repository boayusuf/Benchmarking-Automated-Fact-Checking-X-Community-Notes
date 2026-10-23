"""
Step 6: AFC System C — Google Fact Check Tools API.

Free public API; no billing. Many claims will return NO_RESULT (publishers haven't
fact-checked them) — that's an expected finding.

- Reads `data/filtered/extracted_claims.csv`.
- Calls https://factchecktools.googleapis.com/v1alpha1/claims:search per claim.
- Maps the publisher's `textualRating` to {TRUE, FALSE, MIXED, UNCLEAR, NO_RESULT}.
- Final output: `data/results/afc_google_verdicts.csv`.
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
import requests
from tqdm import tqdm
from dotenv import load_dotenv

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_FACT_CHECK_API_KEY missing from .env", file=sys.stderr)
    raise SystemExit(1)

ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
SLEEP_BETWEEN = 0.5
CHECKPOINT_EVERY = 25
RETRY_STATUS = {429, 500, 502, 503, 504}

# Order matters — check MIXED before FALSE/TRUE (since "mostly true" contains "true").
MIXED_KEYWORDS = ("mostly true", "half true", "mixed", "partly", "partially")
FALSE_KEYWORDS = ("false", "pants on fire", "incorrect", "wrong", "fake", "misleading", "no", "not true")
TRUE_KEYWORDS = ("true", "correct", "accurate", "yes", "right")


def map_rating(rating: str) -> str:
    r = (rating or "").lower().strip()
    if not r:
        return "UNCLEAR"
    if any(k in r for k in MIXED_KEYWORDS):
        return "MIXED"
    if any(k in r for k in FALSE_KEYWORDS):
        return "FALSE"
    if any(k in r for k in TRUE_KEYWORDS):
        return "TRUE"
    return "UNCLEAR"


def search_with_retry(claim: str, max_attempts: int = 5) -> dict:
    for attempt in range(max_attempts):
        try:
            r = requests.get(
                ENDPOINT,
                params={"query": claim, "key": API_KEY, "languageCode": "en"},
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                claims = data.get("claims", [])
                if not claims:
                    return {"verdict": "NO_RESULT", "source": "", "rating": ""}
                cr = claims[0].get("claimReview", [{}])[0]
                rating = cr.get("textualRating", "")
                source = cr.get("publisher", {}).get("name", "Unknown")
                return {"verdict": map_rating(rating), "source": source, "rating": rating}
            if r.status_code in RETRY_STATUS:
                wait = (2 ** attempt) * 5
                print(f"\n  google HTTP {r.status_code}; retry {attempt + 1}/{max_attempts} after {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"\n  google HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return {"verdict": "ERROR", "source": "", "rating": f"http_{r.status_code}"}
        except (requests.Timeout, requests.ConnectionError) as e:
            wait = (2 ** attempt) * 5
            print(f"\n  google {type(e).__name__}; retry {attempt + 1}/{max_attempts} after {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  google non-retryable ({type(e).__name__}): {e}", flush=True)
            return {"verdict": "ERROR", "source": "", "rating": str(e)[:100]}
    return {"verdict": "ERROR", "source": "", "rating": "exhausted retries"}


def main() -> int:
    src_path = PROJECT_ROOT / "data" / "filtered" / "extracted_claims.csv"
    out_path = PROJECT_ROOT / "data" / "results" / "afc_google_verdicts.csv"
    progress_path = PROJECT_ROOT / "data" / "results" / "afc_google_progress.csv"

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

    pbar = tqdm(total=len(df), desc="google-fc", initial=len(results), file=sys.stdout)
    n_since_save = 0

    for _, row in df.iterrows():
        nid = str(row["noteId"])
        if nid in done_ids:
            continue

        out = search_with_retry(str(row["extracted_claim"]))

        results.append({
            "noteId": nid,
            "extracted_claim": row["extracted_claim"],
            "cn_verdict": row["cn_verdict"],
            "category": row["category"],
            "google_verdict": out["verdict"],
            "google_source": out["source"],
            "google_raw_rating": out["rating"],
        })
        done_ids.add(nid)
        n_since_save += 1
        pbar.update(1)

        if n_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            n_since_save = 0
            verdicts = pd.Series([r["google_verdict"] for r in results])
            counts = verdicts.value_counts().to_dict()
            print(f"\n  checkpoint: {len(results)} done | " + " ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

        time.sleep(SLEEP_BETWEEN)

    pd.DataFrame(results).to_csv(progress_path, index=False)
    pbar.close()

    final = pd.DataFrame(results)
    cols = ["noteId", "extracted_claim", "cn_verdict", "category",
            "google_verdict", "google_source", "google_raw_rating"]
    final[cols].to_csv(out_path, index=False)
    print(f"\n  saved {len(final)} rows -> {out_path}", flush=True)
    print("\nVerdict distribution:", flush=True)
    print(final["google_verdict"].value_counts(), flush=True)
    print("\nCoverage by cn_verdict:", flush=True)
    coverage = final.groupby("cn_verdict")["google_verdict"].apply(lambda s: (s != "NO_RESULT").mean())
    print(coverage, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
