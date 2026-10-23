"""
Step 5b (System D): Source-filtered Tavily evidence retrieval + GPT-4o verdicts.

Same shape as 05_afc_tavily_gpt4o.py (System B), but:
  1. Tavily results from BLOCKED_DOMAINS are dropped entirely.
  2. Results from AUTHORITATIVE_DOMAINS are re-ranked to the top of the evidence
     string.
  3. Each evidence snippet is prefixed with [domain.com] so GPT-4o can weight
     credibility itself.
  4. Tavily max_results bumped 5 -> 8 to leave headroom after filtering.

Reads `data/filtered/extracted_claims.csv`, writes
`data/results/afc_filtered_rag_verdicts.csv`.
"""

from pathlib import Path
from urllib.parse import urlparse
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
    os.getenv("TAVILY_API_KEY_4"),
    os.getenv("TAVILY_API_KEY_5"),
) if k]
# Old keys 1-3 are exhausted; only the genuinely-new keys are used here.
# Note: KEY5 happens to be a duplicate of KEY3 (already exhausted), so effective rotation is 1 fresh key.
if not TAVILY_KEYS:
    print("ERROR: no Tavily keys", file=sys.stderr); raise SystemExit(1)
print(f"  loaded {len(TAVILY_KEYS)} Tavily key(s)", flush=True)

client = OpenAI(api_key=OPENAI_KEY)

MODEL_NAME = "gpt-4o"
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 8
TAVILY_TIMEOUT = 30
KEEP_TOP_N = 5            # after filter+boost, keep top N for the prompt
SLEEP_BETWEEN = 0.3
CHECKPOINT_EVERY = 25
QUOTA_BAIL_THRESHOLD = 10
MAX_OUTPUT_TOKENS = 400
EVIDENCE_CHAR_CAP = 4000
VALID_VERDICTS = ("TRUE", "FALSE", "MIXED")

# Misinformation / low-quality sources (conservative — only well-documented bad actors).
BLOCKED_DOMAINS = {
    # US conspiracy / misinformation
    "infowars.com",
    "naturalnews.com",
    "breitbart.com",
    "gatewaypundit.com",
    "thefederalist.com",
    "zerohedge.com",
    "newsmax.com",
    "oann.com",
    "worldnetdaily.com", "wnd.com",
    "americanthinker.com",
    "bigleaguepolitics.com",
    "thegatewaypundit.com",
    "westernjournal.com",
    # Anti-vax / health misinformation
    "mercola.com",
    "greenmedinfo.com",
    "childrenshealthdefense.org",
    # UK conspiracy
    "expose.uk", "expose-news.com",
    "davidicke.com",
    # State propaganda
    "rt.com",
    "sputniknews.com",
    "tass.com",
    "presstv.ir",
    "globaltimes.cn",
    # Climate denial
    "climatedepot.com",
    "wattsupwiththat.com",
}

# Authoritative sources (boosted to top of evidence list).
AUTHORITATIVE_DOMAINS = {
    # Wire services
    "reuters.com", "apnews.com", "ap.org", "afp.com",
    # Major public broadcasters / papers
    "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "wsj.com", "ft.com", "economist.com",
    "npr.org", "pbs.org",
    # Peer-reviewed / academic
    "nature.com", "science.org", "thelancet.com", "nejm.org",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
    "scholar.google.com",
    # Health / science gov
    "cdc.gov", "who.int", "nih.gov", "fda.gov",
    "ecdc.europa.eu", "ukhsa.gov.uk",
    # General gov (US/UK/EU/intergovernmental)
    "gov.uk", "ec.europa.eu", "europa.eu", "un.org",
    "whitehouse.gov", "state.gov", "justice.gov", "treasury.gov",
    "supremecourt.gov", "congress.gov",
    # Fact-checkers
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "reuters.com/fact-check", "apfactcheck.com",
    "leadstories.com", "factcheckni.org",
    # Reference (Wikipedia is mixed but generally directionally correct)
    "wikipedia.org",
}


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def is_authoritative(domain: str) -> bool:
    if domain in AUTHORITATIVE_DOMAINS:
        return True
    # gov./edu. heuristics
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return True
    if domain.endswith(".gov.uk") or domain.endswith(".ac.uk"):
        return True
    return False


def is_blocked(domain: str) -> bool:
    return domain in BLOCKED_DOMAINS


PROMPT = """You are a fact-checking system. Evaluate the following claim using the provided evidence.

Each evidence snippet is prefixed with its source domain in brackets — e.g. [reuters.com], [cdc.gov], or [some-blog.com]. Weight each snippet by source authority: reputable wire services, peer-reviewed journals, government health agencies, and major fact-checkers carry far more weight than unknown blogs or partisan outlets. If only low-credibility sources are available, treat that as weak evidence and lower your confidence.

Claim: "{claim}"

Evidence from web sources:
{evidence}

Respond in this exact JSON format only, no other text:
{{"verdict": "TRUE" or "FALSE" or "MIXED", "confidence": 0.0 to 1.0, "reasoning": "2-3 sentence explanation citing the strongest source(s)"}}

Rules:
- TRUE: High-credibility evidence supports the claim
- FALSE: High-credibility evidence contradicts the claim or shows it is misleading
- MIXED: Conflicting evidence across credible sources, or claim is partially true
- If evidence is insufficient or only from low-credibility sources, note this in reasoning and lower confidence
"""

OPENAI_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
TAVILY_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_tavily_idx = 0


def tavily_search(query: str, max_attempts: int = 5) -> tuple[str, list[str], int, int, str]:
    """
    Returns: (combined_evidence, sources_used, blocked_count, authoritative_count, status)
    status in {'OK', 'EMPTY', 'ERROR'}
    """
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
                    return "", [], 0, 0, "EMPTY"

                kept_auth, kept_other = [], []
                blocked = 0
                for res in results:
                    url = str(res.get("url", ""))
                    content = str(res.get("content", "")).strip()
                    if not content:
                        continue
                    domain = domain_of(url)
                    if is_blocked(domain):
                        blocked += 1
                        continue
                    snippet = f"[{domain or 'unknown'}] {content}"
                    if is_authoritative(domain):
                        kept_auth.append((domain, snippet))
                    else:
                        kept_other.append((domain, snippet))

                ordered = kept_auth + kept_other
                if not ordered:
                    return "", [], blocked, 0, "EMPTY"
                ordered = ordered[:KEEP_TOP_N]
                domains = [d for d, _ in ordered]
                combined = "\n\n".join(s for _, s in ordered)
                if len(combined) > EVIDENCE_CHAR_CAP:
                    combined = combined[:EVIDENCE_CHAR_CAP] + "..."
                return combined, domains, blocked, len(kept_auth), "OK"

            if r.status_code in TAVILY_RETRYABLE_STATUS:
                wait = (2 ** attempt) * 5
                print(f"\n  tavily HTTP {r.status_code} (key#{key_num + 1}); retry {attempt + 1}/{max_attempts} after {wait}s", flush=True)
                time.sleep(wait); continue
            print(f"\n  tavily HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return "", [], 0, 0, "ERROR"
        except (requests.Timeout, requests.ConnectionError) as e:
            wait = (2 ** attempt) * 5
            print(f"\n  tavily {type(e).__name__} (key#{key_num + 1}); retry {attempt + 1}/{max_attempts} after {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n  tavily non-retryable ({type(e).__name__}): {e}", flush=True)
            return "", [], 0, 0, "ERROR"
    return "", [], 0, 0, "ERROR"


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
    out_path = PROJECT_ROOT / "data" / "results" / "afc_filtered_rag_verdicts.csv"
    progress_path = PROJECT_ROOT / "data" / "results" / "afc_filtered_rag_progress.csv"

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

    pbar = tqdm(total=len(df), desc="filtered-rag", initial=len(results), file=sys.stdout)
    n_since_save = 0
    consecutive_failures = 0
    quota_exhausted = False

    for _, row in df.iterrows():
        nid = str(row["noteId"])
        if nid in done_ids:
            continue

        claim = str(row["extracted_claim"])
        evidence, sources, blocked, n_auth, ev_status = tavily_search(claim)
        if not evidence:
            evidence = "(no evidence retrieved from web search after filtering)"

        result = fact_check_with_retry(claim, evidence)
        is_failure = result["verdict"] == "ERROR" or ev_status == "ERROR"
        consecutive_failures = consecutive_failures + 1 if is_failure else 0

        ev_summary = evidence if len(evidence) <= 1500 else evidence[:1500] + "..."

        results.append({
            "noteId": nid,
            "extracted_claim": claim,
            "cn_verdict": row["cn_verdict"],
            "category": row["category"],
            "filtered_rag_evidence_summary": ev_summary,
            "filtered_rag_evidence_sources": "|".join(sources),
            "filtered_rag_blocked_count": blocked,
            "filtered_rag_authoritative_count": n_auth,
            "filtered_rag_status": ev_status,
            "filtered_rag_verdict": result["verdict"],
            "filtered_rag_confidence": result["confidence"],
            "filtered_rag_reasoning": result["reasoning"],
        })
        done_ids.add(nid)
        n_since_save += 1
        pbar.update(1)

        if n_since_save >= CHECKPOINT_EVERY:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            n_since_save = 0
            verdicts = pd.Series([r["filtered_rag_verdict"] for r in results])
            counts = verdicts.value_counts().to_dict()
            n_blocked_total = sum(r["filtered_rag_blocked_count"] for r in results)
            n_auth_total = sum(r["filtered_rag_authoritative_count"] for r in results)
            print(f"\n  checkpoint: {len(results)} | "
                  + " ".join(f"{k}={v}" for k, v in counts.items())
                  + f" | total_blocked={n_blocked_total} total_authoritative={n_auth_total}",
                  flush=True)

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
            "filtered_rag_evidence_summary", "filtered_rag_evidence_sources",
            "filtered_rag_blocked_count", "filtered_rag_authoritative_count",
            "filtered_rag_status",
            "filtered_rag_verdict", "filtered_rag_confidence", "filtered_rag_reasoning"]
    final[cols].to_csv(out_path, index=False)

    print(f"\n  saved {len(final)} rows -> {out_path}", flush=True)
    print(f"\n  Filter stats:", flush=True)
    print(f"    total blocked snippets:      {final['filtered_rag_blocked_count'].sum()}", flush=True)
    print(f"    total authoritative kept:    {final['filtered_rag_authoritative_count'].sum()}", flush=True)
    print(f"    rows with >=1 authoritative: {(final['filtered_rag_authoritative_count'] > 0).sum()} / {len(final)}", flush=True)
    print(f"    rows with >=1 blocked:       {(final['filtered_rag_blocked_count'] > 0).sum()} / {len(final)}", flush=True)
    print(f"\nVerdict distribution:", flush=True)
    print(final.groupby(["cn_verdict", "filtered_rag_verdict"]).size(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
