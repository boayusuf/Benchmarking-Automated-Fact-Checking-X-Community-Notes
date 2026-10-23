"""
Step 7: Merge step 4/5/6 outputs + the extracted_claims source into one master CSV.

- Reads:
  - data/filtered/extracted_claims.csv (claim metadata: noteId, tweetId, summary, category, cn_verdict, extracted_claim)
  - data/results/afc_gpt4o_verdicts.csv  (System A: gpt-4o no evidence)
  - data/results/afc_tavily_progress.csv (System B: gpt-4o + Tavily; some rows may be missing if Tavily quota wasn't filled)
  - data/results/afc_google_verdicts.csv (System C: Google Fact Check)
- Writes:
  - data/results/master_results.csv

Left-joins on noteId so claims missing from any AFC system show up as NaN in
that system's columns; the metrics step (08) filters per-system as needed.
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")


def main() -> int:
    filtered = PROJECT_ROOT / "data" / "filtered"
    results = PROJECT_ROOT / "data" / "results"

    base = pd.read_csv(filtered / "extracted_claims.csv", dtype={"noteId": str, "tweetId": str})
    print(f"  base extracted_claims: {len(base)} rows", flush=True)

    a = pd.read_csv(results / "afc_gpt4o_verdicts.csv", dtype={"noteId": str})
    a = a[["noteId", "gpt4o_verdict", "gpt4o_confidence", "gpt4o_reasoning"]]
    print(f"  System A (gpt-4o):        {len(a)} verdicts", flush=True)

    # Step 5 progress is the latest source of Tavily verdicts. Drop ERROR rows here
    # so master_results only carries evidence-backed Tavily verdicts.
    b_src = pd.read_csv(results / "afc_tavily_progress.csv", dtype={"noteId": str})
    b_ok = b_src[b_src["tavily_status"] == "OK"].copy()
    b = b_ok[[
        "noteId",
        "tavily_evidence_summary",
        "tavily_gpt4o_verdict",
        "tavily_gpt4o_confidence",
        "tavily_gpt4o_reasoning",
    ]]
    print(f"  System B (gpt-4o+Tavily): {len(b)} verdicts (filtered to tavily_status==OK)", flush=True)

    c = pd.read_csv(results / "afc_google_verdicts.csv", dtype={"noteId": str})
    c = c[["noteId", "google_verdict", "google_source", "google_raw_rating"]]
    print(f"  System C (Google FC):     {len(c)} verdicts", flush=True)

    # System D — filtered RAG (only OK rows go into master; ERROR rows have unreliable verdicts)
    d_path = results / "afc_filtered_rag_verdicts.csv"
    if d_path.exists():
        d_src = pd.read_csv(d_path, dtype={"noteId": str})
        d_ok = d_src[d_src["filtered_rag_status"] == "OK"].copy()
        d = d_ok[["noteId",
                  "filtered_rag_verdict", "filtered_rag_confidence", "filtered_rag_reasoning",
                  "filtered_rag_blocked_count", "filtered_rag_authoritative_count"]]
        print(f"  System D (filtered RAG):  {len(d)} verdicts (filtered to OK)", flush=True)
    else:
        d = pd.DataFrame(columns=["noteId"])
        print(f"  System D (filtered RAG):  not yet run", flush=True)

    # System E — Llama 3.3 70B via Groq (only present if step 04c completed)
    e_path = results / "afc_llama_verdicts.csv"
    if e_path.exists():
        e_src = pd.read_csv(e_path, dtype={"noteId": str})
        e = e_src[["noteId", "llama_verdict", "llama_confidence", "llama_reasoning"]]
        print(f"  System E (Llama 70B):     {len(e)} verdicts", flush=True)
    else:
        e = pd.DataFrame(columns=["noteId"])
        print(f"  System E (Llama 70B):     not yet run", flush=True)

    df = base.merge(a, on="noteId", how="left")
    df = df.merge(b, on="noteId", how="left")
    df = df.merge(c, on="noteId", how="left")
    df = df.merge(d, on="noteId", how="left")
    df = df.merge(e, on="noteId", how="left")

    cols = [
        "noteId", "tweetId", "category", "summary", "extracted_claim", "cn_verdict",
        "gpt4o_verdict", "gpt4o_confidence", "gpt4o_reasoning",
        "tavily_gpt4o_verdict", "tavily_gpt4o_confidence", "tavily_gpt4o_reasoning",
        "tavily_evidence_summary",
        "google_verdict", "google_source", "google_raw_rating",
        "filtered_rag_verdict", "filtered_rag_confidence", "filtered_rag_reasoning",
        "filtered_rag_blocked_count", "filtered_rag_authoritative_count",
        "llama_verdict", "llama_confidence", "llama_reasoning",
    ]
    cols = [c for c in cols if c in df.columns]
    out_path = results / "master_results.csv"
    df[cols].to_csv(out_path, index=False)

    print(f"\n  saved {len(df)} rows -> {out_path}", flush=True)
    print("\nCoverage by AFC system:", flush=True)
    coverage_cols = [
        ("A gpt-4o",          "gpt4o_verdict"),
        ("B gpt-4o+Tavily",   "tavily_gpt4o_verdict"),
        ("C Google FC",       "google_verdict"),
        ("D filtered RAG",    "filtered_rag_verdict"),
        ("E Llama 70B",       "llama_verdict"),
    ]
    for sys_name, col in coverage_cols:
        if col not in df.columns:
            print(f"  {sys_name:18}: not yet present", flush=True); continue
        n = df[col].notna().sum()
        print(f"  {sys_name:18}: {n}/{len(df)} ({n/len(df)*100:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
