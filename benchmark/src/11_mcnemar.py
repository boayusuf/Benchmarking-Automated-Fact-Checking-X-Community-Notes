"""
Step 11: McNemar's test — System A vs another system (paired comparison).

Default compares A vs D. Pass a CLI arg to compare against a different system:
  python 11_mcnemar.py D    # A vs D (default)
  python 11_mcnemar.py E    # A vs E (Llama)
  python 11_mcnemar.py B    # A vs B (Tavily RAG)

Implementation: scipy.stats.binomtest on the discordant pairs (n_01, n_10).
"""

from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
RESULTS = PROJECT_ROOT / "data" / "results"

OTHER_SYSTEM = {
    "B": ("System B (GPT-4o + Tavily RAG)",   "tavily_gpt4o_verdict"),
    "D": ("System D (GPT-4o + filtered RAG)", "filtered_rag_verdict"),
    "E": ("System E (Llama 3.3 70B / Groq)",  "llama_verdict"),
}


def map_strict(v) -> int | None:
    if pd.isna(v): return None
    v = str(v).strip().upper()
    if v == "TRUE":  return 1
    if v == "FALSE": return 0
    return None


def cn_to_int(c) -> int:
    return 1 if c == "NOT_MISLEADING" else 0


def main(other_key: str = "D") -> int:
    if other_key not in OTHER_SYSTEM:
        print(f"ERROR: unknown other system '{other_key}'. Choices: {list(OTHER_SYSTEM)}", file=sys.stderr)
        return 2
    other_label, other_col = OTHER_SYSTEM[other_key]

    master = pd.read_csv(RESULTS / "master_results.csv", dtype={"noteId": str})
    if other_col not in master.columns:
        print(f"ERROR: master_results.csv has no {other_col} column. Re-run step 7 first.", file=sys.stderr)
        return 1

    df = master.copy()
    df["y_true"] = df["cn_verdict"].apply(cn_to_int)
    df["pred_A"] = df["gpt4o_verdict"].apply(map_strict)
    df["pred_X"] = df[other_col].apply(map_strict)

    paired = df.dropna(subset=["pred_A", "pred_X"]).copy()
    paired["pred_A"] = paired["pred_A"].astype(int)
    paired["pred_X"] = paired["pred_X"].astype(int)
    paired["correct_A"] = paired["pred_A"] == paired["y_true"]
    paired["correct_X"] = paired["pred_X"] == paired["y_true"]
    print(f"  comparing System A vs {other_label}", flush=True)
    print(f"  paired rows (both A and {other_key} have strict verdicts): {len(paired)}", flush=True)

    # 2x2 contingency: (correct_A, correct_X)
    n_11 = int(((paired["correct_A"]) & (paired["correct_X"])).sum())   # both right
    n_10 = int(((paired["correct_A"]) & (~paired["correct_X"])).sum())  # A right, X wrong
    n_01 = int(((~paired["correct_A"]) & (paired["correct_X"])).sum())  # A wrong, X right
    n_00 = int(((~paired["correct_A"]) & (~paired["correct_X"])).sum()) # both wrong

    print(f"\n  Paired classification table (rows: A correctness, cols: {other_key} correctness):")
    print(f"                {other_key} correct   {other_key} wrong")
    print(f"    A correct   {n_11:7d}   {n_10:7d}")
    print(f"    A wrong     {n_01:7d}   {n_00:7d}")

    # McNemar's exact test (binomial) on discordant pairs
    n_disc = n_01 + n_10
    if n_disc == 0:
        print("\n  No discordant pairs — systems agree on every paired row. p = 1.0", flush=True)
        p_exact = 1.0
        chi2_stat = 0.0
        p_chi2 = 1.0
    else:
        result = binomtest(min(n_01, n_10), n_disc, p=0.5, alternative="two-sided")
        p_exact = result.pvalue
        # Chi-square approximation with continuity correction
        chi2_stat = (abs(n_01 - n_10) - 1) ** 2 / n_disc
        p_chi2 = 1 - chi2.cdf(chi2_stat, df=1)

    print(f"\n  Discordant pairs: n_01 (A wrong, {other_key} right) = {n_01}, "
          f"n_10 (A right, {other_key} wrong) = {n_10}")
    print(f"  McNemar exact (binomial) p-value: {p_exact:.6g}")
    print(f"  McNemar chi-square (cont.corrected): chi2 = {chi2_stat:.4f}, p = {p_chi2:.6g}")
    if p_exact < 0.05:
        if n_01 > n_10:
            verdict = f"System {other_key} is significantly BETTER than System A"
        else:
            verdict = f"System {other_key} is significantly WORSE than System A"
    else:
        verdict = f"No significant difference between System A and System {other_key}"
    print(f"\n  Conclusion (alpha=0.05): {verdict}")

    acc_A = paired["correct_A"].mean()
    acc_X = paired["correct_X"].mean()
    print(f"\n  Marginal accuracy on paired set: A = {acc_A:.4f}, "
          f"{other_key} = {acc_X:.4f} (delta = {acc_X - acc_A:+.4f})")

    new_row = {
        "comparison": f"A_vs_{other_key}",
        "n_paired": len(paired),
        "both_right": n_11,
        f"A_right_{other_key}_wrong": n_10,
        f"A_wrong_{other_key}_right": n_01,
        "both_wrong": n_00,
        "p_exact": p_exact,
        "chi2": chi2_stat,
        "p_chi2": p_chi2,
        "acc_A": acc_A,
        f"acc_{other_key}": acc_X,
        "verdict": verdict,
    }
    out_path = RESULTS / "mcnemar_summary.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[existing["comparison"] != f"A_vs_{other_key}"]
        summary = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        summary = pd.DataFrame([new_row])
    summary.to_csv(out_path, index=False)
    print(f"\n  saved/updated {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    other = sys.argv[1] if len(sys.argv) > 1 else "D"
    raise SystemExit(main(other))
