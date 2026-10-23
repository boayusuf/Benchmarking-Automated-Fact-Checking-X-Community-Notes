"""
Step 12: Cross-system pairwise comparisons + hardest/easiest claims + Venn diagrams.

Produces:
  data/results/cross_system_pairs.csv   — for each (X,Y) pair: verdict-agreement
                                          rate, joint-correctness rate
  data/results/hardest_claims.csv       — rows where ALL eligible systems
                                          gave strict verdicts AND all disagreed
                                          with CN (model-resistant claims)
  data/results/easiest_claims.csv       — rows where ALL eligible systems
                                          gave strict verdicts AND all agreed
                                          with CN
  charts/cross_system_pair_agreement.png  — heatmap of pairwise verdict agreement
  charts/cross_system_pair_correctness.png — heatmap of joint correctness
  charts/venn_A_D_E.png                  — 3-set Venn: claims each system gets right
  charts/venn_A_B_D.png                  — 3-set Venn: no-evidence vs RAG variants

Strict mapping: TRUE=1, FALSE=0, MIXED/NA/NO_RESULT/UNCLEAR/ERROR -> excluded.
"""

from itertools import combinations
from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
RESULTS = PROJECT_ROOT / "data" / "results"
CHARTS = PROJECT_ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

SYSTEMS = [
    ("A", "GPT-4o (no evidence)",     "gpt4o_verdict",        "#4285F4"),
    ("B", "GPT-4o + Tavily RAG",      "tavily_gpt4o_verdict", "#34A853"),
    ("C", "Google Fact Check",        "google_verdict",       "#EA4335"),
    ("D", "GPT-4o + filtered RAG",    "filtered_rag_verdict", "#9C27B0"),
    ("E", "Llama 3.3 70B",            "llama_verdict",        "#FB8C00"),
]
SYSTEM_BY_KEY = {k: (lbl, col) for k, lbl, col, _ in SYSTEMS}
SYSTEM_KEYS = [k for k, *_ in SYSTEMS]
PAIRS_OF_INTEREST = [("A", "B"), ("A", "D"), ("A", "E"), ("B", "D"), ("D", "E")]


def map_strict(v):
    if pd.isna(v): return None
    v = str(v).strip().upper()
    if v == "TRUE":  return 1
    if v == "FALSE": return 0
    return None


def cn_to_int(c) -> int:
    return 1 if c == "NOT_MISLEADING" else 0


def main() -> int:
    df = pd.read_csv(RESULTS / "master_results.csv", dtype={"noteId": str})
    print(f"  loaded master_results: {len(df)} rows", flush=True)

    df["y_true"] = df["cn_verdict"].apply(cn_to_int)
    for k, _, col, _ in SYSTEMS:
        df[f"pred_{k}"] = df[col].apply(map_strict)
        df[f"correct_{k}"] = (df[f"pred_{k}"] == df["y_true"])
        # Note: correct_K is False both when wrong and when pred is NA. We'll
        # filter on pred_K notna whenever joint-correctness matters.

    # ---------- Pairwise stats ----------
    rows = []
    for x, y in PAIRS_OF_INTEREST:
        sub = df.dropna(subset=[f"pred_{x}", f"pred_{y}"]).copy()
        n = len(sub)
        verdict_agree = (sub[f"pred_{x}"] == sub[f"pred_{y}"]).sum()
        both_correct = ((sub[f"pred_{x}"] == sub["y_true"]) & (sub[f"pred_{y}"] == sub["y_true"])).sum()
        both_wrong   = ((sub[f"pred_{x}"] != sub["y_true"]) & (sub[f"pred_{y}"] != sub["y_true"])).sum()
        x_only_correct = ((sub[f"pred_{x}"] == sub["y_true"]) & (sub[f"pred_{y}"] != sub["y_true"])).sum()
        y_only_correct = ((sub[f"pred_{x}"] != sub["y_true"]) & (sub[f"pred_{y}"] == sub["y_true"])).sum()
        rows.append({
            "pair": f"{x} vs {y}",
            "n_paired": n,
            "verdict_agree_n": int(verdict_agree),
            "verdict_agree_rate": float(verdict_agree) / n if n else np.nan,
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            f"{x}_only_correct": int(x_only_correct),
            f"{y}_only_correct": int(y_only_correct),
            "joint_correct_rate": float(both_correct) / n if n else np.nan,
        })
    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(RESULTS / "cross_system_pairs.csv", index=False)
    print(f"\n  saved cross_system_pairs.csv ({len(pairs_df)} pairs)", flush=True)
    print(pairs_df.to_string(index=False))

    # ---------- Heatmap: pairwise verdict-agreement matrix (all 5 systems) ----------
    n_sys = len(SYSTEM_KEYS)
    agree_matrix = np.full((n_sys, n_sys), np.nan)
    correct_matrix = np.full((n_sys, n_sys), np.nan)
    for i, x in enumerate(SYSTEM_KEYS):
        for j, y in enumerate(SYSTEM_KEYS):
            sub = df.dropna(subset=[f"pred_{x}", f"pred_{y}"])
            if len(sub) == 0: continue
            agree_matrix[i, j] = (sub[f"pred_{x}"] == sub[f"pred_{y}"]).mean()
            correct_matrix[i, j] = ((sub[f"pred_{x}"] == sub["y_true"]) & (sub[f"pred_{y}"] == sub["y_true"])).mean()

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(agree_matrix, annot=True, fmt=".3f", cmap="Blues", ax=ax,
                xticklabels=SYSTEM_KEYS, yticklabels=SYSTEM_KEYS, cbar_kws={"label": "Verdict-agreement rate"})
    ax.set_title("Pairwise verdict agreement (strict; rows where both systems gave TRUE/FALSE)")
    plt.tight_layout()
    plt.savefig(CHARTS / "cross_system_pair_agreement.png", dpi=140); plt.close()
    print("  saved cross_system_pair_agreement.png", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(correct_matrix, annot=True, fmt=".3f", cmap="Greens", ax=ax,
                xticklabels=SYSTEM_KEYS, yticklabels=SYSTEM_KEYS, cbar_kws={"label": "Joint-correctness rate"})
    ax.set_title("Pairwise joint correctness (both systems agree with CN)")
    plt.tight_layout()
    plt.savefig(CHARTS / "cross_system_pair_correctness.png", dpi=140); plt.close()
    print("  saved cross_system_pair_correctness.png", flush=True)

    # ---------- Hardest / easiest claims ----------
    # "All systems" here = all 5 gave a strict verdict on the row.
    elig = df.dropna(subset=[f"pred_{k}" for k in SYSTEM_KEYS]).copy()
    print(f"\n  rows where ALL 5 systems gave strict verdicts: {len(elig)}", flush=True)
    if len(elig):
        all_correct = elig[[f"correct_{k}" for k in SYSTEM_KEYS]].all(axis=1)
        all_wrong = elig[[f"correct_{k}" for k in SYSTEM_KEYS]].apply(lambda r: (~r).all(), axis=1)
        hardest = elig[all_wrong]
        easiest = elig[all_correct]
        print(f"  HARDEST (all 5 disagreed with CN): {len(hardest)}", flush=True)
        print(f"  EASIEST (all 5 agreed with CN):    {len(easiest)}", flush=True)

        export_cols = ["noteId", "tweetId", "category", "cn_verdict", "extracted_claim",
                       "gpt4o_verdict", "tavily_gpt4o_verdict", "google_verdict",
                       "filtered_rag_verdict", "llama_verdict"]
        export_cols = [c for c in export_cols if c in df.columns]
        hardest[export_cols].to_csv(RESULTS / "hardest_claims.csv", index=False)
        easiest[export_cols].to_csv(RESULTS / "easiest_claims.csv", index=False)
        print(f"  saved hardest_claims.csv and easiest_claims.csv", flush=True)

    # Also a 4-LLM cut (drop C since it has lots of NO_RESULT)
    llm_keys = ["A", "B", "D", "E"]
    elig4 = df.dropna(subset=[f"pred_{k}" for k in llm_keys]).copy()
    if len(elig4):
        all_correct4 = elig4[[f"correct_{k}" for k in llm_keys]].all(axis=1)
        all_wrong4 = elig4[[f"correct_{k}" for k in llm_keys]].apply(lambda r: (~r).all(), axis=1)
        print(f"\n  rows where all 4 LLM systems (A,B,D,E) gave strict verdicts: {len(elig4)}", flush=True)
        print(f"  HARDEST (all 4 LLMs disagreed with CN): {(all_wrong4).sum()}", flush=True)
        print(f"  EASIEST (all 4 LLMs agreed with CN):    {(all_correct4).sum()}", flush=True)
        elig4[all_wrong4].to_csv(RESULTS / "hardest_claims_4llm.csv", index=False)
        elig4[all_correct4].to_csv(RESULTS / "easiest_claims_4llm.csv", index=False)
        print("  saved hardest_claims_4llm.csv and easiest_claims_4llm.csv", flush=True)

    # ---------- Venn diagrams ----------
    # 3-set Venns of the "right-claim" sets per system. Restrict to rows where
    # all three target systems gave a strict verdict (so the universe is consistent).
    def venn_three(sys_keys, fname, title):
        sub = df.dropna(subset=[f"pred_{k}" for k in sys_keys])
        sets = {k: set(sub.loc[sub[f"correct_{k}"], "noteId"]) for k in sys_keys}
        fig, ax = plt.subplots(figsize=(7, 5.5))
        v = venn3(
            [sets[sys_keys[0]], sets[sys_keys[1]], sets[sys_keys[2]]],
            set_labels=[f"{k}: {SYSTEM_BY_KEY[k][0]}" for k in sys_keys],
            ax=ax,
        )
        ax.set_title(f"{title}\n(N rows where all 3 gave strict verdicts: {len(sub)})")
        plt.tight_layout()
        plt.savefig(CHARTS / fname, dpi=140); plt.close()
        print(f"  saved {fname} | sizes: " + ", ".join(f"{k}={len(sets[k])}" for k in sys_keys), flush=True)

    venn_three(["A", "D", "E"], "venn_A_D_E.png",
               "Claims correctly answered: A vs D vs E (right-claim sets)")
    venn_three(["A", "B", "D"], "venn_A_B_D.png",
               "Claims correctly answered: A vs B vs D (no-evidence vs RAG variants)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
