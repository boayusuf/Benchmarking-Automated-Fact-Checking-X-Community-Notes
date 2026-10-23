"""
Step 8: Compute agreement, Cohen's Kappa, precision/recall/F1 per AFC system,
overall and by category. Generate charts + a summary table.

Auto-detects which systems are present in master_results.csv and produces
metrics + charts for each. Re-runs cleanly when new systems are added (just
re-merge and re-run; this script picks up the new columns).
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score, precision_score, recall_score, f1_score, confusion_matrix,
)

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
RESULTS = PROJECT_ROOT / "data" / "results"
CHARTS = PROJECT_ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

# Master registry — each tuple: (key, label, verdict_col, color, conf_col_or_none)
ALL_SYSTEMS = [
    ("A_gpt4o",         "A: GPT-4o (no evidence)",   "gpt4o_verdict",        "#4285F4", "gpt4o_confidence"),
    ("B_gpt4o_tavily",  "B: GPT-4o + Tavily RAG",    "tavily_gpt4o_verdict", "#34A853", "tavily_gpt4o_confidence"),
    ("C_google_fc",     "C: Google Fact Check",      "google_verdict",       "#EA4335", None),
    ("D_filtered_rag",  "D: GPT-4o + filtered RAG",  "filtered_rag_verdict", "#9C27B0", "filtered_rag_confidence"),
    ("E_llama_groq",    "E: Llama 3.3 70B (Groq)",   "llama_verdict",        "#FB8C00", "llama_confidence"),
]
CATEGORY_COLORS = {"POLITICAL": "#1f77b4", "HEALTH": "#2ca02c", "SCIENTIFIC": "#ff7f0e"}
CATEGORIES = ("POLITICAL", "HEALTH", "SCIENTIFIC")


def map_verdict(v: str, strategy: str) -> int | None:
    if pd.isna(v): return None
    v = str(v).strip().upper()
    if v == "TRUE":  return 1
    if v == "FALSE": return 0
    if v == "MIXED": return 0 if strategy == "mixed_as_false" else None
    return None


def map_truth(cn: str) -> int:
    return 1 if str(cn) == "NOT_MISLEADING" else 0


def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"n": 0, "agreement": np.nan, "kappa": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan}
    agreement = float((y_true == y_pred).mean())
    kappa = cohen_kappa_score(y_true, y_pred) if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else np.nan
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"n": int(len(y_true)), "agreement": agreement,
            "kappa": float(kappa) if not np.isnan(kappa) else np.nan,
            "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def metrics_for(df, verdict_col, strategy):
    sub = df[["cn_verdict", verdict_col]].copy()
    sub["y_pred"] = sub[verdict_col].apply(lambda v: map_verdict(v, strategy))
    sub = sub.dropna(subset=["y_pred"])
    sub["y_true"] = sub["cn_verdict"].apply(map_truth)
    return compute_metrics(sub["y_true"].astype(int).values, sub["y_pred"].astype(int).values)


def main() -> int:
    df = pd.read_csv(RESULTS / "master_results.csv")
    print(f"  loaded master_results: {len(df)} rows", flush=True)

    # Detect available systems
    systems = [(k, lbl, col, color, conf) for (k, lbl, col, color, conf) in ALL_SYSTEMS if col in df.columns]
    print(f"  systems present: {[k for k, *_ in systems]}", flush=True)

    rows = []
    for (key, lbl, col, _, _) in systems:
        for strategy in ("strict", "mixed_as_false"):
            for cat in (None,) + CATEGORIES:
                slice_df = df if cat is None else df[df["category"] == cat]
                m = metrics_for(slice_df, col, strategy)
                rows.append({"system": key, "strategy": strategy, "category": cat or "ALL", **m})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "metrics_summary.csv", index=False)
    print(f"  saved metrics_summary.csv ({len(summary)} rows)", flush=True)
    print()
    print("ALL-category, strict (MIXED dropped):")
    print(summary[(summary.category == "ALL") & (summary.strategy == "strict")].round(3).to_string(index=False))
    print()

    # ---- Confusion matrices ----
    sns.set_style("whitegrid")
    for (key, lbl, col, _, _) in systems:
        sub = df[["cn_verdict", col]].copy()
        sub["y_pred"] = sub[col].apply(lambda v: map_verdict(v, "strict"))
        sub = sub.dropna(subset=["y_pred"])
        sub["y_true"] = sub["cn_verdict"].apply(map_truth)
        cm = confusion_matrix(sub["y_true"].astype(int), sub["y_pred"].astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["AFC FALSE", "AFC TRUE"],
                    yticklabels=["CN MISLEADING", "CN NOT_MISLEADING"])
        ax.set_title(f"Confusion matrix - {lbl}\n(strict; MIXED dropped, n={len(sub)})")
        ax.set_xlabel("AFC prediction"); ax.set_ylabel("CN baseline")
        plt.tight_layout()
        plt.savefig(CHARTS / f"cm_{key}.png", dpi=140); plt.close()
        print(f"  saved cm_{key}.png", flush=True)

    # ---- Grouped bars (system x category, strict) ----
    cat_summary = summary[(summary.strategy == "strict") & (summary.category != "ALL")].copy()
    sys_keys = [k for k, *_ in systems]
    color_by_key = {k: c for k, _, _, c, _ in systems}
    label_by_key = {k: l for k, l, _, _, _ in systems}

    def grouped_bar(metric, ylabel, fname):
        pivot = cat_summary.pivot(index="category", columns="system", values=metric).reindex(CATEGORIES)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(CATEGORIES))
        n_sys = len(sys_keys)
        width = 0.8 / max(n_sys, 1)
        offset_start = -(n_sys - 1) / 2 * width
        for i, key in enumerate(sys_keys):
            vals = [pivot.loc[c, key] if key in pivot.columns else np.nan for c in CATEGORIES]
            ax.bar(x + offset_start + i * width, vals, width, label=label_by_key[key], color=color_by_key[key])
        ax.set_xticks(x); ax.set_xticklabels(CATEGORIES)
        ax.set_ylabel(ylabel); ax.set_title(f"{ylabel} by category x system (strict)")
        ax.legend(loc="best", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(CHARTS / fname, dpi=140); plt.close()
        print(f"  saved {fname}", flush=True)

    grouped_bar("agreement", "Agreement rate", "agreement_by_category.png")
    grouped_bar("kappa",     "Cohen's Kappa",  "kappa_by_category.png")
    grouped_bar("f1",        "F1",             "f1_by_category.png")

    # ---- Confidence histogram for System A (agree vs disagree) ----
    if "gpt4o_verdict" in df.columns and "gpt4o_confidence" in df.columns:
        sub = df.dropna(subset=["gpt4o_verdict", "gpt4o_confidence"]).copy()
        sub["y_pred"] = sub["gpt4o_verdict"].apply(lambda v: map_verdict(v, "strict"))
        sub = sub.dropna(subset=["y_pred"])
        sub["y_true"] = sub["cn_verdict"].apply(map_truth)
        sub["agree"] = sub["y_pred"].astype(int) == sub["y_true"].astype(int)
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 1, 21)
        ax.hist(sub.loc[sub["agree"], "gpt4o_confidence"], bins=bins, alpha=0.6,
                label=f"Agree with CN (n={int(sub['agree'].sum())})", color="#2ca02c")
        ax.hist(sub.loc[~sub["agree"], "gpt4o_confidence"], bins=bins, alpha=0.6,
                label=f"Disagree with CN (n={int((~sub['agree']).sum())})", color="#d62728")
        ax.set_xlabel("GPT-4o confidence (System A)"); ax.set_ylabel("Count")
        ax.set_title("System A confidence: agree vs disagree with CN (strict)")
        ax.legend(); plt.tight_layout()
        plt.savefig(CHARTS / "confidence_agree_vs_disagree.png", dpi=140); plt.close()
        print("  saved confidence_agree_vs_disagree.png", flush=True)

    # ---- Google FC coverage by category ----
    if "google_verdict" in df.columns:
        cov = df.groupby("category", group_keys=False).apply(
            lambda g: (~g["google_verdict"].isin(["NO_RESULT", "UNCLEAR", "ERROR", np.nan])
                       & g["google_verdict"].notna()).mean(),
            include_groups=False,
        ).reindex(CATEGORIES)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(CATEGORIES, cov.values, color=[CATEGORY_COLORS[c] for c in CATEGORIES])
        ax.set_ylabel("Google FC coverage (graded verdicts / total)")
        ax.set_ylim(0, max(0.5, cov.max() * 1.2))
        ax.set_title("Google Fact Check coverage by category")
        for i, v in enumerate(cov.values):
            ax.text(i, v + 0.005, f"{v*100:.1f}%", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(CHARTS / "google_coverage_by_category.png", dpi=140); plt.close()
        print("  saved google_coverage_by_category.png", flush=True)

    # ---- Summary table PNG ----
    keep = summary[summary.strategy == "strict"].copy()
    for c in ("agreement", "kappa", "precision", "recall", "f1"):
        keep[c] = keep[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    keep = keep[["system", "category", "n", "agreement", "kappa", "precision", "recall", "f1"]]
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(keep) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=keep.values, colLabels=keep.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    plt.title("Metrics summary (strict; MIXED dropped)", pad=10)
    plt.tight_layout()
    plt.savefig(CHARTS / "metrics_summary_table.png", dpi=140, bbox_inches="tight"); plt.close()
    print("  saved metrics_summary_table.png", flush=True)

    print(f"\nDone. Charts in: {CHARTS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
