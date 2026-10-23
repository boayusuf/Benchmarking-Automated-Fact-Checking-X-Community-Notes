"""
Step 17: Re-run step 8 metrics with image-dependent claims excluded.

Reads master_results.csv + extracted_claims.csv (which now has the
`image_dependent` column from step 16). Excludes flagged rows and recomputes
strict-mode metrics for systems present.

Saves:
  data/results/metrics_summary_no_image.csv
  charts/agreement_by_category_no_image.png
  charts/kappa_by_category_no_image.png
  charts/f1_by_category_no_image.png
  charts/metrics_summary_table_no_image.png
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
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
RESULTS = PROJECT_ROOT / "data" / "results"
CHARTS = PROJECT_ROOT / "charts"

ALL_SYSTEMS = [
    ("A_gpt4o",         "A: GPT-4o (no evidence)",   "gpt4o_verdict",        "#4285F4"),
    ("B_gpt4o_tavily",  "B: GPT-4o + Tavily RAG",    "tavily_gpt4o_verdict", "#34A853"),
    ("C_google_fc",     "C: Google Fact Check",      "google_verdict",       "#EA4335"),
    ("D_filtered_rag",  "D: GPT-4o + filtered RAG",  "filtered_rag_verdict", "#9C27B0"),
    ("E_llama_groq",    "E: Llama 3.3 70B (Groq)",   "llama_verdict",        "#FB8C00"),
]
CATEGORIES = ("POLITICAL", "HEALTH", "SCIENTIFIC")


def map_strict(v):
    if pd.isna(v): return None
    v = str(v).strip().upper()
    if v == "TRUE":  return 1
    if v == "FALSE": return 0
    return None


def map_truth(c) -> int:
    return 1 if c == "NOT_MISLEADING" else 0


def metrics_for(df, col):
    sub = df[["cn_verdict", col]].copy()
    sub["y_pred"] = sub[col].apply(map_strict)
    sub = sub.dropna(subset=["y_pred"])
    sub["y_true"] = sub["cn_verdict"].apply(map_truth)
    if len(sub) == 0:
        return {"n": 0, "agreement": np.nan, "kappa": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan}
    yt = sub["y_true"].astype(int).values
    yp = sub["y_pred"].astype(int).values
    return {
        "n": int(len(sub)),
        "agreement": float((yt == yp).mean()),
        "kappa": float(cohen_kappa_score(yt, yp)) if len(set(yt)) > 1 and len(set(yp)) > 1 else np.nan,
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall":    float(recall_score(yt, yp, zero_division=0)),
        "f1":        float(f1_score(yt, yp, zero_division=0)),
    }


def main() -> int:
    master = pd.read_csv(RESULTS / "master_results.csv", dtype={"noteId": str})
    extracted = pd.read_csv(PROJECT_ROOT / "data" / "filtered" / "extracted_claims.csv",
                            dtype={"noteId": str})
    if "image_dependent" not in extracted.columns:
        print("ERROR: extracted_claims.csv has no image_dependent column. Run step 16 first.", file=sys.stderr)
        return 1

    flag = extracted[["noteId", "image_dependent"]]
    df = master.merge(flag, on="noteId", how="left")
    n_img = (df["image_dependent"] == True).sum()
    n_kept = len(df) - n_img
    print(f"  master rows: {len(df)}", flush=True)
    print(f"  image-dependent rows excluded: {n_img}", flush=True)
    print(f"  rows kept for metrics:         {n_kept}", flush=True)
    df = df[df["image_dependent"] != True].copy()

    systems = [(k, lbl, col, color) for (k, lbl, col, color) in ALL_SYSTEMS if col in df.columns]

    rows = []
    for (key, _, col, _) in systems:
        for cat in (None,) + CATEGORIES:
            slice_df = df if cat is None else df[df["category"] == cat]
            m = metrics_for(slice_df, col)
            rows.append({"system": key, "category": cat or "ALL", **m})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "metrics_summary_no_image.csv", index=False)
    print(f"  saved metrics_summary_no_image.csv", flush=True)
    print()
    print("ALL-category, strict, image-dependent excluded:")
    print(summary[summary.category == "ALL"].round(3).to_string(index=False))

    # ---- Grouped bars ----
    sys.stdout.flush()
    sns.set_style("whitegrid")
    cat_summary = summary[summary.category != "ALL"].copy()
    sys_keys = [k for k, *_ in systems]
    color_by_key = {k: c for k, _, _, c in systems}
    label_by_key = {k: l for k, l, _, _ in systems}

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
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by category x system (image-dependent excluded; n_kept={n_kept})")
        ax.legend(loc="best", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(CHARTS / fname, dpi=140); plt.close()
        print(f"  saved {fname}", flush=True)

    grouped_bar("agreement", "Agreement rate", "agreement_by_category_no_image.png")
    grouped_bar("kappa",     "Cohen's Kappa",  "kappa_by_category_no_image.png")
    grouped_bar("f1",        "F1",             "f1_by_category_no_image.png")

    # ---- Summary table PNG ----
    keep = summary.copy()
    for c in ("agreement", "kappa", "precision", "recall", "f1"):
        keep[c] = keep[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(keep) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=keep.values, colLabels=keep.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    plt.title(f"Metrics summary (image-dependent excluded; n={n_kept})", pad=10)
    plt.tight_layout()
    plt.savefig(CHARTS / "metrics_summary_table_no_image.png", dpi=140, bbox_inches="tight"); plt.close()
    print("  saved metrics_summary_table_no_image.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
