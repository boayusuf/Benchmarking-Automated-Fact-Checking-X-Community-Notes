"""
Step 15: Bar chart of manually-coded error categories from the 102 hand-coded
disagreement cases.

Reads `data/results/error_analysis_final.csv` (Yusuf's annotated file).
Produces:
  charts/error_categories.png            — overall counts per category
  charts/error_categories_by_topic.png   — stacked: error category x topic (POL/HEALTH/SCI)
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

PROJECT_ROOT = Path(r"C:\Users\yusuf\dissertation")
RESULTS = PROJECT_ROOT / "data" / "results"
CHARTS = PROJECT_ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

CATEGORY_COLORS = {"POLITICAL": "#1f77b4", "HEALTH": "#2ca02c", "SCIENTIFIC": "#ff7f0e"}


def main() -> int:
    df = pd.read_csv(RESULTS / "error_analysis_final.csv")
    df["error_category"] = df["error_category"].fillna("UNCODED").str.strip()
    n = len(df)
    print(f"  loaded {n} cases", flush=True)

    # ---- Overall horizontal bar chart ----
    counts = df["error_category"].value_counts()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(counts) + 1.5))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#4285F4", edgecolor="white")
    for b, v in zip(bars, counts.values[::-1]):
        pct = v / n * 100
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2,
                f"{v} ({pct:.1f}%)", va="center", fontsize=10)
    ax.set_xlabel(f"Cases (out of {n})")
    ax.set_title(f"Error categories on {n} CN-vs-AFC disagreements (manually coded)")
    ax.set_xlim(0, counts.max() * 1.18)
    plt.tight_layout()
    plt.savefig(CHARTS / "error_categories.png", dpi=140); plt.close()
    print("  saved error_categories.png", flush=True)

    # ---- Stacked: error category x topic (POLITICAL/HEALTH/SCIENTIFIC) ----
    if "category" in df.columns:
        pivot = pd.crosstab(df["error_category"], df["category"])
        # order categories by total descending
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        topic_order = [t for t in ("POLITICAL", "HEALTH", "SCIENTIFIC") if t in pivot.columns]

        fig, ax = plt.subplots(figsize=(11, 0.45 * len(pivot) + 2))
        bottom = np.zeros(len(pivot))
        y_pos = np.arange(len(pivot))[::-1]
        for topic in topic_order:
            vals = pivot[topic].values
            ax.barh(y_pos, vals, left=bottom, label=topic, color=CATEGORY_COLORS.get(topic, "gray"))
            for yi, x0, w in zip(y_pos, bottom, vals):
                if w > 0:
                    ax.text(x0 + w / 2, yi, str(int(w)), ha="center", va="center", fontsize=9, color="white")
            bottom += vals
        ax.set_yticks(y_pos); ax.set_yticklabels(pivot.index)
        ax.set_xlabel(f"Cases (out of {n})")
        ax.set_title(f"Error categories by topic (n={n})")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(CHARTS / "error_categories_by_topic.png", dpi=140); plt.close()
        print("  saved error_categories_by_topic.png", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
