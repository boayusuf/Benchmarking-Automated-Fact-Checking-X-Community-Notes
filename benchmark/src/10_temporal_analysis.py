"""
Step 10: Temporal analysis — split master_results by claim creation date and
compare per-system agreement rates before vs after GPT-4o's knowledge cutoff.

GPT-4o cutoff: 2023-10-01 (per OpenAI documentation; current gpt-4o snapshots
all have training data through that date).

Joins createdAtMillis_x from misleading_notes.csv / not_misleading_notes.csv
onto master_results.csv (those are the original filtered note records).

Outputs:
  data/results/temporal_summary.csv
  charts/temporal_analysis.png
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
FILTERED = PROJECT_ROOT / "data" / "filtered"
CHARTS = PROJECT_ROOT / "charts"

CUTOFF_DATE = pd.Timestamp("2023-10-01", tz="UTC")  # GPT-4o knowledge cutoff

SYSTEM_COLORS = {
    "A_gpt4o":         "#4285F4",
    "B_gpt4o_tavily":  "#34A853",
    "C_google_fc":     "#EA4335",
}
SYSTEM_LABELS = {
    "A_gpt4o":        "A: GPT-4o (no evidence)",
    "B_gpt4o_tavily": "B: GPT-4o + Tavily",
    "C_google_fc":    "C: Google Fact Check",
}
SYSTEM_VERDICT_COLS = {
    "A_gpt4o":        "gpt4o_verdict",
    "B_gpt4o_tavily": "tavily_gpt4o_verdict",
    "C_google_fc":    "google_verdict",
}


def map_verdict(v) -> int | None:
    if pd.isna(v): return None
    v = str(v).strip().upper()
    if v == "TRUE":  return 1
    if v == "FALSE": return 0
    return None  # MIXED, NO_RESULT, UNCLEAR, ERROR all dropped (strict)


def map_truth(c) -> int:
    return 1 if c == "NOT_MISLEADING" else 0


def main() -> int:
    master = pd.read_csv(RESULTS / "master_results.csv", dtype={"noteId": str})
    print(f"  master rows: {len(master)}", flush=True)

    # Pull createdAtMillis from the original filtered files
    mis  = pd.read_csv(FILTERED / "misleading_notes.csv", dtype={"noteId": str}, low_memory=False)
    nmis = pd.read_csv(FILTERED / "not_misleading_notes.csv", dtype={"noteId": str})
    notes = pd.concat([mis[["noteId", "createdAtMillis_x"]], nmis[["noteId", "createdAtMillis_x"]]])
    notes = notes.drop_duplicates(subset=["noteId"])

    df = master.merge(notes, on="noteId", how="left")
    n_missing = df["createdAtMillis_x"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} rows missing createdAtMillis_x", flush=True)
    df["created_at"] = pd.to_datetime(df["createdAtMillis_x"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])
    df["era"] = np.where(df["created_at"] < CUTOFF_DATE, "pre-cutoff", "post-cutoff")
    print(f"  date range: {df['created_at'].min().date()} -> {df['created_at'].max().date()}", flush=True)
    print(f"  pre-cutoff (<2023-10):  {(df['era']=='pre-cutoff').sum()}", flush=True)
    print(f"  post-cutoff (>=2023-10): {(df['era']=='post-cutoff').sum()}", flush=True)

    rows = []
    for system, col in SYSTEM_VERDICT_COLS.items():
        for era in ("pre-cutoff", "post-cutoff"):
            sub = df[df["era"] == era].copy()
            sub["y_pred"] = sub[col].apply(map_verdict)
            sub = sub.dropna(subset=["y_pred"])
            sub["y_true"] = sub["cn_verdict"].apply(map_truth)
            n = len(sub)
            if n == 0:
                rows.append({"system": system, "era": era, "n": 0, "agreement": np.nan})
                continue
            agreement = float((sub["y_true"].astype(int) == sub["y_pred"].astype(int)).mean())
            rows.append({"system": system, "era": era, "n": n, "agreement": agreement})

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "temporal_summary.csv", index=False)
    print("\nTemporal agreement (strict; MIXED / NO_RESULT dropped):")
    print(summary.to_string(index=False))

    # Chart: grouped bars system × era
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    eras = ["pre-cutoff", "post-cutoff"]
    systems = list(SYSTEM_VERDICT_COLS.keys())
    x = np.arange(len(eras))
    width = 0.25
    for i, sys_ in enumerate(systems):
        vals = [summary[(summary.system == sys_) & (summary.era == era)]["agreement"].iloc[0] for era in eras]
        ns = [summary[(summary.system == sys_) & (summary.era == era)]["n"].iloc[0] for era in eras]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=SYSTEM_LABELS[sys_], color=SYSTEM_COLORS[sys_])
        for j, b in enumerate(bars):
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.2f}\n(n={ns[j]})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{e}\n({CUTOFF_DATE.date()})" for e in eras])
    ax.set_ylabel("Agreement rate (strict)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-system agreement rate, pre vs post GPT-4o cutoff (2023-10-01)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = CHARTS / "temporal_analysis.png"
    plt.savefig(out, dpi=140); plt.close()
    print(f"\n  saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
