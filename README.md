# Benchmarking Automated Fact-Checking Pipelines Against X Community Notes

Code, data and results for the paper *Benchmarking Automated Fact-Checking Pipelines Against X Community Notes* (Yusuf Shakir, 2026). The work began as the author's BSc (Hons) dissertation at the University of West London (May 2026).

Five automated fact-checking (AFC) configurations are evaluated against Community Notes verdicts on up to 5,469 English claims (political, health, scientific):

| ID | Configuration | Backend |
|----|---------------|---------|
| A  | Parametric LLM (closed) | GPT-4o |
| B  | Naive RAG | GPT-4o + top-5 Tavily snippets |
| C  | Professional fact-check API | Google Fact Check Tools |
| D  | Credibility-filtered RAG | GPT-4o + filtered Tavily |
| E  | Parametric LLM (open) | Llama 3.3 70B via Groq |

Main findings: agreement with the crowd is weak for every system (Cohen's κ ≤ 0.28); adding web snippets lowered GPT-4o's agreement on claims the crowd judged misleading (McNemar p = 0.004 naive, p = 0.043 filtered); Llama 3.3 70B is within two percentage points of GPT-4o (95% CI [−0.23, +1.97]); professional fact-checks gave a usable verdict for about 11% of claims.

## Repository layout

```
benchmark/
  src/                 pipeline scripts, run in numeric order
  data/filtered/       classified_claims.csv, extracted_claims.csv (incl. image-dependent flags)
  data/results/        master_results.csv (all systems, all claims), error_analysis_final.csv (102 coded cases), derived tables
  paper/               scripts that recompute every number and figure in the paper
  charts/              figures from the dissertation
  requirements.txt
app.py, systems/, utils/, scripts/   optional Streamlit demo (see bottom)
```

## Reproducing the paper

No API keys are needed. Everything is computed from the frozen verdicts in `benchmark/data/results/master_results.csv`.

```bash
pip install -r benchmark/requirements.txt
python benchmark/paper/compute_numbers.py   # every statistic in the paper -> benchmark/paper/numbers.json
python benchmark/paper/make_figures.py      # Figures 2 and 3 + kappa bootstrap CIs
```

`benchmark/paper/gmap.py` is the corrected mapping from Google Fact Check textual ratings to verdicts used in the paper (see its Appendix B). Where the paper and the dissertation differ, the paper's numbers are the correct ones.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `00_process_raw_cn.py` | Joins the raw Community Notes `notes` and `noteStatusHistory` TSVs, keeps notes rated helpful / not helpful |
| 1 | `01_filter_dataset.py`, `01b_resample_not_misleading.py` | English, length and spam filters; MISLEADING = rated helpful, NOT_MISLEADING = rated not helpful |
| 2 | `02_classify_notes.py` | Topic classification (POLITICAL / HEALTH / SCIENTIFIC), gpt-4o-mini, temperature 0 |
| 3 | `03_extract_claims.py` | Claim extraction from the note text, gpt-4o-mini, temperature 0 |
| 4 | `04_afc_gpt4o.py` | System A |
| 4c | `04c_afc_llama.py` | System E (`llama-3.3-70b-versatile`, Groq) |
| 5 | `05_afc_tavily_gpt4o.py` | System B |
| 5b | `05b_afc_filtered_rag.py` | System D (blocklist and allowlist defined in the script) |
| 6 | `06_afc_google.py` | System C |
| 7 | `07_merge_results.py` | Builds `master_results.csv` |
| 8 to 17 | `08_` ... `17_` | Dissertation metrics, McNemar tests, temporal and cross-system analysis, error analysis, image-dependent flagging and the no-image sensitivity run |

`04b_afc_gemini.py` is an abandoned Gemini run kept for completeness; it is not part of the reported results. Scripts 08 to 17 were written for the author's machine: set `PROJECT_ROOT` near the top of each before running.

Re-running steps 2 to 6 calls paid APIs and needs the keys in `.env.example`. Hosted model aliases change over time, so re-runs will not reproduce the verdicts exactly; `master_results.csv` is the frozen record of the runs made in late April and early May 2026.

## Data

Source: X Community Notes public data, downloaded 12 April 2026 from communitynotes.x.com. The raw TSVs (about 1.9 GB) are not redistributed; download them from X and run step 0. The derived files contain note text and post IDs from that public release and remain subject to X's terms. `master_results.csv` also contains short web snippets retrieved by Tavily for Systems B and D.

Column guide for `master_results.csv`: `cn_verdict` is the Community Notes label (MISLEADING / NOT_MISLEADING); each system has a `*_verdict` column (TRUE / FALSE / MIXED / NO_RESULT / UNCLEAR / ERROR), plus confidence and reasoning where applicable. Systems B and D were run only on the 1,282-claim original subset (rows where `tavily_gpt4o_verdict` or `filtered_rag_verdict` is non-empty).

## Optional: live demo

`app.py` is a Streamlit app that runs one claim through all five systems side by side. It is not needed to reproduce the paper.

```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENAI, GROQ, TAVILY and GOOGLE_FACT_CHECK keys
streamlit run app.py
```

Each browser session is capped at 20 claims to limit API costs.

## Citation

```bibtex
@misc{shakir2026afccn,
  title  = {Benchmarking Automated Fact-Checking Pipelines Against {X} {Community Notes}},
  author = {Shakir, Yusuf},
  year   = {2026},
  note   = {Preprint}
}
```
