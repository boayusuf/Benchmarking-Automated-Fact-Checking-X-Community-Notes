# AFC × Community Notes

Artefact for the dissertation *Benchmarking Automated Fact-Checking Pipelines Against X Community Notes* (Yusuf Shakir, BSc (Hons) Computer Science, University of West London, May 2026).

This app runs a user-supplied claim through five Automated Fact-Checking (AFC) systems in parallel and displays their verdicts side-by-side. It is the live counterpart to the offline 5,469-claim benchmark reported in the dissertation.

## Systems

| ID | Paradigm | Backend |
|----|----------|---------|
| A  | Parametric LLM (closed-source) | GPT-4o via OpenAI |
| B  | Naive RAG | GPT-4o + Tavily search |
| C  | Professional fact-check API | Google Fact Check Tools |
| D  | Credibility-filtered RAG | GPT-4o + filtered Tavily |
| E  | Parametric LLM (open-source) | Llama 3.3 70B via Groq |

Each system returns a verdict (`TRUE`, `MISLEADING`, `MIXED`, or `NO_RESULT`), a short reasoning, and latency.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your four API keys
streamlit run app.py
```

The app launches at `http://localhost:8501`.

## API keys required

- `OPENAI_API_KEY` — https://platform.openai.com/api-keys
- `GROQ_API_KEY` — https://console.groq.com/keys
- `TAVILY_API_KEY` — https://app.tavily.com/home
- `GOOGLE_FACT_CHECK_API_KEY` — https://console.cloud.google.com/apis/credentials (enable Fact Check Tools API)

Missing a key? The corresponding system shows as unavailable; the others still run.

## Rate limit

The app caps each browser session at 20 claim runs to prevent runaway API costs. Refresh the page to reset.

---

# Offline benchmark (`benchmark/`)

Everything needed to inspect or reproduce the results reported in the dissertation: the pipeline scripts, the per-system verdicts for all 5,469 claims, the computed metrics, the manually coded error analysis, and all charts.

## Layout

```
benchmark/
  src/                 pipeline scripts, run in numeric order
  data/filtered/       classified_claims.csv, extracted_claims.csv (incl. image-dependent flags)
  data/results/        master_results.csv (all systems, all claims) and every derived table
  charts/              all figures used in the dissertation
  requirements.txt
```

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `00_process_raw_cn.py` | Joins the raw X Community Notes `notes` and `noteStatusHistory` TSVs, keeps notes rated helpful / not helpful |
| 1 | `01_filter_dataset.py`, `01b_resample_not_misleading.py` | English, length and spam filters; MISLEADING = rated helpful, NOT_MISLEADING = rated not helpful |
| 2 | `02_classify_notes.py` | Topic classification (POLITICAL / HEALTH / SCIENTIFIC), gpt-4o-mini, temperature 0 |
| 3 | `03_extract_claims.py` | Claim extraction from the note text, gpt-4o-mini, temperature 0 |
| 4 | `04_afc_gpt4o.py` | System A: GPT-4o, parametric only |
| 4c | `04c_afc_llama.py` | System E: Llama 3.3 70B (`llama-3.3-70b-versatile`, Groq) |
| 5 | `05_afc_tavily_gpt4o.py` | System B: GPT-4o + Tavily (top 5 snippets) |
| 5b | `05b_afc_filtered_rag.py` | System D: GPT-4o + credibility-filtered Tavily (blocklist and allowlist defined in the script) |
| 6 | `06_afc_google.py` | System C: Google Fact Check Tools API |
| 7 | `07_merge_results.py` | Builds `master_results.csv` |
| 8 to 17 | `08_` ... `17_` | Metrics, McNemar tests, temporal and cross-system analysis, error analysis, image-dependent flagging (Claude Haiku 4.5) and the no-image sensitivity run |

`04b_afc_gemini.py` is an abandoned Gemini run kept for completeness; it is not part of the reported results.

## Reproducing the numbers

The metric scripts (steps 8 to 17) need no API keys and run directly on `data/results/master_results.csv`:

```bash
cd benchmark
pip install -r requirements.txt
python src/08_calculate_metrics.py
python src/11_mcnemar.py D      # A vs D; also B or E
python src/12_cross_system_analysis.py
```

The scripts were written for the author's machine: set `PROJECT_ROOT` near the top of each script to the absolute path of this `benchmark/` folder before running.

Re-running steps 2 to 6 calls paid APIs (OpenAI, Groq, Tavily, Google, Anthropic) and needs the keys listed in `.env.example`. Hosted model aliases change over time, so re-runs will not reproduce the verdicts exactly; `master_results.csv` is the frozen record of the runs made in late April and early May 2026.

## Data

Source: X Community Notes public data, downloaded 12 April 2026 from communitynotes.x.com. The raw TSVs (about 1.9 GB) are not redistributed here; download them from X and run step 0. The derived files in `data/` contain note text and tweet IDs from that public release and remain subject to X's terms for the Community Notes data. `master_results.csv` also contains short web search snippets retrieved by Tavily for Systems B and D.

Column guide for `master_results.csv`: `cn_verdict` is the Community Notes label (MISLEADING / NOT_MISLEADING); each system has a `*_verdict` column (TRUE / FALSE / MIXED / NO_RESULT / UNCLEAR / ERROR), plus confidence and reasoning where applicable. Systems B and D were run only on the 1,282-claim original subset (rows where `tavily_gpt4o_verdict` or `filtered_rag_verdict` is non-empty).

## Citation

Shakir, Y. (2026) *Benchmarking Automated Fact-Checking Pipelines Against X Community Notes*. BSc (Hons) dissertation, University of West London.
