# finbert-volatility-signals

**Does FinBERT sentiment on financial news predict next-day extreme stock
price moves (>1.5%)?**

A PySpark + FinBERT pipeline turns 2.4M NASDAQ news articles into a
daily sentiment signal for 15 stocks, joins it with Yahoo Finance prices,
and tests whether that signal helps an XGBoost classifier catch large
next-day price swings — first across all stocks, then narrowed down to
where the signal actually works.

## What was done

**Data pipeline (PySpark, on a Dataproc cluster).** The FNSPID dataset
(2,467,690 articles, 2009–2024, downloaded once from HuggingFace to GCS)
is cleaned, deduplicated on `(Date, Article_title, Stock_symbol)`, and
filtered to 15 liquid NASDAQ tickers chosen for having 8,000+ articles
each and spanning tech, finance, retail, energy, telecom, industrial and
international sectors: AAPL, GOOG, MSFT, NVDA, AMD, TSLA, INTC, BABA, GE,
DIS, WMT, GS, WFC, CVX, T. That leaves 129,340 articles, which are
inner-joined against daily closing prices from `yfinance` on
`(Date, Stock_symbol)` — an inner join because a price-less article can't
be scored, at the cost of dropping ~9% of articles on weekends/holidays —
giving 118,403 news-with-price rows.

**Sentiment labeling.** Each article's LSA extractive summary (not the
full text — FinBERT's 512-token limit made summaries necessary) is run
through `ProsusAI/finbert`. The saved checkpoint used for modelling has
110,759 labeled articles: 36.8% neutral, 33.6% positive, 29.5% negative,
with FinBERT most confident on negative calls (mean confidence 0.86 vs.
0.81–0.82 for the other two classes). The notebook also wires up VADER as
a rule-based baseline and prints a VADER-vs-FinBERT agreement rate — but
the checkpoint actually used in this run doesn't carry VADER labels, so
that comparison never ran; only the FinBERT distribution is real here.
One thing worth checking before reusing this dataset: **TSLA is present
through the price join (8,420 articles) but is absent from the final
110,759-row labeled checkpoint** — the sentiment-labeling step ends with
articles for only 14 of the 15 target stocks. Nothing in the notebook
explains the drop; it isn't a rounding effect (there's no TSLA row at
all in the per-stock output), so it looks like something silently
filtered TSLA out between the join and the FinBERT run.

**Modelling.** Articles are aggregated to 20,085 stock-day rows (5.5
articles/day on average), with momentum, rolling volatility, moving
averages, RSI and lagged/shifted sentiment features added — lagged and
shifted specifically to keep the target's own future information out of
the predictors. After dropping rows with insufficient history for the
rolling windows, 19,791 stock-days remain, chronologically split 80/20
into train (2012-06-06 → 2021-10-25, 12,793 rows) and test
(2021-10-26 → 2024-01-08, 6,998 rows). Two targets are built:
next-day direction (`Close` up or down) and an extreme-move flag
(`|next-day return| > 1.5%`, true for 35.5% of stock-days).

Four models were compared on the test set:

| Model | Accuracy | Extreme recall | Sentiment importance |
|---|---|---|---|
| Baseline (always predict "quiet") | 59.3% | — | — |
| 1 — GradientBoosting, next-day direction | 49.6% | — | 29.1% |
| 2 — XGBoost, extreme move, all 14 stocks | 58.6% | 70.4% | 28.7% |
| 3 — XGBoost, high-volatility stocks only | 56.4% | 66.6% | 34.6% |
| 4 — XGBoost, high-volatility + high-news-volume days | 61.5% | **70.7%** | **38.8%** |

Next-day direction (Model 1) came out at 49.6% — below the 50% coin-flip
line, i.e. no signal; the notebook reads this as ordinary market
efficiency and moves on. Extreme-move detection is the more interesting
target, but its 59.3% "always quiet" baseline makes raw accuracy a
misleading headline number for Models 2–4: precision on the extreme
class stays in the 49–58% range throughout, so a majority of "extreme"
predictions are still false alarms. The metric that actually improves
with each restriction is recall on the extreme class (the fraction of
real >1.5% moves the model manages to catch) and the share of feature
importance sentiment carries: going from all 14 stocks (Model 2:
70.4% recall, sentiment 28.7% of importance) to the 7 most volatile
stocks (AMD, NVDA, BABA, INTC, DIS, GOOG, WFC — ranked by mean 10-day
realized volatility) on days where news volume spikes to 1.5x+ the
trailing 5-day average (Model 4, 9.8% of all stock-days, 1,940 rows)
pushes recall to 70.7% and sentiment's share of feature importance from
29% to 39% — `sentiment_lag1`, `sentiment_negative` and `sentiment_positive`
all land in the top 12 features, alongside `news_spike`. The read: FinBERT
sentiment isn't a strong general-purpose signal, but filtering to the
regime where news actually clusters — volatile stocks having an unusually
newsy day — is where it earns its keep.

## Layout

```
Code with outputs.html   the notebook, exported with all cell outputs — both
                          the PySpark data pipeline and the XGBoost modelling
                          are here as one file (see "Reproducing" below)
requirements.txt          Python packages the notebook imports/pip-installs
                          (added in this pass — the repo shipped without one)
.gitignore                 keeps checkpoints, GCS-scale data and Jupyter/OS
                          junk out of git (added in this pass — see below)
```

There is no `.ipynb` in the repo, only its HTML export — the raw code
cells can be read out of the HTML (each is a `jp-CodeCell`), but there is
currently no way to open and re-run this as a live notebook without first
reconstructing one from that export.

## Reproducing

This is not a "clone and run" pipeline — parts of it are hard-wired to
the author's own infrastructure:

- **Part 1 (data pipeline) needs a Spark cluster**, specifically one that
  hands the notebook a live `spark` `SparkSession` (e.g. a GCP Dataproc
  Jupyter kernel) — the notebook uses `spark` throughout but never
  constructs one itself.
- **All checkpoints and the raw dataset live in one GCS bucket**,
  hard-coded as `GCS_BUCKET = 'gs://assesment2-dc'`. Reusing this
  pipeline means pointing that constant at a bucket you control and
  re-running from scratch (raw download included — the FNSPID CSV alone
  is ~22GB) or building your own checkpoints at each of the four stages
  (`01_df_clean`, `02_df_stocks`, `03_df_joined`, `04_pd_news`).
- **`gcloud`/`gsutil` must be installed and authenticated** — checkpoint
  and dataset I/O shells out to them directly rather than using a Python
  GCS client.
- **Part 2 (modelling) only needs plain Python** once `04_pd_news.csv` is
  available locally:

  ```bash
  pip install -r requirements.txt
  gcloud storage cp gs://<your-bucket>/checkpoints/04_pd_news.csv /tmp/pd_news.csv
  ```

  then the modelling cell (feature engineering → Models 1–4 → comparison
  table → plots) runs as ordinary pandas/scikit-learn/XGBoost code.

No model, prediction, or results file is committed to this repo — Part 2
writes `model_summary.csv`, `feature_importance_model4.csv`,
`daily_clean_final.csv` and a plot to `gs://assesment2-dc/results/`, all
GCS-only.

## Repo hygiene notes

- **No large data files are committed** — `git log --stat` and a scan of
  every blob in the repo's history show only the two copies of the HTML
  export (`P1.html`, renamed to `Code with outputs.html`, ~1MB each);
  the 2.4M-article dataset and every checkpoint genuinely stay in GCS as
  the notebook intends. The `.gitignore` added in this pass is
  precautionary, not a fix for an existing leak.
- **No `.gitignore` or `requirements.txt` existed before this pass** —
  both are added above.
- **No stray junk** (`.DS_Store`, `.ipynb_checkpoints`, empty
  placeholders) was found in the repo.
- One code cell in Part 1 (Spark session setup) contains bare
  `--num-executors 2 --executor-cores 2 --executor-memory 2G
  --driver-memory 2G` as if it were a shell command, which raises a
  `SyntaxError` when the cell is actually run in Jupyter — visible
  in that cell's own saved output. It looks like Dataproc PySpark-kernel
  magic arguments that were pasted into a code cell instead of the
  kernel's config field; harmless to the rest of the run since every
  later cell re-checks its checkpoint before doing any work, but worth
  fixing or removing if the notebook is reconstructed as a runnable
  `.ipynb`.

## Context

Assignment 2 for a Distributive Computing course (the notebook's own
title cell: "Distributive Computing — Assignment 2, Part 1: Data
Pipeline (PySpark)"). No institution is named in the notebook itself.
