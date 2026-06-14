# tco_flood_newspapers

A [DSPy](https://dspy.ai/) pipeline for finding and structuring evidence of
historical **Ontario flood events** in OCR-extracted text from historical
Canadian newspapers.

Newspaper OCR is noisy, and most articles that mention "flood" are irrelevant
(metaphors, floods elsewhere, flood-light ads, etc.). This pipeline cleans the
OCR text, filters articles down to real floods that occurred in Ontario, and
extracts structured data from the ones that pass.

## How it works

`process_article(raw_text, title)` runs each article through four DSPy
predictors in three stages and returns a result dict.

| Stage | Signature | Purpose |
|-------|-----------|---------|
| 1. Clean | `OCRCorrection` ([src/ocr.py](src/ocr.py)) | Fix OCR errors (spelling, spacing, character substitutions) while preserving period-appropriate language. |
| 2a. Verify flood | `floodIdentification` ([src/signatures.py](src/signatures.py)) | Keep only articles that reference a *real* flood event; drop metaphorical, hypothetical, and non-water uses of "flood". |
| 2b. Verify Ontario | `isOntario` ([src/signatures.py](src/signatures.py)) | Keep only floods that *occurred in* Ontario (not just reported by an Ontario paper). |
| 3. Extract | `FloodExtraction` ([src/signatures.py](src/signatures.py)) | Pull structured fields: `date`, `location`, `intensity`. |

An article that fails either filter is rejected early and never reaches
extraction.

### Output

```python
# Accepted
{
    "status": "accepted",
    "date": "1954-03",
    "location": "Cambridge, Grand River",
    "intensity": "severe — homes flooded, several families displaced",
    "corrected_text": "...",
}

# Rejected (one of)
{"status": "rejected", "reason": "no real flood"}
{"status": "rejected", "reason": "not Ontario"}
```

## Setup

Requires Python 3.13 and a local [Ollama](https://ollama.com/) server, since the
pipeline is configured to use the `llama3.1:8b` model
([src/pipeline.py](src/pipeline.py)).

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start Ollama and pull the model (in a separate shell)
ollama serve
ollama pull llama3.1:8b
```

To use a different model or provider, edit the `dspy.LM(...)` line in
[src/pipeline.py](src/pipeline.py).

## Usage

The pipeline is a library — import and call `process_article`:

```python
from src.pipeline import process_article

result = process_article(
    raw_text="...raw OCR text of a newspaper article...",
    title="Grand River Overflows Its Banks",
)
print(result)
```

> Run from the project root, or run from inside `src/` (the modules import each
> other by bare name, e.g. `from signatures import ...`).

## Optimizing the filters

The two Stage 2 filters can be tuned against manually-labelled data using DSPy's
few-shot optimization ([src/optimize.py](src/optimize.py)).

### Labelled data format

The optimizer reads `data/raw/annotations_so_far.csv` (override with `--csv`).
Expected header (an optional unnamed leading index column, as written by
`pandas.to_csv`, is ignored):

```csv
,id,date,example,changed,flood,ontario
0,1289135103,1954-02-18,"Six Persons Die as Snow Floods Stall Traffic ...",True,True,True
1,2923320586,1868-04-17,"IT 1UB TKADE BETWEEN CANADA AND THE STATES ...",True,True,False
```

| Column | Meaning |
|--------|---------|
| `id` | Row identifier (free-form). |
| `date` | Article/flood date (reference only; not used for training). |
| `example` | An extract of a newspaper article — the text being classified. |
| `changed` | **Ignored** by the optimizer. |
| `flood` | `True` if the extract references a *real* flood event, else `False`. |
| `ontario` | `True` if that flood occurred in Ontario, else `False`. |

`flood` and `ontario` are binary (`True`/`False`, also accepting `1`/`0`,
`yes`/`no`). The bundled dataset has 250 rows (196 flood, of which 57 Ontario).

### Running optimization

```bash
cd src
python optimize.py                 # optimize both filters (data/raw/annotations_so_far.csv)
python optimize.py --filters flood # just the flood filter
python optimize.py --help          # all options (model, dev-frac, demos, seed, ...)
```

This loads the CSV, evaluates each filter's baseline accuracy on a held-out
split, compiles improved few-shot prompts with `BootstrapFewShot`, prints
before/after accuracy, and saves the compiled programs to
`artifacts/flood_filter.json` and `artifacts/ontario_filter.json`.

The pipeline **automatically loads these artifacts on import** if they exist, so
optimized filters are used by `process_article` with no code changes; delete the
files in `artifacts/` to revert to the un-optimized prompts.

Notes:
- Requires Ollama running (same model as the pipeline), since optimization makes
  live LM calls. Override with `--model` / `--api-base`.
- The Ontario filter is trained only on rows where `flood == 1`, mirroring the
  pipeline cascade where `isOntario` only sees articles that passed the flood
  filter.

## Overnight run on ProQuest TDM (`src/tdm_overnight.py`)

`src/tdm_overnight.py` is a **single self-contained script** built to run
unattended overnight inside ProQuest TDM Studio against the OpenAI-compatible
proxy. It does the whole job — optimize, evaluate, then label — in one pass,
favouring **recall over precision** (we would rather over-include than lose a
real Ontario flood).

**Phase A — optimize & pick a prompt (recall-first, budget-aware).** Reads
`annotations_so_far.csv`, derives the target `is_ontario_flood = flood AND
ontario`, and uses DSPy (`BootstrapFewShot`) to compile few-shot prompts.
Several candidate models are tried (default `gpt-4o-mini`, `gpt-5-nano`,
`gpt-5-mini`, `gpt-4.1-nano`) and each is scored on a held-out split with a
recall-first metric (a missed flood costs the most; over-inclusion is
tolerated). Crucially the winner must be **cheap enough to label the whole
corpus in one day's budget**: candidates whose projected cost for `--corpus-rows`
(91k) exceeds the cap are ranked last, so you get the highest-recall model that
still fits. It saves **evaluation metrics for the top 3 prompts** (recall,
precision, F1, **F2**, confusion matrix, $/1k rows, projected full-corpus cost,
fits-budget?) to `artifacts/prompt_eval_report.{md,json}` and the winner to
`artifacts/best_program.json`.

> On the real data this picks **`gpt-4o-mini`** (recall ≈ 0.94 at ~$0.31/1k rows
> ⇒ ~$28 for all 91k) over `gpt-4.1` (higher F2 but ~$4.3/1k ⇒ ~$385 for 91k —
> impossible in a day).

**Phase B — label the corpus.** Runs the winning prompt over the unlabelled
extracts (schema of `extracted_only_1.csv`), sized so the whole corpus fits the
daily dollar **budget** (default `$50`). For every row it emits exactly the
requested columns:

| Column | Meaning |
|--------|---------|
| `is_ontario_flood` | the decision (recall-first) |
| `decision` | `ontario_flood` / `flood_not_ontario` / `not_flood` |
| `reason` | a **specific** justification citing clues from that text |
| `flood_location` | where the flood was (filled when it's *not* Ontario) |
| `not_flood_reason` | why it isn't a flood (`metaphor`, `not_specific_event`, `artificial`, …) |
| `flood_type` | `river` / `lake` / `flash` / `ice_jam` / `dam_break` / … |

Output is checkpointed to JSONL (`--no-resume` to disable) and written to
`data/processed/ontario_flood_predictions.csv`. Spend is tracked in a
**persistent daily ledger** (`artifacts/usage_ledger.jsonl`) that is re-read on
every run, so the `$50/day` cap holds across restarts/crashes within a day; an
interrupted run resumes cleanly, and on restart the same day it **reuses the
already-optimized prompt** (no second optimization charge).

```bash
pip install dspy-ai tiktoken          # openai/litellm come with dspy

cd src
python tdm_overnight.py --self-test                 # offline mechanical check (no proxy/spend)
python tdm_overnight.py                              # real run: optimize + label all 91k within $50
python tdm_overnight.py --optimize-only             # just Phase A + the top-3 report
python tdm_overnight.py --reuse-best --skip-optimize  # relabel using a saved prompt
python tdm_overnight.py --reasoning-effort low      # give gpt-5/o-series more reasoning
python tdm_overnight.py --models gpt-4o-mini,gpt-5-mini,o4-mini  # custom model sweep
python tdm_overnight.py --help                      # all options (budget, corpus-rows, workers, …)
```

Operational details:
- **Network-restricted (TDM).** Sets `LITELLM_LOCAL_MODEL_COST_MAP` before
  importing LiteLLM so it never tries to fetch its remote cost map (the
  `[Errno 101] Network is unreachable` warning).
- **One call per row.** Forces a single `response_format={"type":"json_object"}`
  request (the mode the proxy supports), skipping DSPy's structured-output
  attempt that otherwise fails and retries — halving latency over 91k rows.
- **Auth/proxy.** Reads the token from `/home/ec2-user/SageMaker/.token/.agaitoken`
  and calls the proxy via DSPy/LiteLLM (`openai/<model>`).
- **gpt-5 / reasoning models.** `gpt-5*` and `o3/o4-mini` get `temperature=1`,
  `max_completion_tokens≥16000`, and `reasoning_effort=minimal` (keeps gpt-5
  fast and cheap); any model the account can't reach is probed and skipped
  rather than crashing the run.
- **Cost.** Uses the proxy's measured per-call cost when present, else TDM's
  `scripts/model_pricing.py` (with a sanity guard), else a built-in table;
  override exactly with `--price-in`/`--price-out`.

## Project layout

```
src/
  pipeline.py     # Orchestration: process_article() ties the stages together
  ocr.py          # Stage 1 signature: OCR correction
  signatures.py   # Stage 2 + 3 signatures: flood/Ontario filters and extraction
  optimize.py     # Compile better filter prompts from labelled data
  run_inference.py    # Batch the local (Ollama) pipeline over extracts
  tdm_overnight.py    # One-shot TDM job: optimize -> top-3 report -> budgeted labelling
data/
  raw/            # Input OCR text + annotations_so_far.csv (git-ignored)
  processed/      # Pipeline output (git-ignored)
artifacts/        # Compiled programs, prompt_eval_report.*, usage_ledger.jsonl (git-ignored)
requirements.txt
```
