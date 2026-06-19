# Cluster step: local-LLM flood detail extraction (UofT CS SLURM)

This folder runs the **deep extraction** stage of the pipeline on the
University of Toronto CS research cluster, using a **local LLM served by
Ollama** on a GPU node — no external API and no spend.

## Where this fits

1. **TDM (upstream).** Inside ProQuest TDM Studio, `../src/tdm_overnight.py`
   labels the full ~91k-row corpus and filters it down to the rows that are
   real **Ontario floods**, writing `ontario_flood_predictions.csv`.
2. **Export.** You export those Ontario-flood rows out of TDM, joined back to
   their article text, into a single CSV (here called
   `ontario_floods_export.csv`).
3. **This folder (the cluster step).** For each Ontario flood, a local model
   extracts the finer structured detail the filter did not: **event date,
   flood type, location, water body, cause, intensity/impact (deaths,
   injuries, people displaced, damage estimate, infrastructure impact), and
   article date.**

The cluster does the heavy generation for free on a GPU, so we can afford a
richer per-article extraction than the budgeted TDM proxy run.

## Input / output

**Input:** a CSV with at least an id column and an article-text column. Common
export shapes work out of the box — the script auto-detects the text column
among `extracted_text`, `text`, `article_text`, `ocr_text`, `body`:

```csv
id,date,extracted_text,is_ontario_flood,decision,flood_type
1289135103,1954-10,"Hurricane Hazel sent the Humber River over its banks...",True,ontario_flood,river
```

If an `is_ontario_flood` column is present, the script keeps only the truthy
rows by default (pass `--no-filter` to process everything).

**Output:**
- `flood_details.jsonl` — one record per row, checkpointed and **resumable**
  (re-running skips ids already done).
- `flood_details.csv` — flat table of the extracted fields, rewritten each run.

Output columns: `id, date, event_date, flood_type, location, water_body,
cause, intensity, deaths, injuries, people_displaced, damage_estimate,
infrastructure_impact, article_date, status, error`.

## Files

| File | Purpose |
|------|---------|
| `extract_floods.py` | The extractor: DSPy + local Ollama, checkpointed/resumable. Has an offline `--self-test`. |
| `env.sh` | Shared env (PATH, model cache dir, per-job port, model tag). `source` it everywhere. |
| `install_ollama.sh` | Install Ollama under `$HOME` — **no admin needed**. |
| `start_ollama.sh` | Start a loopback-only Ollama server and wait until it answers. |
| `slurm_extract.sbatch` | SLURM job: GPU node → start Ollama → extract → stop Ollama. |
| `requirements.txt` | Python deps (`dspy-ai`, `tqdm`). |
| `docker/` | Optional Docker path, pinned to the reserved network ranges (see below). |

## Quick start (recommended: native, no Docker)

Per our CS contact, you can install Ollama in your home directory while on
`comps0`, and you have **no admin privileges** — so the native path below is
the simplest and avoids Docker's network pitfalls entirely.

### 1. One-time setup on a login node (e.g. `comps0`, has internet)

```bash
cd cluster

# Install Ollama under $HOME (no sudo).
bash install_ollama.sh

# Python env for the extractor.
python -m venv ~/flood_env
source ~/flood_env/bin/activate
pip install -r requirements.txt

# Pre-pull the model NOW, on the login node — compute nodes have no internet.
# This caches it under $OLLAMA_MODELS so the GPU node can load it offline.
source env.sh
bash start_ollama.sh
ollama pull "$OLLAMA_MODEL"     # default: llama3.1:8b
```

> **Model cache location.** Models are multi-GB. By default they go to
> `$HOME/.ollama/models`. If your home quota is small, point
> `CLUSTER_SCRATCH` at your scratch space before sourcing `env.sh`
> (`export CLUSTER_SCRATCH=/scratch/$USER`), and the cache moves with it.

### 2. Stage the data

Copy your exported Ontario-flood CSV into this folder (or pass `--csv` with a
full path):

```bash
cp /path/to/ontario_floods_export.csv cluster/ontario_floods_export.csv
```

### 3. Submit the GPU job

```bash
sbatch slurm_extract.sbatch
squeue --me                          # watch it schedule/run
tail -f flood-extract-<jobid>.out    # follow progress
```

The job starts Ollama on the allocated GPU node (on a **per-job port** so it
won't collide with other users on a shared node), runs the extractor with a
progress bar, writes `flood_details.{jsonl,csv}`, then stops the server.

To pass extra extractor flags or different paths, set env vars at submit time:

```bash
INPUT_CSV=my_floods.csv EXTRA_ARGS="--limit 500 --cot" sbatch slurm_extract.sbatch
```

> **Partition/QoS.** The `#SBATCH --partition`/`--qos` lines are commented out
> because names differ per cluster. Run `sinfo` to see GPU partitions and check
> the [SLURM resource page](https://support.cs.toronto.edu/systems/slurmresource.html),
> then uncomment and set them in `slurm_extract.sbatch`.

### Run interactively instead (debugging)

```bash
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00
source env.sh && source ~/flood_env/bin/activate
bash start_ollama.sh
python extract_floods.py --csv ontario_floods_export.csv --limit 50
```

## Extractor options

```bash
python extract_floods.py --self-test          # offline mechanical check (no Ollama/GPU)
python extract_floods.py --csv FILE            # run over an export
python extract_floods.py --limit 100           # quick sample
python extract_floods.py --no-filter           # ignore is_ontario_flood, do every row
python extract_floods.py --cot                 # ChainOfThought (more accurate, slower)
python extract_floods.py --model mistral:7b    # any pulled Ollama model
python extract_floods.py --total-rows 8000     # add a wall-time projection to the report
python extract_floods.py --help                # all options
```

The run is checkpointed to JSONL and resumes automatically; if a job hits its
time limit, just `sbatch` again and it continues where it stopped.

## Verify before you queue a GPU

The self-test exercises CSV parsing, the Ontario-flood filter, resume, and CSV
writing with a stubbed model — no network, no GPU, runs in a second:

```bash
python extract_floods.py --self-test
```

## A note on Docker (the network-range caveat)

Our CS contact flagged that **Docker's default bridge subnet collides with the
cluster's "red" network range**, causing unpredictable failures, and that you
have no admin rights. The native path above sidesteps Docker completely and is
the recommended workflow.

If you must use Docker, use the reserved ranges:

```
docker1  192.168.152.0/24
docker2  192.168.153.0/24
docker3  192.168.154.0/24
```

`docker/run_docker_ollama.sh` does this **without daemon changes**: it creates
a user bridge network pinned to a reserved subnet and attaches the Ollama
container to it, binding the API to `127.0.0.1` only. If you do control the
daemon, `docker/daemon.json` shows the daemon-wide equivalent
(`default-address-pools` + `bip`) so *every* network Docker creates stays off
the red range.

```bash
cd cluster/docker
DOCKER_SUBNET=192.168.152.0/24 DOCKER_GATEWAY=192.168.152.1 bash run_docker_ollama.sh
# then, from cluster/:
python extract_floods.py --csv ontario_floods_export.csv --port 11434
docker stop flood-ollama
```

## Troubleshooting

- **`model ... not found` on the compute node.** The model wasn't pre-pulled.
  Compute nodes have no internet — pull on a login node (step 1) so it lands in
  the shared `$OLLAMA_MODELS` cache.
- **Port already in use.** `env.sh` derives a per-job port from `$SLURM_JOB_ID`;
  override with `export OLLAMA_PORT=NNNNN` if needed.
- **Server never becomes ready.** Check `~/ollama_<jobid>.log`. Usually a GPU
  driver/VRAM issue — try a smaller model or confirm `nvidia-smi` works on the node.
- **Slow / CPU-only.** Confirm the job got a GPU (`echo $CUDA_VISIBLE_DEVICES`,
  `nvidia-smi`). Without `--gres=gpu:1` Ollama falls back to CPU and is far slower.
```
