#!/usr/bin/env python3
"""Local-LLM detail extraction for Ontario floods, built to run on the UofT CS
SLURM cluster against a *local* Ollama server (no external API, no spend).

Pipeline context
----------------
Inside ProQuest TDM, ``src/tdm_overnight.py`` filters the full ~91k-row corpus
down to the rows that are real Ontario floods and writes
``ontario_flood_predictions.csv``. You then export those Ontario-flood rows out
of TDM (joined back to their article text) and bring them to the cluster. This
script is the *deep extraction* step: for each Ontario flood it pulls the
structured detail the TDM filter did not — event date, flood type, severity /
impact (deaths, injuries, displaced, damage, infrastructure), water body, and
cause — using a local model served by Ollama on a GPU node.

Why a separate, self-contained script
--------------------------------------
The cluster has no internet access from compute nodes, no admin rights, and a
shared environment, so this file deliberately repeats a little of the project's
DSPy setup rather than importing ``src/`` — it can be copied to ``$HOME`` and
run on its own. It talks only to ``http://127.0.0.1:$OLLAMA_PORT``.

Input CSV
---------
Any CSV with an id column, an article-text column, and (optionally) the TDM
prediction columns. By default the script keeps only rows the TDM step marked
as Ontario floods (``is_ontario_flood`` truthy); pass ``--no-filter`` to
extract from every row instead.

    id, date, extracted_text, is_ontario_flood, decision, flood_type, ...

Output
------
Checkpointed JSONL (one record per row, resumable) plus a flat CSV with the
extracted fields. Re-running skips ids already in the JSONL.

Usage
-----
    # offline mechanical check — no Ollama, no GPU, no network
    python extract_floods.py --self-test

    # real run (Ollama already serving on $OLLAMA_PORT; see slurm_extract.sbatch)
    python extract_floods.py \
        --csv ontario_floods_export.csv \
        --model llama3.1:8b \
        --port 11434

    python extract_floods.py --limit 100        # quick sample
    python extract_floods.py --help             # all options
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Article extracts can be very large; raise CSV's per-field limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Ordered list of the fields we extract. Kept here (not on the signature) so the
# CSV writer, self-test, and resume logic share one source of truth.
DETAIL_FIELDS = [
    "event_date",        # when the flood happened (YYYY / YYYY-MM / YYYY-MM-DD / season+year / unknown)
    "flood_type",        # river | lake | flash | ice_jam | dam_break | spring_freshet | storm_surge | urban_storm | coastal | unknown
    "location",          # Ontario place(s) affected, comma-separated
    "water_body",        # named river/lake/creek, else unknown
    "cause",             # rain | snowmelt | ice_jam | dam_failure | storm | freshet | unknown
    "intensity",         # short free-text severity/impact summary
    "deaths",            # integer-as-text or unknown
    "injuries",          # integer-as-text or unknown
    "people_displaced",  # evacuated/homeless count or unknown
    "damage_estimate",   # monetary figure exactly as stated, else unknown
    "infrastructure_impact",  # bridges/roads/dams/buildings affected, else unknown
    "article_date",      # publication date if stated, else unknown
]

_TRUE = {"1", "true", "yes", "y", "t"}


def is_truthy(value) -> bool:
    return str(value).strip().lower() in _TRUE


# ---------------------------------------------------------------------------
# DSPy signature + program (imported lazily so --self-test needs no dspy/ollama)
# ---------------------------------------------------------------------------
def build_signature():
    import dspy

    class FloodDetailExtraction(dspy.Signature):
        """Extract structured detail about a REAL Ontario flood from a noisy OCR
        newspaper extract.

        The article has already been verified to describe a flood that occurred
        in Ontario. The text is OCR from 1800s-1900s newspapers; read past
        obvious OCR errors (e.g. "fl ood", "tlie", "rivor").

        Output rules (important):
        - Extract ONLY what the article itself states. Do NOT invent, guess, or
          infer from outside knowledge.
        - Each field is a SHORT value only — no sentences, no explanations, no
          parenthetical notes, no markdown.
        - If the article does not state a field, output exactly: unknown
        - Numeric fields (deaths, injuries, people_displaced) must be a number
          or 'unknown' — never a range or a word.
        - Never copy values from these instructions; read them from the article.
        """

        article_text: str = dspy.InputField(desc="OCR extract of a single Ontario flood article")

        event_date: str = dspy.OutputField(
            desc="When the flood occurred, from the article's own words: a year "
            "(YYYY), year-month (YYYY-MM), full date (YYYY-MM-DD), or season+year. "
            "'unknown' if not stated. Do not infer from context."
        )
        flood_type: str = dspy.OutputField(
            desc="One of: river | lake | flash | ice_jam | dam_break | "
            "spring_freshet | storm_surge | urban_storm | coastal | unknown."
        )
        location: str = dspy.OutputField(
            desc="Ontario place(s) the flood affected (city/town, river, watershed) "
            "as named in the article. Comma-separated if several. 'unknown' if not stated."
        )
        water_body: str = dspy.OutputField(
            desc="Named river/lake/creek that flooded, if stated. 'unknown' otherwise."
        )
        cause: str = dspy.OutputField(
            desc="What caused the flood: rain | snowmelt | ice_jam | dam_failure | "
            "storm | freshet | unknown."
        )
        intensity: str = dspy.OutputField(
            desc="A few words on severity/impact (water levels, scale of damage, "
            "evacuations) as stated. 'unknown' if not stated."
        )
        deaths: str = dspy.OutputField(
            desc="Number of deaths stated, as a plain integer. 'unknown' if not stated."
        )
        injuries: str = dspy.OutputField(
            desc="Number of people injured stated, as a plain integer. 'unknown' if not stated."
        )
        people_displaced: str = dspy.OutputField(
            desc="Number evacuated/left homeless stated, as a plain integer. 'unknown' if not stated."
        )
        damage_estimate: str = dspy.OutputField(
            desc="Monetary damage figure exactly as stated (keep the currency/units). "
            "'unknown' if not stated."
        )
        infrastructure_impact: str = dspy.OutputField(
            desc="Bridges/roads/dams/railways/buildings damaged or destroyed, as stated. "
            "'unknown' if not stated."
        )
        article_date: str = dspy.OutputField(
            desc="Publication date of the article if stated (YYYY-MM-DD if possible). "
            "'unknown' otherwise."
        )

    return FloodDetailExtraction


# Injectable so --self-test can swap in a fake program with no LM at all.
_PROGRAM_FACTORY = None


def build_program(args):
    """Return ``extract(text) -> dict`` backed by a local Ollama model via DSPy."""
    if _PROGRAM_FACTORY is not None:  # self-test injection
        return _PROGRAM_FACTORY(args)

    import dspy

    api_base = f"http://{args.host}:{args.port}"
    # ollama_chat/ uses Ollama's /api/chat endpoint, which DSPy/LiteLLM prefer.
    lm = dspy.LM(
        f"ollama_chat/{args.model}",
        api_base=api_base,
        api_key="ollama",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_retries=args.num_retries,
        cache=not args.no_cache,
    )
    dspy.configure(lm=lm, track_usage=True)

    signature = build_signature()
    predictor = dspy.ChainOfThought(signature) if args.cot else dspy.Predict(signature)

    def extract(text: str) -> tuple[dict, dict]:
        hist_start = len(lm.history)
        pred = predictor(article_text=text)
        fields = {f: _clean(getattr(pred, f, "unknown")) for f in DETAIL_FIELDS}
        usage = _usage_since(lm, hist_start)
        return fields, usage

    return extract


def _clean(value) -> str:
    s = "" if value is None else str(value).strip()
    return s if s else "unknown"


def _usage_since(lm, hist_start: int) -> dict:
    """Sum prompt/completion tokens + call count from new lm.history entries."""
    ptok = ctok = calls = 0
    for entry in lm.history[hist_start:]:
        calls += 1
        u = entry.get("usage") or {}
        ptok += int(u.get("prompt_tokens") or 0)
        ctok += int(u.get("completion_tokens") or 0)
    return {"n_calls": calls, "prompt_tokens": ptok, "completion_tokens": ctok}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_done_ids(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def pick_text_column(fieldnames, requested):
    """Resolve the article-text column, tolerating the common TDM/export names."""
    if requested and requested in (fieldnames or []):
        return requested
    for candidate in (requested, "extracted_text", "text", "article_text", "ocr_text", "body"):
        if candidate and candidate in (fieldnames or []):
            return candidate
    return None


def write_csv(jsonl_path: Path, csv_path: Path):
    """Flatten the JSONL output into a flat CSV of the extracted fields."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "date"] + DETAIL_FIELDS + ["status", "error"]
    with open(jsonl_path, encoding="utf-8") as f_in, \
            open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            writer.writerow(json.loads(line))


def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def report(stats: dict, total_target: int):
    n = stats["rows"]
    if n == 0:
        print("\nNo new rows processed (all skipped via resume?). Nothing to report.")
        return
    secs = stats["seconds"]
    print("\n" + "=" * 60)
    print(f"EXTRACTION METRICS  ({n} rows this run)")
    print("=" * 60)
    print(f"  ok                : {stats['ok']}")
    print(f"  errors            : {stats['errors']}")
    print(f"  LM calls          : {stats['calls']}  (avg {stats['calls']/n:.2f}/row)")
    print(f"  prompt tokens     : {stats['ptok']:,}")
    print(f"  completion tokens : {stats['ctok']:,}")
    print(f"  wall time         : {_fmt_hms(secs)}  (avg {secs/n:.2f}s/row)")
    if total_target and total_target > n:
        scale = total_target / n
        print(f"  projected {total_target:,} rows : {_fmt_hms(secs*scale)} (single GPU, this throughput)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args):
    if not args.csv.exists():
        raise SystemExit(f"Input CSV not found: {args.csv}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set() if args.no_resume else load_done_ids(args.out)
    if done:
        print(f"[resume] {len(done)} rows already in {args.out}; skipping them.")

    extract = build_program(args)

    stats = {"rows": 0, "ok": 0, "errors": 0, "calls": 0,
             "ptok": 0, "ctok": 0, "seconds": 0.0}

    with open(args.csv, newline="", encoding="utf-8", errors="replace") as f_in, \
            open(args.out, "a", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        text_col = pick_text_column(reader.fieldnames, args.text_col)
        if text_col is None:
            raise SystemExit(
                f"No text column found. Looked for '{args.text_col}'. "
                f"CSV columns: {reader.fieldnames}. Use --text-col."
            )
        if text_col != args.text_col:
            print(f"[info] using text column '{text_col}'")
        has_filter_col = "is_ontario_flood" in (reader.fieldnames or [])
        if args.filter and not has_filter_col:
            print("[info] no 'is_ontario_flood' column present; extracting all rows.")

        try:
            from tqdm import tqdm
            reader = tqdm(reader, desc="extract", unit="row")
        except Exception:
            pass

        for row in reader:
            if args.limit is not None and stats["rows"] >= args.limit:
                break
            row_id = str(row.get(args.id_col, "")).strip()
            if not row_id or row_id in done:
                continue
            if args.filter and has_filter_col and not is_truthy(row.get("is_ontario_flood")):
                continue
            text = (row.get(text_col) or "").strip()
            if not text:
                continue

            t0 = time.perf_counter()
            try:
                fields, usage = extract(text)
                status, error = "ok", None
            except Exception as e:  # never let one row kill the batch
                fields = {f: "unknown" for f in DETAIL_FIELDS}
                usage = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
                status, error = "error", f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0

            rec = {"id": row_id, "date": row.get(args.date_col, ""), "status": status}
            rec.update(fields)
            rec.update(usage)
            rec["seconds"] = round(elapsed, 3)
            if error:
                rec["error"] = error
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            done.add(row_id)

            stats["rows"] += 1
            stats["ok"] += int(status == "ok")
            stats["errors"] += int(status == "error")
            stats["calls"] += usage["n_calls"]
            stats["ptok"] += usage["prompt_tokens"]
            stats["ctok"] += usage["completion_tokens"]
            stats["seconds"] += elapsed

    report(stats, args.total_rows)
    write_csv(args.out, args.out_csv)
    print(f"\nWrote {args.out_csv}")


# ---------------------------------------------------------------------------
# Self-test: exercise IO, resume, and CSV writing offline (no dspy/ollama)
# ---------------------------------------------------------------------------
def run_self_test():
    import tempfile

    global _PROGRAM_FACTORY

    def fake_factory(args):
        def extract(text):
            fields = {f: "unknown" for f in DETAIL_FIELDS}
            fields["event_date"] = "1954-10"
            fields["flood_type"] = "river"
            fields["location"] = "Toronto"
            fields["intensity"] = f"sample ({len(text)} chars)"
            return fields, {"n_calls": 1, "prompt_tokens": 10, "completion_tokens": 5}
        return extract

    _PROGRAM_FACTORY = fake_factory
    print("=== SELF-TEST (fake program, no network) ===")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        csv_path = d / "in.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "date", "extracted_text", "is_ontario_flood"])
            w.writeheader()
            w.writerow({"id": "1", "date": "1954", "extracted_text": "Hurricane Hazel flooded the Humber River.", "is_ontario_flood": "True"})
            w.writerow({"id": "2", "date": "1900", "extracted_text": "A flood of letters arrived.", "is_ontario_flood": "False"})
            w.writerow({"id": "3", "date": "1936", "extracted_text": "The Grand River overflowed at Galt.", "is_ontario_flood": "1"})

        args = build_args([
            "--csv", str(csv_path),
            "--out", str(d / "out.jsonl"),
            "--out-csv", str(d / "out.csv"),
        ])
        run(args)

        lines = (d / "out.jsonl").read_text().splitlines()
        ids = {json.loads(l)["id"] for l in lines}
        assert ids == {"1", "3"}, f"filter failed: {ids}"  # row 2 dropped by is_ontario_flood
        print(f"[ok] filtered to Ontario floods: {sorted(ids)}")

        # Resume: a second run adds nothing.
        run(args)
        lines2 = (d / "out.jsonl").read_text().splitlines()
        assert len(lines2) == len(lines), "resume re-processed rows"
        print(f"[ok] resume skipped already-done rows ({len(lines2)} records)")

        # --no-filter picks up the metaphor row too.
        args2 = build_args([
            "--csv", str(csv_path),
            "--out", str(d / "out_all.jsonl"),
            "--out-csv", str(d / "out_all.csv"),
            "--no-filter",
        ])
        run(args2)
        ids_all = {json.loads(l)["id"] for l in (d / "out_all.jsonl").read_text().splitlines()}
        assert ids_all == {"1", "2", "3"}, f"--no-filter failed: {ids_all}"
        print(f"[ok] --no-filter kept all rows: {sorted(ids_all)}")

        header = (d / "out.csv").read_text().splitlines()[0]
        assert "event_date" in header and "flood_type" in header
        print(f"[ok] CSV header has detail fields")

    _PROGRAM_FACTORY = None
    print("=== SELF-TEST PASSED ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, default=Path("ontario_floods_export.csv"),
                   help="Input CSV exported from TDM (Ontario-flood rows + text)")
    p.add_argument("--out", type=Path, default=Path("flood_details.jsonl"),
                   help="Checkpoint JSONL (append/resume)")
    p.add_argument("--out-csv", type=Path, default=Path("flood_details.csv"),
                   help="Flat CSV of extracted fields (rewritten each run)")
    p.add_argument("--text-col", default="extracted_text", help="Column holding article text")
    p.add_argument("--id-col", default="id", help="Column holding the row id")
    p.add_argument("--date-col", default="date", help="Column carried through to output")

    # Local Ollama endpoint.
    p.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
                   help="Ollama model tag (must be pulled already)")
    p.add_argument("--host", default="127.0.0.1", help="Ollama host")
    p.add_argument("--port", type=int, default=int(os.environ.get("OLLAMA_PORT", "11434")),
                   help="Ollama port (set per SLURM job to avoid clashes)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--num-retries", type=int, default=3)
    p.add_argument("--no-cache", action="store_true", help="Disable DSPy/LiteLLM response cache")
    p.add_argument("--cot", action="store_true",
                   help="Use ChainOfThought (more accurate, slower) instead of Predict")

    p.add_argument("--filter", dest="filter", action="store_true", default=True,
                   help="Keep only is_ontario_flood rows (default)")
    p.add_argument("--no-filter", dest="filter", action="store_false",
                   help="Extract from every row regardless of is_ontario_flood")
    p.add_argument("--limit", type=int, default=None, help="Process at most N new rows")
    p.add_argument("--no-resume", action="store_true", help="Reprocess rows already in --out")
    p.add_argument("--total-rows", type=int, default=0,
                   help="Expected total rows, for the wall-time projection (0 = skip)")

    p.add_argument("--self-test", action="store_true",
                   help="Run an offline mechanical check and exit (no Ollama/GPU)")
    return p.parse_args(argv)


def main():
    args = build_args()
    if args.self_test:
        run_self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
