"""Batch-run the optimized pipeline over unlabeled newspaper extracts.

Reads ``data/raw/extracted_only_1.csv`` (columns: id, date, extracted_text),
runs each row through :func:`pipeline.process_article` (which loads the
optimized flood/Ontario filters from ``artifacts/``), and appends one JSON
record per row to ``data/processed/``.

The run is checkpointed: results are appended to a JSONL file and already-seen
``id``s are skipped on restart, so an interrupted run resumes cleanly.

Every row is instrumented for LM token usage, wall-time, and cost. At the end a
report prints the measured totals/averages and extrapolates them to the full
shard (~22,860 rows) and corpus (~91,000 rows).

Run from the ``src/`` directory:

    cd src && python run_inference.py --limit 100        # sample run
    cd src && python run_inference.py                    # full shard
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import dspy
from tqdm import tqdm

from pipeline import lm, process_article

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "extracted_only_1.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "extracted_only_1.predictions.jsonl"

# Some article extracts are huge; CSV's default field-size limit can choke.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_ENC = None


def _approx_tokens(text: str) -> int:
    """Token estimate via tiktoken; falls back to a chars/4 heuristic."""
    global _ENC
    if not text:
        return 0
    try:
        if _ENC is None:
            import tiktoken

            _ENC = tiktoken.get_encoding("cl100k_base")
        return len(_ENC.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _entry_usage(entry: dict) -> tuple[int, int, float]:
    """Extract (prompt_tokens, completion_tokens, cost) from an lm.history entry.

    Prefers the usage reported by LiteLLM/Ollama; falls back to estimating
    tokens from the rendered prompt/response so the report is never empty.
    """
    usage = entry.get("usage") or {}
    ptok = int(usage.get("prompt_tokens") or 0)
    ctok = int(usage.get("completion_tokens") or 0)
    cost = entry.get("cost")

    if ptok == 0 and ctok == 0:
        # Fallback: estimate from the messages and the textual outputs.
        prompt_text = ""
        for msg in entry.get("messages") or []:
            if isinstance(msg, dict):
                prompt_text += str(msg.get("content", "")) + "\n"
        if not prompt_text:
            prompt_text = str(entry.get("prompt") or "")
        resp_text = ""
        for out in entry.get("outputs") or []:
            resp_text += str(out) + "\n"
        ptok = _approx_tokens(prompt_text)
        ctok = _approx_tokens(resp_text)

    return ptok, ctok, float(cost) if isinstance(cost, (int, float)) else 0.0


def load_processed_ids(out_path: Path) -> set:
    """Collect ids already written to the JSONL output (for resume)."""
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


def build_record(row_id, date, text, result, n_calls, ptok, ctok, cost, seconds):
    """Flatten a process_article result into a single JSONL-friendly dict."""
    rec = {
        "id": row_id,
        "date": date,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "ext_date": result.get("date"),
        "location": result.get("location"),
        "intensity": result.get("intensity"),
        "corrected_text": result.get("corrected_text"),
        "n_calls": n_calls,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "cost": round(cost, 6),
        "seconds": round(seconds, 3),
    }
    if "error" in result:
        rec["error"] = result["error"]
    return rec


def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def report(stats: dict, args):
    """Print measured totals + extrapolation to the shard and full corpus."""
    n = stats["rows"]
    if n == 0:
        print("\nNo new rows processed (all skipped via resume?). Nothing to report.")
        return

    ptok, ctok = stats["ptok"], stats["ctok"]
    ttok = ptok + ctok
    secs, cost, calls = stats["seconds"], stats["cost"], stats["calls"]
    accepted, errors = stats["accepted"], stats["errors"]

    # Per-row averages embed the observed accept-rate mix (accepted rows cost
    # ~4 LM calls, rejected ~1-2), so scaling them by row count is a fair model.
    def avg(x):
        return x / n

    print("\n" + "=" * 64)
    print(f"SAMPLE METRICS  ({n} rows processed this run)")
    print("=" * 64)
    print(f"  accepted (Ontario floods) : {accepted}  ({accepted / n:.1%})")
    print(f"  rejected                  : {n - accepted - errors}")
    print(f"  errors                    : {errors}")
    print(f"  LM calls                  : {calls}  (avg {avg(calls):.2f}/row)")
    print(f"  prompt tokens             : {ptok:,}  (avg {avg(ptok):,.0f}/row)")
    print(f"  completion tokens         : {ctok:,}  (avg {avg(ctok):,.0f}/row)")
    print(f"  total tokens              : {ttok:,}  (avg {avg(ttok):,.0f}/row)")
    print(f"  wall time                 : {_fmt_hms(secs)}  (avg {avg(secs):.2f}s/row)")
    print(f"  measured cost             : ${cost:,.4f}")
    if args.price_in or args.price_out:
        priced = (ptok / 1000) * args.price_in + (ctok / 1000) * args.price_out
        print(f"  priced cost (@cli rates)  : ${priced:,.4f}  "
              f"(in ${args.price_in}/1K, out ${args.price_out}/1K)")

    print("\n" + "-" * 64)
    print(f"{'PROJECTION':<14}{'rows':>10}{'tokens':>16}{'wall (1 thread)':>18}{'cost':>12}")
    print("-" * 64)
    for label, rows in (("this shard", args.shard_rows), ("full corpus", args.corpus_rows)):
        scale = rows / n
        proj_tok = ttok * scale
        proj_secs = secs * scale
        if args.price_in or args.price_out:
            proj_cost = ((ptok / 1000) * args.price_in + (ctok / 1000) * args.price_out) * scale
        else:
            proj_cost = cost * scale
        print(f"{label:<14}{rows:>10,}{proj_tok:>16,.0f}{_fmt_hms(proj_secs):>18}{('$%.2f' % proj_cost):>12}")
    print("-" * 64)
    print("Caveats: projection assumes this sample's accept-rate and throughput")
    print("hold across the corpus; wall-time is single-threaded (parallelism cuts it).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV")
    parser.add_argument("--text-col", default="extracted_text", help="Column holding article text")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSONL (append/resume)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N new rows")
    parser.add_argument("--shard-rows", type=int, default=22860, help="Rows in this shard (extrapolation)")
    parser.add_argument("--corpus-rows", type=int, default=91000, help="Rows in full corpus (extrapolation)")
    parser.add_argument("--price-in", type=float, default=0.0, help="$ per 1K prompt tokens (0 = local)")
    parser.add_argument("--price-out", type=float, default=0.0, help="$ per 1K completion tokens (0 = local)")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess rows even if already in --out")
    parser.add_argument("--clean-ocr", action="store_true",
                        help="Run OCR correction on accepted articles before extraction (off by default)")
    parser.add_argument("--accepted-csv", type=Path, default=None,
                        help="Also write accepted records to this CSV at the end")
    args = parser.parse_args()

    dspy.settings.configure(track_usage=True)

    if not args.csv.exists():
        parser.error(f"Input CSV not found: {args.csv}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    processed = set() if args.no_resume else load_processed_ids(args.out)
    if processed:
        print(f"[resume] {len(processed)} rows already in {args.out}; skipping them.")

    stats = {"rows": 0, "accepted": 0, "errors": 0, "calls": 0,
             "ptok": 0, "ctok": 0, "cost": 0.0, "seconds": 0.0}

    with open(args.csv, newline="", encoding="utf-8", errors="replace") as f_in, \
            open(args.out, "a", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        if args.text_col not in (reader.fieldnames or []):
            parser.error(f"--text-col '{args.text_col}' not in CSV columns {reader.fieldnames}")

        pbar = tqdm(reader, desc="inference", unit="row")
        for row in pbar:
            if args.limit is not None and stats["rows"] >= args.limit:
                break
            row_id = str(row.get("id", "")).strip()
            if row_id in processed:
                continue
            text = (row.get(args.text_col) or "").strip()
            if not text:
                continue

            hist_start = len(lm.history)
            t0 = time.perf_counter()
            try:
                result = process_article(text, title="", clean_ocr=args.clean_ocr)
            except Exception as e:  # never let one row kill the batch
                result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            elapsed = time.perf_counter() - t0

            ptok = ctok = 0
            cost = 0.0
            new_calls = lm.history[hist_start:]
            for entry in new_calls:
                p, c, k = _entry_usage(entry)
                ptok += p
                ctok += c
                cost += k

            rec = build_record(row_id, row.get("date", ""), text, result,
                               len(new_calls), ptok, ctok, cost, elapsed)
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            processed.add(row_id)

            stats["rows"] += 1
            stats["accepted"] += int(result.get("status") == "accepted")
            stats["errors"] += int(result.get("status") == "error")
            stats["calls"] += len(new_calls)
            stats["ptok"] += ptok
            stats["ctok"] += ctok
            stats["cost"] += cost
            stats["seconds"] += elapsed
            pbar.set_postfix(acc=stats["accepted"], err=stats["errors"])

    report(stats, args)

    if args.accepted_csv:
        write_accepted_csv(args.out, args.accepted_csv)
        print(f"\nWrote accepted records -> {args.accepted_csv}")


def write_accepted_csv(jsonl_path: Path, csv_path: Path):
    """Derive an accepted-only CSV from the JSONL output."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "date", "ext_date", "location", "intensity"]
    with open(jsonl_path, encoding="utf-8") as f_in, \
            open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "accepted":
                writer.writerow(rec)


if __name__ == "__main__":
    main()
