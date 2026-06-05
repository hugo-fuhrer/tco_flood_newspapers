"""Optimize the Stage 2 filters with DSPy using manually-labelled data.

Reads labelled examples from ``data/raw/annotations_so_far.csv`` and uses them
as a trainset to compile better few-shot prompts for the two filter signatures
(:class:`floodIdentification` and :class:`isOntario`), then saves the compiled
programs to ``artifacts/`` so the pipeline can load them.

CSV columns (header required): ``id, date, example, changed, flood, ontario``
(an unnamed leading index column, as written by ``pandas.to_csv``, is ignored).
  - ``example`` : an extract of a newspaper article (the text to classify)
  - ``flood``   : True if the extract references a *real* flood event, else False
  - ``ontario`` : True if that flood occurred in Ontario, else False
  - ``date``    : not used for training (kept for reference)
  - ``changed`` : ignored

The Ontario filter is only trained on rows where ``flood == 1``, mirroring the
pipeline cascade where ``isOntario`` only ever sees articles that already passed
``floodIdentification``.

Run from the ``src/`` directory (matching the pipeline's bare imports):

    cd src && python optimize.py --help
"""
import argparse
import csv
import random
from pathlib import Path

import dspy

from signatures import floodIdentification, isOntario

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "annotations_so_far.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Truthy spellings accepted in the binary label columns.
_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f", ""}


def _to_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a binary label")


def load_rows(csv_path: Path) -> list[dict]:
    """Read the labelled CSV into a list of dicts with parsed binary labels."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Labelled data not found at {csv_path}. Expected columns: "
            "id, date, example, changed, flood, ontario"
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"example", "flood", "ontario"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")
        for i, r in enumerate(reader, 1):
            text = (r.get("example") or "").strip()
            if not text:
                continue  # skip blank extracts
            rows.append(
                {
                    "id": (r.get("id") or "").strip(),
                    "date": (r.get("date") or "").strip(),
                    "text": text,
                    "flood": _to_bool(r.get("flood")),
                    "ontario": _to_bool(r.get("ontario")),
                }
            )
    if not rows:
        raise ValueError(f"No usable rows found in {csv_path}")
    return rows


def build_examples(rows: list[dict]):
    """Build (flood_examples, ontario_examples) as dspy.Example lists.

    Both filter signatures take ``article_text`` and ``title`` as inputs; the
    CSV has no separate title, so ``title`` is left empty.
    """
    flood_examples = [
        dspy.Example(
            article_text=r["text"], title="", flood=r["flood"]
        ).with_inputs("article_text", "title")
        for r in rows
    ]
    ontario_examples = [
        dspy.Example(
            article_text=r["text"], title="", ontario=r["ontario"]
        ).with_inputs("article_text", "title")
        for r in rows
        if r["flood"]  # only real floods reach the Ontario filter in the pipeline
    ]
    return flood_examples, ontario_examples


def split(examples: list, dev_frac: float, seed: int):
    """Shuffle and split into (train, dev). Falls back to dev == train if tiny."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_dev = int(round(len(shuffled) * dev_frac))
    if len(shuffled) < 4 or n_dev == 0:
        # Too few examples to hold any out meaningfully; train on all, eval on all.
        return shuffled, shuffled
    return shuffled[n_dev:], shuffled[:n_dev]


def truncate_demos(examples: list, max_chars: int) -> list:
    """Shorten each example's article_text so baked-in demos stay compact.

    Few-shot demos are sent on every inference call, so long demo articles
    dominate the prompt. Truncating them cuts tokens/latency at inference; the
    eval/dev set is left untouched so accuracy is measured on full text.
    """
    if not max_chars:
        return examples
    out = []
    for ex in examples:
        data = {k: (v[:max_chars] if k == "article_text" and isinstance(v, str) else v)
                for k, v in ex.items()}
        out.append(dspy.Example(**data).with_inputs(*ex.inputs().keys()))
    return out


def flood_metric(example, pred, trace=None) -> bool:
    return bool(getattr(pred, "flood_mentioned", None)) == bool(example.flood)


def ontario_metric(example, pred, trace=None) -> bool:
    return bool(getattr(pred, "is_ontario", None)) == bool(example.ontario)


def optimize_filter(name, signature, metric, examples, args):
    """Baseline-evaluate, compile with BootstrapFewShot, re-evaluate, and save."""
    if len(examples) < 2:
        print(f"[{name}] only {len(examples)} example(s); skipping (need >= 2).")
        return None

    train, dev = split(examples, args.dev_frac, args.seed)
    train = truncate_demos(train, args.max_demo_chars)
    print(f"\n=== Optimizing {name} filter ===")
    print(f"  examples={len(examples)}  train={len(train)}  dev={len(dev)}"
          f"  max_demo_chars={args.max_demo_chars}")

    program = dspy.Predict(signature)
    evaluate = dspy.Evaluate(
        devset=dev, metric=metric, num_threads=args.num_threads, display_progress=True
    )

    print(f"[{name}] baseline accuracy:")
    baseline = evaluate(program)

    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=args.max_labeled_demos,
        max_rounds=args.max_rounds,
    )
    compiled = optimizer.compile(program, trainset=train)

    print(f"[{name}] optimized accuracy:")
    optimized = evaluate(compiled)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"{name}_filter.json"
    compiled.save(str(out_path))
    print(f"[{name}] baseline={baseline}  optimized={optimized}  saved -> {out_path}")
    return compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to labelled CSV")
    parser.add_argument("--model", default="ollama/llama3.1:8b", help="DSPy LM identifier")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434", help="LM API base URL")
    parser.add_argument("--api-key", default="ollama", help="LM API key")
    parser.add_argument(
        "--filters", choices=["flood", "ontario", "both"], default="both",
        help="Which filter(s) to optimize",
    )
    parser.add_argument("--dev-frac", type=float, default=0.25, help="Fraction held out for evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle/split seed")
    parser.add_argument("--max-demos", type=int, default=4, help="max_bootstrapped_demos")
    parser.add_argument("--max-labeled-demos", type=int, default=16, help="max_labeled_demos")
    parser.add_argument("--max-rounds", type=int, default=1, help="BootstrapFewShot rounds")
    parser.add_argument("--max-demo-chars", type=int, default=None,
                        help="Truncate each demo's article_text to this many chars (cuts inference tokens)")
    parser.add_argument("--num-threads", type=int, default=1, help="Eval threads")
    args = parser.parse_args()

    lm = dspy.LM(args.model, api_base=args.api_base, api_key=args.api_key)
    dspy.configure(lm=lm)

    rows = load_rows(args.csv)
    flood_examples, ontario_examples = build_examples(rows)
    print(
        f"Loaded {len(rows)} rows: {sum(r['flood'] for r in rows)} flood, "
        f"{sum(r['ontario'] for r in rows)} ontario."
    )

    if args.filters in ("flood", "both"):
        optimize_filter("flood", floodIdentification, flood_metric, flood_examples, args)
    if args.filters in ("ontario", "both"):
        optimize_filter("ontario", isOntario, ontario_metric, ontario_examples, args)


if __name__ == "__main__":
    main()
