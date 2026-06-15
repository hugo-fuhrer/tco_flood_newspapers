"""Overnight ProQuest TDM job: optimize a recall-first prompt, then label floods.

This is a SINGLE self-contained script meant to be dropped into a ProQuest TDM
Studio notebook/instance and left to run overnight. It does three things in one
pass:

  Phase A — OPTIMIZE & SELECT a prompt
      Reads the 250 hand-labelled rows in ``annotations_so_far.csv`` and uses
      DSPy to compile few-shot prompts for an "is this a real flood that
      occurred in Ontario?" classifier. Several candidate models are tried and
      each is scored on a held-out split with a RECALL-FIRST objective (we would
      rather over-include than lose a real flood). Evaluation metrics for the
      top 3 prompts are printed and written to ``artifacts/``.

  Phase B — RUN the best prompt on the unlabelled corpus
      Takes the winning (model + compiled prompt) and labels a subset of the
      ~91k unlabelled extracts (schema of ``extracted_only_1.csv``), sized to
      fit the daily budget. For every row it emits a specific REASON, the flood
      LOCATION (when the flood was outside Ontario), WHY it is not a flood
      (metaphor / not a specific event / artificial / ...), and the flood TYPE
      (river / lake / flash / ice jam / ...).

Everything is checkpointed to JSONL so an interrupted overnight run resumes
cleanly the next day, and a hard dollar budget is enforced throughout.

------------------------------------------------------------------------------
TDM platform notes (matched to the previous OCR-correction script):
  * Auth token file : /home/ec2-user/SageMaker/.token/.agaitoken
  * OpenAI-compatible proxy base_url:
        https://agai-proxy.prod.int.tdmstudio.proquest.com/large-language-models-openai-compatible/
  * Pricing helper  : ./scripts/model_pricing.py  (MODEL_PRICING, METRIC)
  DSPy talks to the proxy through LiteLLM using the ``openai/<model>`` prefix.
------------------------------------------------------------------------------

Examples
--------
    # Dry mechanical check, no network / no proxy / no spend:
    python tdm_overnight.py --self-test

    # Real overnight run with defaults ($50 budget, 3 candidate models):
    python tdm_overnight.py

    # Just (re)label using a prompt compiled on a previous night:
    python tdm_overnight.py --reuse-best --skip-optimize

    # Try the reasoning / gpt-5 models too:
    python tdm_overnight.py --models gpt-4o-mini,gpt-4.1-nano,o4-mini,gpt-5

    # (1) CONTINUE labelling the leftover unlabelled rows with the saved prompt,
    #     pushing through the whole remainder regardless of the daily cap:
    python tdm_overnight.py --continue-run --ignore-budget

    # (2) TEST more models/configs on the 250 hand-labelled rows (no corpus run);
    #     writes artifacts/model_test_report.{md,json} + per-row predictions:
    python tdm_overnight.py --test-models \
        --models gpt-4o-mini,gpt-4.1-mini,gpt-5-nano --strategies zeroshot,fewshot \
        --reasoning-efforts minimal,low

    # (3) GENERATE a full markdown report (metrics, confusion matrices,
    #     distributions, FP/FN analysis on the 250, and cost/time projections
    #     for the remaining corpus that fold in the rows already labelled):
    python tdm_overnight.py --report --report-model gpt-4o-mini
    python tdm_overnight.py --report --reuse-eval artifacts/labeled_eval__gpt-4o-mini__best.jsonl
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- Network-restricted environment (TDM) -----------------------------------
# TDM blocks outbound internet, so LiteLLM must NOT try to fetch its remote
# model-cost map (the "[Errno 101] Network is unreachable" warning). Force the
# local backup BEFORE importing dspy/litellm, and quiet the import-time chatter.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_LOG", "ERROR")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import dspy  # noqa: E402
from dspy.adapters.base import Adapter  # noqa: E402

try:  # belt-and-suspenders: silence LiteLLM and let it drop unknown params
    import litellm

    litellm.suppress_debug_info = True
    litellm.drop_params = True  # so reasoning_effort/verbosity never hard-error
    litellm.telemetry = False
except Exception:
    pass

# ----------------------------------------------------------------------------
# Paths (defaults mirror the rest of the repo; override on the CLI for TDM).
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "raw" / "annotations_so_far.csv"
DEFAULT_EXTRACTS = PROJECT_ROOT / "data" / "raw" / "extracted_only_1.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TDM_TOKEN_FILE = "/home/ec2-user/SageMaker/.token/.agaitoken"
TDM_BASE_URL = (
    "https://agai-proxy.prod.int.tdmstudio.proquest.com/"
    "large-language-models-openai-compatible/"
)

# Big OCR extracts blow past csv's default field-size limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Some article extracts are huge; classifying the whole thing wastes tokens and
# the signal for "flood / Ontario / type" is almost always near the top.
MAX_TEXT_CHARS = 6000


# ============================================================================
# Model names + pricing
# ============================================================================
# The user listed models with underscores (gpt_41, gpt_4o_mini, ...); the proxy
# wants the canonical OpenAI ids (gpt-4.1, gpt-4o-mini, ...). Accept either.
MODEL_ALIASES = {
    "gpt_41": "gpt-4.1",
    "gpt_41_2025_04_14": "gpt-4.1-2025-04-14",
    "gpt_41_nano": "gpt-4.1-nano",
    "gpt_41_nano_2025_04_14": "gpt-4.1-nano-2025-04-14",
    "gpt_41_mini": "gpt-4.1-mini",
    "gpt_4o": "gpt-4o",
    "gpt_4o_2024_05_13": "gpt-4o-2024-05-13",
    "gpt_4o_2024_08_06": "gpt-4o-2024-08-06",
    "gpt_4o_2024_11_20": "gpt-4o-2024-11-20",
    "gpt_4o_mini": "gpt-4o-mini",
    "gpt_4o_mini_2024_07_18": "gpt-4o-mini-2024-07-18",
    "o3_2025_04_16": "o3-2025-04-16",
    "o3_mini": "o3-mini",
    "o3_mini_2025_01_31": "o3-mini-2025-01-31",
    "o4_mini": "o4-mini",
    "o4_mini_2025_04_16": "o4-mini-2025-04-16",
    "gpt_5": "gpt-5",
    "gpt_5_mini": "gpt-5-mini",
    "gpt_5_nano": "gpt-5-nano",
}

# Fallback list prices, USD per 1K tokens (input, output). Used ONLY for budget
# estimation/projection when the proxy does not report a per-call cost and the
# TDM ``model_pricing`` table is unavailable. These are approximate — override
# exactly with --price-in/--price-out, or rely on the proxy's measured cost.
FALLBACK_PRICE_PER_1K = {
    "gpt-4o": (0.0025, 0.0100),
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4.1": (0.0020, 0.0080),
    "gpt-4.1-mini": (0.00040, 0.00160),
    "gpt-4.1-nano": (0.00010, 0.00040),
    "o3": (0.0020, 0.0080),
    "o3-mini": (0.00110, 0.00440),
    "o4-mini": (0.00110, 0.00440),
    "gpt-5": (0.00125, 0.01000),
    "gpt-5-mini": (0.00025, 0.00200),
    "gpt-5-nano": (0.00005, 0.00040),
}


def canonical_model(name: str) -> str:
    """Normalize a user-supplied model name to the proxy's canonical id."""
    name = name.strip()
    return MODEL_ALIASES.get(name, MODEL_ALIASES.get(name.replace("-", "_"), name))


def is_reasoning_model(model: str) -> bool:
    """o-series and gpt-5 (but NOT gpt-5-chat) are reasoning models: temperature
    must be 1, max_tokens >= 16000, and DSPy maps max_tokens ->
    max_completion_tokens. This mirrors dspy.LM's own detection so we never trip
    its validation."""
    m = model.split("/")[-1].lower()
    if m.startswith("gpt-5") and not m.startswith("gpt-5-chat"):
        return True
    return bool(re.match(r"^o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?$", m))


def _base_model_key(model: str) -> str:
    """Strip a trailing -YYYY-MM-DD date suffix for pricing lookups."""
    parts = model.split("-")
    if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit() and parts[-1].isdigit():
        return "-".join(parts[:-3])
    return model


# Optional TDM pricing table (./scripts/model_pricing.py). Imported best-effort.
_TDM_PRICING = None
_TDM_METRIC = None


def _load_tdm_pricing() -> None:
    global _TDM_PRICING, _TDM_METRIC
    for p in ("./scripts", str(PROJECT_ROOT / "scripts"), "."):
        if p not in sys.path:
            sys.path.append(p)
    try:
        from model_pricing import MODEL_PRICING, METRIC  # type: ignore

        _TDM_PRICING, _TDM_METRIC = MODEL_PRICING, METRIC
        print(f"[pricing] loaded TDM model_pricing ({len(MODEL_PRICING)} entries, METRIC={METRIC!r})")
    except Exception as e:  # not on TDM, or different layout — fall back silently
        print(f"[pricing] TDM model_pricing not available ({type(e).__name__}); using fallback table")


def _fallback_price_per_1k(model: str):
    for key in (model, _base_model_key(model)):
        if FALLBACK_PRICE_PER_1K.get(key):
            return FALLBACK_PRICE_PER_1K[key]
    return None


def _tdm_price_per_1k(model: str):
    """Interpret the TDM ``model_pricing`` table -> (in, out) USD per 1K tokens.

    The table's units are unknown to us; ``METRIC`` is the token unit prices are
    quoted per (observed METRIC=1000 => prices are per 1K tokens). We sanity-check
    the result against the built-in fallback and ignore it if the magnitude looks
    wrong (a unit mismatch must never silently inflate the budget).
    """
    if not _TDM_PRICING:
        return None
    metric = float(_TDM_METRIC or 1000)
    for key in (model, _base_model_key(model), model.replace("-", "_"),
                _base_model_key(model).replace("-", "_")):
        entry = _TDM_PRICING.get(key)
        if entry is None:
            continue
        if isinstance(entry, dict):
            pin = entry.get("input", entry.get("prompt", entry.get("in")))
            pout = entry.get("output", entry.get("completion", entry.get("out", pin)))
        elif isinstance(entry, (list, tuple)) and entry:
            pin, pout = entry[0], entry[-1]
        else:
            pin = pout = entry
        try:
            scale = 1000.0 / metric
            per_in, per_out = float(pin) * scale, float(pout) * scale
        except (TypeError, ValueError):
            continue
        if per_in < 0 or per_out < 0 or (per_in + per_out) == 0:
            continue
        fb = _fallback_price_per_1k(model)
        if fb:  # distrust a wildly different magnitude (likely a unit mismatch)
            ratio = (per_in + per_out) / (fb[0] + fb[1] or 1e-9)
            if ratio > 50 or ratio < 0.02:
                print(f"[pricing] WARN: TDM price for {model} looks off "
                      f"(~{ratio:.1f}x fallback); using fallback. Override with --price-in/out.")
                return fb
        return (per_in, per_out)
    return None


def resolve_price_per_1k(model: str, args) -> tuple[float, float]:
    """(input, output) USD per 1K tokens. Precedence: CLI override > TDM
    model_pricing table > built-in fallback > generic default."""
    if args.price_in is not None and args.price_out is not None:
        return args.price_in, args.price_out
    tdm = _tdm_price_per_1k(model)
    if tdm is not None:
        return tdm
    fb = _fallback_price_per_1k(model)
    if fb is not None:
        return fb
    # Unknown model: assume a mid/cheap price so projections aren't wildly off.
    return (0.0005, 0.0015)


# ============================================================================
# Token / cost accounting
# ============================================================================
_ENC = None


def approx_tokens(text: str) -> int:
    """Token estimate via tiktoken; falls back to a chars/4 heuristic."""
    global _ENC
    if not text:
        return 0
    try:
        if _ENC is None:
            import tiktoken

            try:
                _ENC = tiktoken.get_encoding("o200k_base")
            except Exception:
                _ENC = tiktoken.get_encoding("cl100k_base")
        return len(_ENC.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def entry_usage(entry: dict) -> tuple[int, int, float]:
    """(prompt_tokens, completion_tokens, measured_cost) from one lm.history row."""
    usage = entry.get("usage") or {}
    ptok = int(usage.get("prompt_tokens") or 0)
    ctok = int(usage.get("completion_tokens") or 0)
    cost = entry.get("cost")
    cost = float(cost) if isinstance(cost, (int, float)) else 0.0
    if ptok == 0 and ctok == 0:  # proxy didn't report usage — estimate it
        prompt_text = ""
        for msg in entry.get("messages") or []:
            if isinstance(msg, dict):
                prompt_text += str(msg.get("content", "")) + "\n"
        prompt_text = prompt_text or str(entry.get("prompt") or "")
        resp_text = "\n".join(str(o) for o in (entry.get("outputs") or []))
        ptok, ctok = approx_tokens(prompt_text), approx_tokens(resp_text)
    return ptok, ctok, cost


def cost_from_tokens(model: str, ptok: int, ctok: int, args) -> float:
    pin, pout = resolve_price_per_1k(model, args)
    return (ptok / 1000.0) * pin + (ctok / 1000.0) * pout


def history_usage(lm, start: int, model: str, args) -> tuple[int, int, float]:
    """Sum usage over lm.history[start:]; prefer measured cost, else estimate."""
    ptok = ctok = 0
    measured = 0.0
    has_measured = False
    for entry in lm.history[start:]:
        p, c, k = entry_usage(entry)
        ptok += p
        ctok += c
        if k > 0:
            measured += k
            has_measured = True
    cost = measured if has_measured else cost_from_tokens(model, ptok, ctok, args)
    return ptok, ctok, cost


def flatten_usage(usage) -> tuple[int, int, float]:
    """Pull (prompt_tokens, completion_tokens, cost) out of pred.get_lm_usage()."""
    ptok = ctok = 0
    cost = 0.0
    if isinstance(usage, dict):
        for v in usage.values():
            if isinstance(v, dict):
                ptok += int(v.get("prompt_tokens") or 0)
                ctok += int(v.get("completion_tokens") or 0)
                c = v.get("cost")
                if isinstance(c, (int, float)):
                    cost += float(c)
    return ptok, ctok, cost


# ============================================================================
# Persistent daily usage ledger
# ============================================================================
# Re-read on every run so the $/day cap holds across restarts/crashes within the
# same calendar day. Phase A spend lives in this ledger; Phase B spend is read
# straight from the per-row predictions JSONL (each row carries its cost + ts).
DEFAULT_LEDGER = ARTIFACTS_DIR / "usage_ledger.jsonl"


def _today() -> str:
    return dt.date.today().isoformat()


def record_run_usage(ledger_path: Path, phase: str, model: str, rows, ptok, ctok, cost):
    """Append one timestamped accounting line for a phase of this run."""
    try:
        Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": dt.datetime.now().isoformat(timespec="seconds"),
                "day": _today(), "phase": phase, "model": model, "rows": rows,
                "prompt_tokens": int(ptok), "completion_tokens": int(ctok),
                "cost": round(float(cost), 6),
            }) + "\n")
    except Exception as e:
        print(f"[ledger] WARN: could not write {ledger_path}: {e}")


def _sum_today_cost(path: Path, where=None, day=None) -> float:
    """Sum the 'cost' field of today's records in a JSONL file."""
    day = day or _today()
    total = 0.0
    if not Path(path).exists():
        return 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec_day = rec.get("day") or str(rec.get("ts", ""))[:10]
            if rec_day != day or (where and not where(rec)):
                continue
            c = rec.get("cost")
            if isinstance(c, (int, float)):
                total += float(c)
    return total


def today_spend(args) -> tuple[float, float, float]:
    """(phaseA, phaseB, total) USD already spent TODAY across all runs."""
    a = _sum_today_cost(args.ledger, where=lambda r: str(r.get("phase", "")).startswith("A"))
    b = _sum_today_cost(args.out)  # predictions JSONL: per-row cost, day from ts
    return a, b, a + b


# ============================================================================
# The classifier signature (one call -> decision + all requested columns)
# ============================================================================
class OntarioFloodClassifier(dspy.Signature):
    """Decide whether a historical Canadian newspaper extract is evidence of a
    REAL flood event that OCCURRED IN ONTARIO, Canada.

    The text is noisy OCR from 1800s-1900s newspapers; read past obvious OCR
    errors (e.g. "fl ood", "tlie", "rivor").

    RECALL IS MORE IMPORTANT THAN PRECISION. Missing a genuine Ontario flood is
    far worse than wrongly flagging a borderline article. When the text
    plausibly refers to a real flood in Ontario but is ambiguous (vague place,
    broken OCR, brief mention in an obituary/anniversary piece), LEAN TOWARD
    is_ontario_flood = True. Only choose False when you are reasonably confident
    it is NOT a real flood, or the flood clearly happened outside Ontario.

    Definitions:
    - Real flood = actual water overflowing onto normally dry land: river or
      lake overflow, ice-jam flooding, flash flood, dam/dike break, spring
      freshet/thaw, storm surge, or storm/urban/basement flooding from a real
      event (past, present, imminent, or remembered). NOT a real flood:
      metaphor ("flood of letters/applications"), flood-light or flood-insurance
      ads with no event, purely hypothetical/general discussion, or artificial
      controlled water releases that caused no flooding.
    - Ontario = the flood physically happened at an Ontario place (city, town,
      river, lake, or watershed). A flood elsewhere that merely appears in an
      Ontario newspaper is NOT an Ontario flood.
    """

    article_text: str = dspy.InputField(desc="OCR extract of a single newspaper article")

    is_ontario_flood: bool = dspy.OutputField(
        desc="True if this extract is evidence of a real flood that occurred in "
        "Ontario. If genuinely unsure but a real Ontario flood is plausible, "
        "choose True (favor recall)."
    )
    decision: str = dspy.OutputField(
        desc="Exactly one of: ontario_flood | flood_not_ontario | not_flood"
    )
    reason: str = dspy.OutputField(
        desc="ONE specific sentence justifying the decision using concrete clues "
        "from THIS text (named place, water body, what happened). Not generic."
    )
    flood_location: str = dspy.OutputField(
        desc="Where the flood occurred. For flood_not_ontario give the place/"
        "region (city/province/country). For ontario_flood give the Ontario "
        "place if named else 'Ontario'. For not_flood output 'n/a'."
    )
    not_flood_reason: str = dspy.OutputField(
        desc="If decision is not_flood, the short category why, one of: metaphor "
        "| not_specific_event | artificial | flood_light_or_ad | hypothetical | "
        "other, plus a few words. Otherwise output 'n/a'."
    )
    flood_type: str = dspy.OutputField(
        desc="If a real flood, its type: river | lake | flash | ice_jam | "
        "dam_break | spring_freshet | storm_surge | urban_storm | coastal | "
        "unknown. If not_flood output 'n/a'."
    )


OUTPUT_FIELDS = [
    "is_ontario_flood",
    "decision",
    "reason",
    "flood_location",
    "not_flood_reason",
    "flood_type",
]


# ============================================================================
# Adapter: force a single json_object call per row
# ============================================================================
class JsonObjectAdapter(dspy.JSONAdapter):
    """Always do ONE call with ``response_format={"type":"json_object"}``.

    DSPy's default path tries OpenAI *structured outputs* (a json_schema
    response_format) first; the TDM proxy rejects that, so DSPy logs
    "Failed to use structured output format, falling back to JSON mode." and
    RETRIES — doubling latency on every one of the 90k rows. Plain json_object
    mode is exactly what the previous TDM script used and the proxy supports it,
    so we skip the schema attempt (and the ChatAdapter->JSONAdapter fallback)
    entirely by calling the base adapter directly with json_object forced on.
    """

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        lm_kwargs = dict(lm_kwargs)
        lm_kwargs["response_format"] = {"type": "json_object"}
        return Adapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        lm_kwargs = dict(lm_kwargs)
        lm_kwargs["response_format"] = {"type": "json_object"}
        return await Adapter.acall(self, lm, lm_kwargs, signature, demos, inputs)


def configure_dspy():
    """Global DSPy config: usage tracking + the single-call json_object adapter."""
    dspy.configure(track_usage=True, adapter=JsonObjectAdapter())


# ============================================================================
# LM construction (injectable so --self-test can swap in a DummyLM)
# ============================================================================
_LM_FACTORY = None  # set by self-test to bypass the real proxy


def read_api_key(args) -> str:
    if args.api_key:
        return args.api_key
    path = args.token_file
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception as e:
        raise SystemExit(
            f"Could not read TDM API token from {path} ({e}). "
            f"Pass --api-key or --token-file."
        )


def build_lm(model: str, args, api_key: str | None = None) -> dspy.LM:
    """Construct a dspy.LM pointed at the TDM OpenAI-compatible proxy."""
    if _LM_FACTORY is not None:  # self-test injection
        return _LM_FACTORY(model)

    kwargs = dict(
        model=f"openai/{model}",
        api_base=args.base_url,
        api_key=api_key,
        cache=not args.no_cache,
        num_retries=args.num_retries,
    )
    if is_reasoning_model(model):
        # o-series / gpt-5 reject temperature!=1 and need headroom for hidden
        # reasoning tokens; DSPy converts max_tokens -> max_completion_tokens.
        # reasoning_effort='minimal' keeps gpt-5 fast+cheap enough for 90k rows.
        kwargs["temperature"] = 1.0
        kwargs["max_tokens"] = max(args.max_tokens, 16000)
        effort = args.reasoning_effort
        if effort == "minimal" and not model.lower().startswith("gpt-5"):
            effort = "low"  # o-series supports low/medium/high, not 'minimal'
        if effort and effort != "none":
            kwargs["reasoning_effort"] = effort
    else:
        kwargs["temperature"] = args.temperature
        kwargs["max_tokens"] = args.max_tokens
    return dspy.LM(**kwargs)


def make_program(strategy: str) -> dspy.Module:
    """A fresh, uncompiled program for the given prompt strategy."""
    if strategy == "cot":
        return dspy.ChainOfThought(OntarioFloodClassifier)
    return dspy.Predict(OntarioFloodClassifier)


# ============================================================================
# Data loading + splitting
# ============================================================================
_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f", ""}


def to_bool(value) -> bool:
    v = str(value).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a binary label")


def load_annotations(csv_path: Path) -> list[dict]:
    """Read annotations_so_far.csv -> rows with text + is_ontario_flood target.

    The optimization target is is_ontario_flood = (flood AND ontario): the thing
    we ultimately must not lose.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Labelled data not found at {csv_path}. Expected columns: "
            "id, date, example, changed, flood, ontario"
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        missing = {"example", "flood", "ontario"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required column(s): {sorted(missing)}")
        for r in reader:
            text = (r.get("example") or "").strip()
            if not text:
                continue
            flood = to_bool(r.get("flood"))
            ontario = to_bool(r.get("ontario"))
            rows.append(
                {
                    "id": (r.get("id") or "").strip(),
                    "text": text[:MAX_TEXT_CHARS],
                    "flood": flood,
                    "ontario": ontario,
                    "is_ontario_flood": bool(flood and ontario),
                }
            )
    if not rows:
        raise ValueError(f"No usable rows in {csv_path}")
    return rows


def to_examples(rows: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(article_text=r["text"], is_ontario_flood=r["is_ontario_flood"]).with_inputs(
            "article_text"
        )
        for r in rows
    ]


def stratified_split(rows: list[dict], eval_frac: float, seed: int):
    """Split rows into (train, eval) stratified on is_ontario_flood."""
    rng = random.Random(seed)
    pos = [r for r in rows if r["is_ontario_flood"]]
    neg = [r for r in rows if not r["is_ontario_flood"]]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cut(lst):
        n = int(round(len(lst) * eval_frac))
        n = min(max(n, 1), len(lst) - 1) if len(lst) >= 2 else 0
        return lst[n:], lst[:n]

    tr_p, ev_p = cut(pos)
    tr_n, ev_n = cut(neg)
    train, ev = tr_p + tr_n, ev_p + ev_n
    rng.shuffle(train)
    rng.shuffle(ev)
    return train, ev


# ============================================================================
# Recall-first metric + evaluation
# ============================================================================
def recall_metric(example, pred, trace=None):
    """Recall-first scoring.

    During bootstrapping (trace set) keep only exactly-correct demos. During
    evaluation, reward correctness but make a FALSE NEGATIVE (a missed Ontario
    flood) cost the most and tolerate FALSE POSITIVES — i.e. recall over
    precision.
    """
    y = bool(example.is_ontario_flood)
    p = bool(getattr(pred, "is_ontario_flood", False))
    correct = p == y
    if trace is not None:
        return correct
    if correct:
        return 1.0
    return 0.0 if (y and not p) else 0.6  # missed flood worst; over-include ok


def prf(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    b2 = 4  # beta^2 with beta=2 (weights recall 2x precision)
    f2 = (1 + b2) * precision * recall / (b2 * precision + recall) if (b2 * precision + recall) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return precision, recall, f1, f2, acc


def evaluate_program(program, examples, model, args) -> dict:
    """Run program over examples (threaded) and compute recall-first metrics."""
    lm = program.get_lm()
    start = len(lm.history)
    preds: dict[int, object] = {}
    latencies = []

    def work(i_ex):
        i, ex = i_ex
        t0 = time.perf_counter()
        try:
            out = program(article_text=ex.article_text)
            ok = bool(getattr(out, "is_ontario_flood", False))
        except Exception as e:
            ok = False  # parse/timeout -> treat as negative for the matrix
            out = None
            if args.verbose:
                print(f"    [eval] row {i} error: {type(e).__name__}: {e}")
        return i, ok, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=args.eval_workers) as pool:
        for i, ok, lat in pool.map(work, list(enumerate(examples))):
            preds[i] = ok
            latencies.append(lat)

    tp = fp = fn = tn = 0
    score_sum = 0.0
    false_neg_idx = []
    for i, ex in enumerate(examples):
        y = bool(ex.is_ontario_flood)
        p = bool(preds.get(i, False))
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif (not p) and y:
            fn += 1
            false_neg_idx.append(i)
        else:
            tn += 1
        score_sum += 1.0 if p == y else (0.0 if (y and not p) else 0.6)

    precision, recall, f1, f2, acc = prf(tp, fp, fn, tn)
    n = len(examples)
    ptok, ctok, cost = history_usage(lm, start, model, args)
    return {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "f2": f2,
        "accuracy": acc,
        "recall_score": score_sum / n if n else 0.0,
        "false_negatives": len(false_neg_idx),
        "prompt_tokens": ptok, "completion_tokens": ctok,
        "eval_cost": cost,
        "cost_per_1k_rows": (cost / n * 1000.0) if n else 0.0,
        "avg_tokens_per_row": ((ptok + ctok) / n) if n else 0.0,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
    }


# ============================================================================
# Phase A: optimize, evaluate candidates, pick the best prompt
# ============================================================================
def compile_fewshot(base_program, trainset, args):
    """BootstrapFewShot compile. max_labeled_demos=0 so every demo carries ALL
    output fields (filled by the teacher), keeping the output format consistent.
    """
    optimizer = dspy.BootstrapFewShot(
        metric=recall_metric,
        max_bootstrapped_demos=args.max_demos,
        max_labeled_demos=0,
        max_rounds=args.max_rounds,
    )
    return optimizer.compile(base_program, trainset=trainset)


def phase_optimize(args, api_key, remaining_budget):
    rows = load_annotations(args.annotations)
    n_pos = sum(r["is_ontario_flood"] for r in rows)
    print(
        f"\n[Phase A] loaded {len(rows)} labelled rows "
        f"({n_pos} Ontario floods / {len(rows) - n_pos} not).  "
        f"flood={sum(r['flood'] for r in rows)}  ontario={sum(r['ontario'] for r in rows)}"
    )
    train_rows, eval_rows = stratified_split(rows, args.eval_frac, args.seed)
    train_ex, eval_ex = to_examples(train_rows), to_examples(eval_rows)
    ep = sum(e.is_ontario_flood for e in eval_ex)
    print(
        f"[Phase A] train={len(train_ex)}  eval={len(eval_ex)} "
        f"(eval positives={ep})  strategies={args.strategies}"
    )

    # Hard requirement: the chosen model must label the WHOLE corpus within the
    # remaining daily budget (so we can do all ~{corpus_rows} rows in one day).
    afford_cap = remaining_budget * args.budget_safety
    opt_cap = args.optimize_budget if args.optimize_budget else min(5.0, remaining_budget * 0.25)
    print(
        f"[Phase A] must fit {args.corpus_rows:,} rows in ${afford_cap:.2f} "
        f"(=> <= ${afford_cap/args.corpus_rows*1000:.3f}/1k rows).  "
        f"optimization spend cap ≈ ${opt_cap:.2f}."
    )

    models = [canonical_model(m) for m in args.models]
    candidates = []
    tot_ptok = tot_ctok = 0
    spent = 0.0

    for model in models:
        if spent >= opt_cap:
            print(f"[Phase A] optimization budget hit (${spent:.2f}); skipping remaining models.")
            break
        # Build + probe the model so an unavailable one (e.g. gpt-5 not yet
        # enabled for this account) is skipped instead of killing the night.
        try:
            lm = build_lm(model, args, api_key)
            probe = make_program("predict")
            probe.set_lm(lm)
            start = len(lm.history)
            _ = probe(article_text="The Grand River overflowed its banks at Galt, Ontario, flooding homes.")
            pt, ct, c = history_usage(lm, start, model, args)
            spent += c; tot_ptok += pt; tot_ctok += ct
            print(f"\n[Phase A] model '{model}' OK (probe cost ${c:.4f}).")
        except Exception as e:
            print(f"\n[Phase A] SKIP model '{model}': {type(e).__name__}: {str(e)[:160]}")
            continue

        for strat in args.strategies:
            if spent >= opt_cap:
                print(f"  [Phase A] budget hit; skipping '{strat}' for {model}.")
                break
            tag = f"{model} :: {strat}"
            try:
                lm = build_lm(model, args, api_key)
                base = make_program("cot" if strat == "cot" else "predict")
                base.set_lm(lm)

                comp_cost = 0.0
                if strat == "fewshot":
                    comp_start = len(lm.history)
                    with dspy.context(lm=lm):
                        program = compile_fewshot(base, train_ex, args)
                    program.set_lm(lm)
                    pt, ct, comp_cost = history_usage(lm, comp_start, model, args)
                    spent += comp_cost; tot_ptok += pt; tot_ctok += ct
                else:  # zeroshot or cot, used as-is
                    program = base

                metrics = evaluate_program(program, eval_ex, model, args)
                spent += metrics["eval_cost"]
                tot_ptok += metrics["prompt_tokens"]; tot_ctok += metrics["completion_tokens"]
                full_cost = metrics["cost_per_1k_rows"] / 1000.0 * args.corpus_rows
                metrics.update({
                    "model": model, "strategy": strat, "tag": tag,
                    "compile_cost": comp_cost, "program": program,
                    "full_corpus_cost": full_cost,
                    "affordable": full_cost <= afford_cap,
                })
                candidates.append(metrics)
                print(
                    f"  [{tag:<26}] R={metrics['recall']:.3f} P={metrics['precision']:.3f} "
                    f"F2={metrics['f2']:.3f} FN={metrics['fn']} "
                    f"${metrics['cost_per_1k_rows']:.3f}/1k "
                    f"corpus=${full_cost:.2f} {'OK' if metrics['affordable'] else 'TOO$'}"
                )
            except Exception as e:
                print(f"  [{tag}] FAILED: {type(e).__name__}: {str(e)[:160]}")

    if not candidates:
        raise SystemExit("[Phase A] no candidate produced metrics — check proxy/models.")

    # Rank: affordable-for-the-whole-corpus FIRST, then recall-first (F2, recall),
    # then cheaper. This guarantees we pick a model that can do all the rows today.
    candidates.sort(
        key=lambda m: (m["affordable"], m["f2"], m["recall"], -m["cost_per_1k_rows"]),
        reverse=True,
    )
    if any(m["affordable"] for m in candidates):
        best = candidates[0]
    else:
        # Nothing fits the whole corpus in budget: pick the cheapest option with
        # at least usable recall so we still cover as many rows as possible.
        print("[Phase A] WARNING: no model can label the full corpus within budget; "
              "choosing the cheapest decent-recall option. Use a cheaper model, raise "
              "--budget, or lower --corpus-rows to fit the full corpus in one day.")
        pool = [m for m in candidates if m["recall"] >= 0.6] or candidates
        best = min(pool, key=lambda m: m["full_corpus_cost"])

    write_report(candidates, train_ex, eval_ex, args, remaining_budget)
    record_run_usage(args.ledger, "A", best["model"], len(candidates), tot_ptok, tot_ctok, spent)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    best["program"].save(str(ARTIFACTS_DIR / "best_program.json"))
    with open(ARTIFACTS_DIR / "best_program.meta.json", "w") as f:
        json.dump(
            {"day": _today(), "model": best["model"], "strategy": best["strategy"],
             "f2": best["f2"], "recall": best["recall"], "precision": best["precision"],
             "cost_per_1k_rows": best["cost_per_1k_rows"],
             "full_corpus_cost": best["full_corpus_cost"], "affordable": best["affordable"]},
            f, indent=2,
        )
    args._best_cost_per_row = best["cost_per_1k_rows"] / 1000.0  # for Phase B sizing
    print(
        f"\n[Phase A] BEST = {best['tag']}  (recall={best['recall']:.3f}, F2={best['f2']:.3f}, "
        f"~${best['full_corpus_cost']:.2f} for {args.corpus_rows:,} rows).  "
        f"Phase A spend ≈ ${spent:.4f}"
    )
    return best["program"], best["model"], spent


def write_report(candidates, train_ex, eval_ex, args, remaining_budget):
    """Print + persist evaluation metrics for the top-3 prompts (and full board)."""
    top3 = candidates[:3]
    afford_cap = remaining_budget * args.budget_safety
    print("\n" + "=" * 78)
    print(f"TOP 3 PROMPTS  (recall-first, must fit {args.corpus_rows:,} rows in ${afford_cap:.0f})")
    print("=" * 78)
    for rank, m in enumerate(top3, 1):
        fits = "fits budget" if m["affordable"] else "OVER BUDGET for full corpus"
        print(
            f"#{rank}  {m['tag']}\n"
            f"     recall={m['recall']:.3f}  precision={m['precision']:.3f}  "
            f"F1={m['f1']:.3f}  F2={m['f2']:.3f}  acc={m['accuracy']:.3f}\n"
            f"     confusion: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}  "
            f"(missed floods={m['fn']})\n"
            f"     ~{m['avg_tokens_per_row']:.0f} tok/row  ${m['cost_per_1k_rows']:.3f}/1k rows  "
            f"{m['avg_latency_s']:.2f}s/row  |  full corpus ≈ ${m['full_corpus_cost']:.2f} ({fits})"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    serial = [{k: v for k, v in m.items() if k != "program"} for m in candidates]
    with open(ARTIFACTS_DIR / "prompt_eval_report.json", "w") as f:
        json.dump(
            {"eval_size": len(eval_ex), "train_size": len(train_ex),
             "corpus_rows": args.corpus_rows, "afford_cap": afford_cap,
             "ranking": "affordable-for-full-corpus first, then F2 desc, recall desc, cost asc",
             "candidates": serial},
            f, indent=2,
        )

    lines = [
        "# Prompt optimization report (recall-first, budget-aware)",
        "",
        f"- Train rows: {len(train_ex)}  |  Held-out eval rows: {len(eval_ex)}",
        "- Target: `is_ontario_flood = flood AND ontario`",
        f"- Constraint: label the full **{args.corpus_rows:,}-row** corpus within "
        f"**${afford_cap:.0f}** (one day).",
        "- Ranking: **fits-budget first**, then **F2** (recall-weighted) → recall → cheaper.",
        "- Note: prompts are *selected* on this same held-out set, so the winner's",
        "  numbers are mildly optimistic; treat them as comparative.",
        "",
        "## Top 3 prompts",
        "",
        "| # | model | strategy | recall | precision | F1 | F2 | missed | $/1k rows | full corpus $ | fits? |",
        "|---|-------|----------|-------:|----------:|---:|---:|------:|----------:|-------------:|:-----:|",
    ]
    for rank, m in enumerate(top3, 1):
        lines.append(
            f"| {rank} | {m['model']} | {m['strategy']} | {m['recall']:.3f} | "
            f"{m['precision']:.3f} | {m['f1']:.3f} | {m['f2']:.3f} | {m['fn']} | "
            f"${m['cost_per_1k_rows']:.3f} | ${m['full_corpus_cost']:.2f} | "
            f"{'✅' if m['affordable'] else '❌'} |"
        )
    lines += ["", "## Full leaderboard", "",
              "| model | strategy | recall | precision | F2 | missed | $/1k rows | full corpus $ | fits? |",
              "|-------|----------|-------:|----------:|---:|------:|----------:|-------------:|:-----:|"]
    for m in candidates:
        lines.append(
            f"| {m['model']} | {m['strategy']} | {m['recall']:.3f} | {m['precision']:.3f} | "
            f"{m['f2']:.3f} | {m['fn']} | ${m['cost_per_1k_rows']:.3f} | "
            f"${m['full_corpus_cost']:.2f} | {'✅' if m['affordable'] else '❌'} |"
        )
    (ARTIFACTS_DIR / "prompt_eval_report.md").write_text("\n".join(lines) + "\n")
    print(f"\n[Phase A] wrote {ARTIFACTS_DIR/'prompt_eval_report.md'} and .json")


# ============================================================================
# Phase B: label a budget-sized subset of the unlabelled corpus
# ============================================================================
def load_processed_ids(out_path: Path) -> set:
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


def phase_run(program, model, spent_so_far, args, api_key):
    if not args.extracts.exists():
        raise SystemExit(f"[Phase B] extracts CSV not found: {args.extracts}")

    # Make sure the program has a usable LM (it does after Phase A; rebuild when
    # reused from disk).
    try:
        program.get_lm()
    except Exception:
        program.set_lm(build_lm(model, args, api_key))

    out_jsonl = args.out
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed = set() if args.no_resume else load_processed_ids(out_jsonl)
    if processed:
        print(f"[Phase B] resume: {len(processed)} rows already done in {out_jsonl}")

    # ---- size the subset to the remaining budget -------------------------
    remaining = max(0.0, args.budget - spent_so_far)
    # expected per-row cost from Phase A measurement if present, else a guess
    per_row = getattr(args, "_best_cost_per_row", 0.0)
    if per_row <= 0:
        per_row = cost_from_tokens(model, 900, 180, args)  # rough default
    affordable = int((remaining * args.budget_safety) / per_row) if per_row > 0 else 10**9
    corpus_left = max(0, args.corpus_rows - len(processed))

    if args.ignore_budget:
        # --continue-run --ignore-budget: push through the whole remainder in one
        # go (the daily cap is intentionally bypassed). Size to the corpus, not $:
        # sweep every unprocessed CSV row (the resume-skip handles what's done).
        n_target = args.max_rows if args.max_rows is not None else 10**9
        est_rows = corpus_left if corpus_left > 0 else 0
        if args.max_rows is not None:
            est_rows = min(est_rows, args.max_rows) if est_rows else args.max_rows
        target_str = f"{n_target:,}" if args.max_rows is not None else "all remaining rows"
        print(
            f"\n[Phase B] model={model}  --ignore-budget (daily ${args.budget:.2f} cap bypassed)\n"
            f"[Phase B] est ${per_row*1000:.3f}/1k rows; corpus left ≈ {corpus_left:,}; "
            f"target this run = {target_str}; est cost ≈ ${per_row*est_rows:,.2f}."
        )
    else:
        n_target = affordable
        if args.max_rows is not None:
            n_target = min(n_target, args.max_rows)
        print(
            f"\n[Phase B] model={model}  budget=${args.budget:.2f}  spent=${spent_so_far:.4f}  "
            f"remaining=${remaining:.4f}\n"
            f"[Phase B] est ${per_row*1000:.3f}/1k rows -> can afford ~{affordable:,} rows; "
            f"target this run = {n_target:,} (cap --max-rows={args.max_rows})."
        )
    if n_target <= 0:
        print("[Phase B] no budget left for inference; stopping.")
        return

    # commit() runs only on the main thread (inside the as_completed loop), so
    # the shared counters/file need no lock; workers only run classify().
    run_cost = 0.0
    stats = {"rows": 0, "ontario": 0, "flood_not_on": 0, "not_flood": 0, "errors": 0,
             "ptok": 0, "ctok": 0, "seconds": 0.0}
    stop = threading.Event()

    def classify(row):
        if stop.is_set():
            return None
        rid = str(row.get(args.id_col, "")).strip()
        text = (row.get(args.text_col) or "").strip()
        if not text:
            return None
        text = text[:MAX_TEXT_CHARS]
        t0 = time.perf_counter()
        rec = {"id": rid, "date": row.get(args.date_col, ""), "model": model}
        try:
            pred = program(article_text=text)
            for fld in OUTPUT_FIELDS:
                rec[fld] = getattr(pred, fld, None)
            rec["is_ontario_flood"] = bool(getattr(pred, "is_ontario_flood", False))
            ptok, ctok, cost = flatten_usage(pred.get_lm_usage())
            if ptok == 0 and ctok == 0:  # proxy didn't report -> estimate
                ptok = approx_tokens(text) + 350  # + instruction/demos overhead
                ctok = approx_tokens(" ".join(str(rec.get(f, "")) for f in OUTPUT_FIELDS))
            if cost <= 0:
                cost = cost_from_tokens(model, ptok, ctok, args)
            rec["status"] = "ok"
        except Exception as e:
            rec.update({f: None for f in OUTPUT_FIELDS})
            rec["is_ontario_flood"] = ""  # unknown -> keep for manual review (recall!)
            rec["decision"] = "error"
            rec["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
            ptok = ctok = 0
            cost = 0.0
            rec["status"] = "error"
        rec["prompt_tokens"] = ptok
        rec["completion_tokens"] = ctok
        rec["cost"] = round(cost, 6)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        rec["ts"] = dt.datetime.now().isoformat(timespec="seconds")  # for daily spend
        return rec

    def commit(rec, fout):
        nonlocal run_cost
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        processed.add(rec["id"])
        stats["rows"] += 1
        stats["ptok"] += rec["prompt_tokens"]
        stats["ctok"] += rec["completion_tokens"]
        stats["seconds"] += rec["seconds"]
        run_cost += rec["cost"]
        d = rec.get("decision")
        if rec["status"] == "error":
            stats["errors"] += 1
        elif rec.get("is_ontario_flood") is True or d == "ontario_flood":
            stats["ontario"] += 1
        elif d == "flood_not_ontario":
            stats["flood_not_on"] += 1
        else:
            stats["not_flood"] += 1
        # hard budget guard (bypassed by --ignore-budget so a continue-run can
        # finish the whole remaining corpus across the daily cap)
        if (not args.ignore_budget) and spent_so_far + run_cost >= args.budget * 0.99:
            stop.set()

    # ---- stream rows, dispatch in bounded batches, checkpoint as we go ----
    t_start = time.perf_counter()
    with open(args.extracts, newline="", encoding="utf-8", errors="replace") as fin, \
            open(out_jsonl, "a", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        if args.text_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"[Phase B] --text-col '{args.text_col}' not in {reader.fieldnames}"
            )

        batch = []
        batch_size = max(1, min(args.workers * 4, n_target))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            def flush(batch):
                futs = [pool.submit(classify, r) for r in batch]
                for fu in as_completed(futs):
                    rec = fu.result()
                    if rec is not None:
                        commit(rec, fout)

            for row in reader:
                # Stop accumulating once committed + in-flight would reach the
                # target, so a single batch can't overshoot n_target (= budget).
                if stop.is_set() or stats["rows"] + len(batch) >= n_target:
                    break
                rid = str(row.get(args.id_col, "")).strip()
                if rid in processed:
                    continue
                batch.append(row)
                if len(batch) >= batch_size:
                    flush(batch)
                    batch = []
                    if stats["rows"] and stats["rows"] % args.log_every < batch_size:
                        print(
                            f"  ...{stats['rows']:,} done | ON={stats['ontario']} "
                            f"elsewhere={stats['flood_not_on']} not-flood={stats['not_flood']} "
                            f"err={stats['errors']} | ${run_cost:.3f}"
                        )
                    if stop.is_set():
                        break
            if batch and not stop.is_set():
                flush(batch)

    wall = time.perf_counter() - t_start
    write_predictions_csv(out_jsonl, args.out_csv, args)
    report_run(stats, run_cost, wall, spent_so_far, model, args)


def write_predictions_csv(jsonl_path: Path, csv_path: Path, args):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "date", "is_ontario_flood", "decision", "reason", "flood_location",
            "not_flood_reason", "flood_type", "model", "prompt_tokens",
            "completion_tokens", "cost", "status"]
    n = 0
    with open(jsonl_path, encoding="utf-8") as fin, \
            open(csv_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            w.writerow(json.loads(line))
            n += 1
    print(f"[Phase B] wrote {n:,} labelled rows -> {csv_path}")


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def report_run(stats, run_cost, wall, spent_before, model, args):
    n = stats["rows"]
    print("\n" + "=" * 78)
    print(f"PHASE B SUMMARY  ({n:,} rows labelled this run, model={model})")
    print("=" * 78)
    if n == 0:
        print("  nothing processed (all skipped / no budget).")
        return
    print(f"  Ontario floods            : {stats['ontario']:,}  ({stats['ontario']/n:.1%})")
    print(f"  floods elsewhere          : {stats['flood_not_on']:,}")
    print(f"  not a flood               : {stats['not_flood']:,}")
    print(f"  errors (kept for review)  : {stats['errors']:,}")
    print(f"  tokens                    : {stats['ptok']+stats['ctok']:,}")
    print(f"  wall time                 : {_hms(wall)}  ({wall/n:.2f}s/row, {args.workers} workers)")
    print(f"  this-run cost             : ${run_cost:.4f}")
    print(f"  today's total spend (A+B) : ${spent_before + run_cost:.4f} / ${args.budget:.2f} budget")
    per_row = run_cost / n
    total = args.corpus_rows
    print(f"  proj: full corpus {total:,} -> ${per_row*total:,.2f}, "
          f"{_hms((wall/n)*total)} wall (at {args.workers} workers)")
    done_est = len(load_processed_ids(args.out))
    left = max(0, total - done_est)
    print(f"  corpus labelled so far    : {done_est:,} / {total:,}  (remaining ~{left:,})")
    if left == 0:
        print("\n  ✅ ENTIRE CORPUS LABELLED.")
    else:
        print(f"\n  Re-run to continue (checkpoint = {args.out}); "
              f"~{left*per_row:.2f} more needed for the rest.")


# ============================================================================
# Detailed evaluation on the labelled set  (--test-models and --report)
# ============================================================================
# Phase A's evaluate_program() only keeps the boolean decision (enough to rank
# prompts). The benchmark + report modes need the model's full output per row
# (decision, reason, location, type) so they can show distributions and analyse
# the actual false positives / false negatives. run_on_labeled() does that.

GOLD_DECISION = {  # 3-way gold label derived from (flood, ontario)
    (True, True): "ontario_flood",
    (True, False): "flood_not_ontario",
    (False, True): "not_flood",   # ontario w/o flood is still not a flood
    (False, False): "not_flood",
}


def _clone_args(args, **overrides):
    """Shallow Namespace copy with field overrides (for per-config LM builds)."""
    a = copy.copy(args)
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _snippet(text: str, n: int = 320) -> str:
    """Collapse whitespace and truncate to n chars for compact report tables."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _md_escape(text: str) -> str:
    """Make a snippet safe to drop inside a markdown table cell."""
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def run_on_labeled(program, rows, model, args, max_workers=None):
    """Run the full classifier over labelled rows; return (metrics, per_row).

    Each per_row record carries the gold labels, the model's full prediction,
    its cost/latency, and a tp/fp/fn/tn ``category`` against the recall target
    ``is_ontario_flood``. Threaded like Phase A's evaluation.
    """
    lm = program.get_lm()
    workers = max_workers or args.eval_workers
    results: dict[int, dict] = {}

    def work(i_row):
        i, r = i_row
        t0 = time.perf_counter()
        rec = {
            "id": str(r.get("id", "")),
            "date": r.get("date", ""),
            "text": r["text"],
            "gold_flood": bool(r["flood"]),
            "gold_ontario": bool(r["ontario"]),
            "gold_is_ontario_flood": bool(r["is_ontario_flood"]),
            "gold_decision": GOLD_DECISION[(bool(r["flood"]), bool(r["ontario"]))],
            "model": model,
        }
        try:
            pred = program(article_text=r["text"])
            for fld in OUTPUT_FIELDS:
                rec[fld] = getattr(pred, fld, None)
            rec["is_ontario_flood"] = bool(getattr(pred, "is_ontario_flood", False))
            ptok, ctok, cost = flatten_usage(pred.get_lm_usage())
            if ptok == 0 and ctok == 0:  # proxy didn't report -> estimate
                ptok = approx_tokens(r["text"]) + 350
                ctok = approx_tokens(" ".join(str(rec.get(f, "")) for f in OUTPUT_FIELDS))
            if cost <= 0:
                cost = cost_from_tokens(model, ptok, ctok, args)
            rec["status"] = "ok"
        except Exception as e:
            for fld in OUTPUT_FIELDS:
                rec[fld] = None
            rec["is_ontario_flood"] = False  # parse/timeout -> negative for matrix
            rec["decision"] = "error"
            rec["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
            ptok = ctok = 0
            cost = 0.0
            rec["status"] = "error"
            if args.verbose:
                print(f"    [eval] row {i} error: {type(e).__name__}: {e}")
        rec["prompt_tokens"] = ptok
        rec["completion_tokens"] = ctok
        rec["cost"] = round(cost, 6)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        y, p = rec["gold_is_ontario_flood"], rec["is_ontario_flood"]
        rec["category"] = ("tp" if p and y else "fp" if p and not y else
                           "fn" if (not p) and y else "tn")
        return i, rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rec in pool.map(work, list(enumerate(rows))):
            results[i] = rec
    per_row = [results[i] for i in range(len(rows))]
    return summarize_labeled(per_row, model, args), per_row


def summarize_labeled(per_row, model, args) -> dict:
    """Confusion matrix + recall-first metrics + cost over labelled per-row recs."""
    cats = Counter(r["category"] for r in per_row)
    tp, fp, fn, tn = cats["tp"], cats["fp"], cats["fn"], cats["tn"]
    precision, recall, f1, f2, acc = prf(tp, fp, fn, tn)
    n = len(per_row)
    ptok = sum(int(r.get("prompt_tokens") or 0) for r in per_row)
    ctok = sum(int(r.get("completion_tokens") or 0) for r in per_row)
    cost = sum(float(r.get("cost") or 0.0) for r in per_row)
    secs = sum(float(r.get("seconds") or 0.0) for r in per_row)
    errors = sum(1 for r in per_row if r.get("status") == "error")
    full_cost = (cost / n * args.corpus_rows) if n else 0.0
    return {
        "model": model, "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "f2": f2, "accuracy": acc,
        "errors": errors,
        "prompt_tokens": ptok, "completion_tokens": ctok, "cost": cost,
        "cost_per_1k_rows": (cost / n * 1000.0) if n else 0.0,
        "avg_tokens_per_row": ((ptok + ctok) / n) if n else 0.0,
        "avg_latency_s": (secs / n) if n else 0.0,
        "full_corpus_cost": full_cost,
        "gold_dist": dict(Counter(r["gold_decision"] for r in per_row).most_common()),
        "pred_dist": dict(Counter((r.get("decision") or "?") for r in per_row).most_common()),
    }


def write_labeled_eval_jsonl(path: Path, per_row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in per_row:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_labeled_eval_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def labeled_eval_path(model: str, strategy: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{model}__{strategy}")
    return ARTIFACTS_DIR / f"labeled_eval__{safe}.jsonl"


def get_labeled_program(args, api_key):
    """Build the program used to predict on the 250 for the report.

    --report-strategy 'best' loads the compiled prompt that actually labels the
    corpus (artifacts/best_program.json); 'zeroshot'/'cot' build a fresh program
    on --report-model. Returns (program, model, strategy_label).
    """
    strat = args.report_strategy
    model = canonical_model(args.report_model) if args.report_model else None
    if strat == "best":
        meta_path = ARTIFACTS_DIR / "best_program.meta.json"
        prog_path = ARTIFACTS_DIR / "best_program.json"
        if prog_path.exists():
            meta = json.load(open(meta_path)) if meta_path.exists() else {}
            best_model = canonical_model(meta.get("model", model or args.models[0]))
            model = model or best_model
            strategy = meta.get("strategy", "fewshot")
            program = make_program("cot" if strategy == "cot" else "predict")
            program.load(str(prog_path))
            program.set_lm(build_lm(model, args, api_key))
            return program, model, f"best/{strategy} (compiled prompt in use)"
        print("[report] no saved best_program.json; falling back to a fresh zeroshot program.")
        strat = "zeroshot"
    model = model or canonical_model(args.models[0])
    program = make_program("cot" if strat == "cot" else "predict")
    program.set_lm(build_lm(model, args, api_key))
    return program, model, strat


# ============================================================================
# Reading what Phase B has already labelled (the ~14k done) + projections
# ============================================================================
def phaseB_progress(path: Path) -> dict:
    """Aggregate the per-row predictions JSONL: counts, distributions, measured
    cost + wall-time, so the report can fold in the rows already done."""
    agg = {
        "n": 0, "cost": 0.0, "seconds": 0.0, "ptok": 0, "ctok": 0, "errors": 0,
        "ontario": 0, "flood_not_on": 0, "not_flood": 0,
        "decisions": {}, "flood_types": {}, "locations": {}, "models": {}, "by_day": {},
    }
    if not Path(path).exists():
        return agg
    decisions, ftypes, locs, models, days = (Counter() for _ in range(5))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            agg["n"] += 1
            agg["cost"] += float(rec.get("cost") or 0.0)
            agg["seconds"] += float(rec.get("seconds") or 0.0)
            agg["ptok"] += int(rec.get("prompt_tokens") or 0)
            agg["ctok"] += int(rec.get("completion_tokens") or 0)
            d, st = rec.get("decision"), rec.get("status")
            decisions[d or "?"] += 1
            if m := rec.get("model"):
                models[m] += 1
            if ts := (rec.get("ts") or ""):
                days[ts[:10]] += 1
            ft = rec.get("flood_type")
            if ft and ft not in ("n/a", "", None):
                ftypes[ft] += 1
            if st == "error" or d == "error":
                agg["errors"] += 1
                continue
            if rec.get("is_ontario_flood") is True or d == "ontario_flood":
                agg["ontario"] += 1
            elif d == "flood_not_ontario":
                agg["flood_not_on"] += 1
                if loc := rec.get("flood_location"):
                    locs[_snippet(loc, 40)] += 1
            else:
                agg["not_flood"] += 1
    agg["decisions"] = dict(decisions.most_common())
    agg["flood_types"] = dict(ftypes.most_common())
    agg["locations"] = dict(locs.most_common(15))
    agg["models"] = dict(models.most_common())
    agg["by_day"] = dict(sorted(days.items()))
    return agg


def compute_projections(prog: dict, corpus_rows: int, workers: int) -> dict:
    """Project cost + wall-time for the remaining corpus from measured done rows.

    Per-row ``seconds`` is single-call latency; with W concurrent workers the
    wall-clock is ≈ sum(seconds)/W, so we divide the projection by workers.
    """
    n = prog["n"]
    per_cost = (prog["cost"] / n) if n else 0.0
    per_lat = (prog["seconds"] / n) if n else 0.0
    remaining = max(0, corpus_rows - n)
    workers = max(1, workers)
    rem_cost = remaining * per_cost
    rem_wall = remaining * per_lat / workers
    done_wall = prog["seconds"] / workers
    return {
        "done": n, "remaining": remaining, "corpus_rows": corpus_rows,
        "per_row_cost": per_cost, "per_row_latency": per_lat, "workers": workers,
        "done_cost": prog["cost"], "remaining_cost": rem_cost,
        "total_cost": prog["cost"] + rem_cost,
        "done_wall_s": done_wall, "remaining_wall_s": rem_wall,
        "total_wall_s": done_wall + rem_wall,
        "cost_per_1k": per_cost * 1000.0,
    }


def _dist_table(title: str, dist: dict, total: int) -> list[str]:
    """Render a count/percent markdown table for a distribution dict."""
    lines = [f"| {title} | count | share |", "|---|---:|---:|"]
    for k, v in dist.items():
        share = f"{v/total:.1%}" if total else "—"
        lines.append(f"| {_md_escape(k)} | {v:,} | {share} |")
    if not dist:
        lines.append("| _(none)_ | 0 | — |")
    return lines


# ============================================================================
# (3) Full markdown report
# ============================================================================
def write_full_report(per_row, lab, prog, proj, args, model, strategy_label):
    """Assemble the comprehensive report from the labelled-set evaluation (per_row
    + summary ``lab``), the Phase B progress ``prog`` and projections ``proj``."""
    n = lab["n"]
    out = args.report_out
    L: list[str] = []
    L.append(f"# Ontario flood labelling — full report")
    L.append("")
    L.append(f"_Generated {dt.datetime.now().isoformat(timespec='seconds')}._")
    L.append("")
    L.append(f"- **Model under analysis:** `{model}`  (strategy: {strategy_label})")
    L.append(f"- **Hand-labelled rows scored:** {n}  "
             f"(gold Ontario floods = {lab['tp'] + lab['fn']}, "
             f"not-Ontario-flood = {lab['fp'] + lab['tn']})")
    L.append(f"- **Corpus target:** {args.corpus_rows:,} rows  "
             f"(`--corpus-rows`); already labelled = {prog['n']:,}; "
             f"remaining = {proj['remaining']:,}")
    L.append(f"- **Recall target:** `is_ontario_flood = flood AND ontario` "
             f"(recall valued over precision).")
    L.append("")

    # ---- 1. headline metrics ------------------------------------------------
    L.append("## 1. Accuracy metrics (on the hand-labelled set)")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append(f"| Recall (sensitivity) | **{lab['recall']:.3f}** |")
    L.append(f"| Precision | {lab['precision']:.3f} |")
    L.append(f"| Accuracy | {lab['accuracy']:.3f} |")
    L.append(f"| F1 | {lab['f1']:.3f} |")
    L.append(f"| F2 (recall-weighted) | {lab['f2']:.3f} |")
    L.append(f"| Errors (unparsed rows) | {lab['errors']} |")
    L.append(f"| Cost to score these {n} rows | ${lab['cost']:.4f} "
             f"(${lab['cost_per_1k_rows']:.3f}/1k) |")
    L.append("")
    if "best" in strategy_label:
        L.append("> ⚠️ The `best` strategy reuses the compiled few-shot prompt; a few of "
                 "these rows may have served as demos, so these numbers are mildly "
                 "optimistic. Use `--report-strategy zeroshot` for a clean read.")
        L.append("")

    # ---- 2. confusion matrices ---------------------------------------------
    L.append("## 2. Confusion matrix")
    L.append("")
    L.append("Binary target `is_ontario_flood` (positive = a real Ontario flood):")
    L.append("")
    L.append("| | pred + | pred − |")
    L.append("|---|---:|---:|")
    L.append(f"| **actual +** | TP = {lab['tp']} | FN = {lab['fn']} |")
    L.append(f"| **actual −** | FP = {lab['fp']} | TN = {lab['tn']} |")
    L.append("")
    L.append(f"- **False negatives (missed Ontario floods): {lab['fn']}** — the costly "
             f"errors for this recall-first task.")
    L.append(f"- False positives (over-included): {lab['fp']} — tolerated; they get "
             f"filtered downstream.")
    L.append("")
    # 3-way decision breakdown (gold decision vs predicted decision)
    decisions3 = ["ontario_flood", "flood_not_ontario", "not_flood", "error"]
    gold3 = [r["gold_decision"] for r in per_row]
    grid = {g: Counter() for g in ("ontario_flood", "flood_not_ontario", "not_flood")}
    for r in per_row:
        grid[r["gold_decision"]][(r.get("decision") or "?")] += 1
    L.append("3-way decision breakdown (rows = gold, columns = predicted):")
    L.append("")
    L.append("| gold \\ pred | " + " | ".join(decisions3) + " | total |")
    L.append("|---|" + "|".join(["---:"] * (len(decisions3) + 1)) + "|")
    for g in ("ontario_flood", "flood_not_ontario", "not_flood"):
        row = grid[g]
        cells = " | ".join(str(row.get(d, 0)) for d in decisions3)
        L.append(f"| {g} | {cells} | {sum(row.values())} |")
    L.append("")

    # ---- 3. label distribution ---------------------------------------------
    L.append("## 3. Label distribution (hand-labelled set)")
    L.append("")
    L.append("Gold vs. predicted decision over the labelled rows:")
    L.append("")
    L += _dist_table("gold decision", lab["gold_dist"], n)
    L.append("")
    L += _dist_table("predicted decision", lab["pred_dist"], n)
    L.append("")

    # ---- 4. false negatives -------------------------------------------------
    fns = [r for r in per_row if r["category"] == "fn"]
    fps = [r for r in per_row if r["category"] == "fp"]
    L.append(f"## 4. False negatives — missed Ontario floods ({len(fns)})")
    L.append("")
    L.append("These are real Ontario floods the model marked **not** an Ontario flood "
             "— the errors that matter most. Review to tighten the prompt.")
    L.append("")
    if fns:
        L.append("| id | model reason | predicted | extract |")
        L.append("|---|---|---|---|")
        for r in fns[: args.fn_examples]:
            L.append(f"| {_md_escape(r['id'])} | {_md_escape(_snippet(r.get('reason'), 160))} "
                     f"| {_md_escape(r.get('decision'))} "
                     f"| {_md_escape(_snippet(r.get('text'), 240))} |")
        if len(fns) > args.fn_examples:
            L.append(f"\n_…and {len(fns) - args.fn_examples} more (see "
                     f"`{Path(args._labeled_eval_saved).name}`)._" if getattr(args, "_labeled_eval_saved", None)
                     else f"\n_…and {len(fns) - args.fn_examples} more._")
    else:
        L.append("_None — the model caught every Ontario flood in this set._")
    L.append("")

    # ---- 5. false positives -------------------------------------------------
    L.append(f"## 5. False positives — over-included ({len(fps)})")
    L.append("")
    L.append("Rows flagged as Ontario floods that are not. Tolerated under the "
             "recall-first objective, but high counts inflate downstream work.")
    L.append("")
    if fps:
        L.append("| id | gold | model reason | extract |")
        L.append("|---|---|---|---|")
        for r in fps[: args.fp_examples]:
            L.append(f"| {_md_escape(r['id'])} | {_md_escape(r['gold_decision'])} "
                     f"| {_md_escape(_snippet(r.get('reason'), 160))} "
                     f"| {_md_escape(_snippet(r.get('text'), 240))} |")
        if len(fps) > args.fp_examples:
            L.append(f"\n_…and {len(fps) - args.fp_examples} more._")
    else:
        L.append("_None._")
    L.append("")

    # ---- 6. corpus progress -------------------------------------------------
    L.append("## 6. Corpus progress so far")
    L.append("")
    if prog["n"] == 0:
        L.append(f"No predictions found yet at `{args.out}`. Run Phase B / "
                 "`--continue-run` first to populate progress + projections.")
        L.append("")
    else:
        pct = prog["n"] / args.corpus_rows if args.corpus_rows else 0
        L.append(f"- Rows labelled: **{prog['n']:,} / {args.corpus_rows:,}** ({pct:.1%})")
        L.append(f"- Ontario floods: {prog['ontario']:,}  |  floods elsewhere: "
                 f"{prog['flood_not_on']:,}  |  not a flood: {prog['not_flood']:,}  "
                 f"|  errors: {prog['errors']:,}")
        L.append(f"- Spend so far on these rows: **${prog['cost']:.2f}**  "
                 f"({prog['ptok'] + prog['ctok']:,} tokens)")
        if prog["models"]:
            L.append(f"- Model(s) used: " + ", ".join(
                f"`{k}` ({v:,})" for k, v in prog["models"].items()))
        if prog["by_day"]:
            L.append(f"- Rows per day: " + ", ".join(
                f"{k}: {v:,}" for k, v in prog["by_day"].items()))
        L.append("")
        L += _dist_table("predicted decision (corpus)", prog["decisions"], prog["n"])
        L.append("")
        if prog["flood_types"]:
            L += _dist_table("flood type (real floods)", prog["flood_types"],
                             sum(prog["flood_types"].values()))
            L.append("")
        if prog["locations"]:
            L += _dist_table("top non-Ontario locations", prog["locations"],
                             prog["flood_not_on"])
            L.append("")

    # ---- 7. cost & time projections ----------------------------------------
    L.append("## 7. Cost & time projections for the remaining corpus")
    L.append("")
    if prog["n"] == 0:
        lab_cost = lab["cost_per_1k_rows"] / 1000.0
        L.append("No corpus rows labelled yet — projecting from the hand-labelled "
                 f"run instead (~${lab['cost_per_1k_rows']:.3f}/1k rows):")
        L.append("")
        L.append(f"- Full {args.corpus_rows:,} rows ≈ **${lab_cost*args.corpus_rows:,.2f}** "
                 f"at ~{lab['avg_latency_s']:.2f}s/row "
                 f"({_hms(lab['avg_latency_s']*args.corpus_rows/max(1, args.workers))} "
                 f"wall at {args.workers} workers).")
        L.append("")
    else:
        L.append(f"Measured from the **{proj['done']:,} rows already labelled** "
                 f"(per-row ${proj['per_row_cost']:.5f}, {proj['per_row_latency']:.2f}s; "
                 f"wall assumes {proj['workers']} workers):")
        L.append("")
        L.append("| | rows | cost | wall-time |")
        L.append("|---|---:|---:|---:|")
        L.append(f"| Already done | {proj['done']:,} | ${proj['done_cost']:.2f} | "
                 f"{_hms(proj['done_wall_s'])} |")
        L.append(f"| Remaining | {proj['remaining']:,} | ${proj['remaining_cost']:.2f} | "
                 f"{_hms(proj['remaining_wall_s'])} |")
        L.append(f"| **Full corpus** | **{proj['corpus_rows']:,}** | "
                 f"**${proj['total_cost']:.2f}** | **{_hms(proj['total_wall_s'])}** |")
        L.append("")
        L.append(f"- Effective rate: **${proj['cost_per_1k']:.3f}/1k rows**.")
        L.append(f"- To finish in one go: `python tdm_overnight.py --continue-run "
                 f"--ignore-budget` (≈ ${proj['remaining_cost']:.2f} more).")
        lab_per_row = (lab["cost"] / lab["n"]) if lab["n"] else 0.0
        if lab_per_row > 0 and proj["per_row_cost"] > 0:
            L.append(f"- Cross-check: hand-labelled per-row cost ${lab_per_row:.5f} vs "
                     f"corpus ${proj['per_row_cost']:.5f} "
                     f"({proj['per_row_cost']/lab_per_row:.2f}× — extracts run longer/"
                     f"shorter than the labelled snippets).")
        L.append("")

    # ---- 8. optional: Phase A leaderboard -----------------------------------
    lb_path = ARTIFACTS_DIR / "prompt_eval_report.json"
    if lb_path.exists():
        try:
            board = json.load(open(lb_path)).get("candidates", [])
        except Exception:
            board = []
        if board:
            L.append("## 8. Prompt-selection leaderboard (Phase A)")
            L.append("")
            L.append("| model | strategy | recall | precision | F2 | $/1k | full corpus $ | fits? |")
            L.append("|---|---|---:|---:|---:|---:|---:|:--:|")
            for m in board:
                L.append(f"| {m.get('model')} | {m.get('strategy')} | "
                         f"{m.get('recall', 0):.3f} | {m.get('precision', 0):.3f} | "
                         f"{m.get('f2', 0):.3f} | ${m.get('cost_per_1k_rows', 0):.3f} | "
                         f"${m.get('full_corpus_cost', 0):.2f} | "
                         f"{'✅' if m.get('affordable') else '❌'} |")
            L.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")

    # machine-readable companion
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps({
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "model": model, "strategy": strategy_label,
        "labeled": {k: v for k, v in lab.items()},
        "corpus_progress": prog, "projections": proj,
        "false_negatives": [{k: r.get(k) for k in
                             ("id", "gold_decision", "decision", "reason", "flood_location")}
                            for r in fns],
        "false_positives": [{k: r.get(k) for k in
                             ("id", "gold_decision", "decision", "reason", "flood_location")}
                            for r in fps],
    }, indent=2), encoding="utf-8")
    return out, json_out


def run_report(args, api_key):
    """Mode (3): generate the full markdown report."""
    print("=" * 78)
    print("FULL REPORT")
    print("=" * 78)

    # 1) predictions on the 250 — reuse a saved eval, else run the chosen model.
    if args.reuse_eval:
        if not Path(args.reuse_eval).exists():
            raise SystemExit(f"--reuse-eval file not found: {args.reuse_eval}")
        per_row = load_labeled_eval_jsonl(args.reuse_eval)
        if not per_row:
            raise SystemExit(f"--reuse-eval file is empty: {args.reuse_eval}")
        model = per_row[0].get("model", "?")
        strategy_label = "reused predictions"
        args._labeled_eval_saved = str(args.reuse_eval)
        lab = summarize_labeled(per_row, model, args)
        print(f"[report] reused {len(per_row)} labelled predictions from {args.reuse_eval} "
              f"(model={model}).")
    else:
        rows = load_annotations(args.annotations)
        program, model, strategy_label = get_labeled_program(args, api_key)
        print(f"[report] scoring {len(rows)} hand-labelled rows with '{model}' "
              f"({strategy_label}); workers={args.eval_workers}…")
        lab, per_row = run_on_labeled(program, rows, model, args)
        save_path = labeled_eval_path(model, args.report_strategy)
        write_labeled_eval_jsonl(save_path, per_row)
        args._labeled_eval_saved = str(save_path)
        print(f"[report] per-row predictions -> {save_path}")

    print(f"[report] recall={lab['recall']:.3f} precision={lab['precision']:.3f} "
          f"acc={lab['accuracy']:.3f} | TP={lab['tp']} FP={lab['fp']} "
          f"FN={lab['fn']} TN={lab['tn']} | ${lab['cost']:.4f}")

    # 2) corpus progress + projections (folds in the rows already labelled)
    prog = phaseB_progress(args.out)
    proj = compute_projections(prog, args.corpus_rows, args.workers)
    if prog["n"]:
        print(f"[report] corpus: {prog['n']:,} done (${prog['cost']:.2f}); "
              f"remaining {proj['remaining']:,} ≈ ${proj['remaining_cost']:.2f}, "
              f"{_hms(proj['remaining_wall_s'])} wall.")

    md, js = write_full_report(per_row, lab, prog, proj, args, model, strategy_label)
    print(f"\n[report] wrote {md}\n[report] wrote {js}")


# ============================================================================
# (2) Test more models / configs on the hand-labelled set
# ============================================================================
def _expand_configs(args):
    """Cartesian sweep of (model, strategy, temperature, reasoning_effort).

    Effort only varies reasoning models; temperature only varies non-reasoning
    models — anything else is collapsed so the board has no meaningless dupes.
    """
    models = [canonical_model(m) for m in args.models]
    efforts = args.reasoning_efforts or [args.reasoning_effort]
    temps = args.temperatures or [args.temperature]
    seen, configs = set(), []
    for model in models:
        reasoning = is_reasoning_model(model)
        for strat in args.strategies:
            for eff in (efforts if reasoning else [args.reasoning_effort]):
                for temp in ([1.0] if reasoning else temps):
                    key = (model, strat, eff if reasoning else None,
                           None if reasoning else temp)
                    if key in seen:
                        continue
                    seen.add(key)
                    configs.append({"model": model, "strategy": strat,
                                    "temperature": temp, "reasoning_effort": eff,
                                    "reasoning": reasoning})
    return configs


def run_test_models(args, api_key):
    """Mode (2): benchmark several models/configs on the 250 hand-labelled rows."""
    print("=" * 78)
    print("TEST MODELS / CONFIGS  (on the hand-labelled set)")
    print("=" * 78)
    rows = load_annotations(args.annotations)
    n_pos = sum(r["is_ontario_flood"] for r in rows)
    print(f"[test] {len(rows)} labelled rows ({n_pos} Ontario floods).")

    if args.eval_on_full:
        eval_rows, train_rows = rows, rows
        print("[test] evaluating on ALL labelled rows (--eval-on-full); fewshot demos "
              "are bootstrapped from the same rows (mildly optimistic for fewshot).")
    else:
        train_rows, eval_rows = stratified_split(rows, args.eval_frac, args.seed)
        print(f"[test] train={len(train_rows)} eval={len(eval_rows)} "
              f"(held-out, stratified, seed={args.seed}).")
    train_ex = to_examples(train_rows)

    configs = _expand_configs(args)
    print(f"[test] {len(configs)} config(s): "
          f"models={[canonical_model(m) for m in args.models]} strategies={args.strategies} "
          f"efforts={args.reasoning_efforts or [args.reasoning_effort]} "
          f"temps={args.temperatures or [args.temperature]}\n")

    results = []
    probed_ok: dict[str, bool] = {}
    for cfg in configs:
        model, strat = cfg["model"], cfg["strategy"]
        cargs = _clone_args(args, temperature=cfg["temperature"],
                            reasoning_effort=cfg["reasoning_effort"])
        eff = cfg["reasoning_effort"] if cfg["reasoning"] else f"t={cfg['temperature']}"
        tag = f"{model} :: {strat} :: {eff}"
        # probe each distinct model once so an unreachable one is skipped cleanly
        if model not in probed_ok:
            try:
                lm = build_lm(model, cargs, api_key)
                probe = make_program("predict")
                probe.set_lm(lm)
                _ = probe(article_text="The Grand River overflowed its banks at Galt, "
                                       "Ontario, flooding homes.")
                probed_ok[model] = True
            except Exception as e:
                probed_ok[model] = False
                print(f"[test] SKIP model '{model}': {type(e).__name__}: {str(e)[:140]}")
        if not probed_ok[model]:
            continue
        try:
            lm = build_lm(model, cargs, api_key)
            program = make_program("cot" if strat == "cot" else "predict")
            program.set_lm(lm)
            if strat == "fewshot":
                with dspy.context(lm=lm):
                    program = compile_fewshot(program, train_ex, cargs)
                program.set_lm(lm)
            lab, per_row = run_on_labeled(program, eval_rows, model, cargs)
            save_path = labeled_eval_path(model, f"{strat}-{eff}")
            write_labeled_eval_jsonl(save_path, per_row)
            lab.update({"tag": tag, "strategy": strat, "config": eff,
                        "affordable": lab["full_corpus_cost"] <= args.budget * args.budget_safety,
                        "pred_path": str(save_path)})
            results.append(lab)
            print(f"  [{tag:<34}] R={lab['recall']:.3f} P={lab['precision']:.3f} "
                  f"F2={lab['f2']:.3f} acc={lab['accuracy']:.3f} FN={lab['fn']} FP={lab['fp']} "
                  f"${lab['cost_per_1k_rows']:.3f}/1k corpus=${lab['full_corpus_cost']:.2f}")
        except Exception as e:
            print(f"  [{tag}] FAILED: {type(e).__name__}: {str(e)[:160]}")

    if not results:
        raise SystemExit("[test] no config produced metrics — check proxy/models.")
    results.sort(key=lambda m: (m["f2"], m["recall"], -m["cost_per_1k_rows"]), reverse=True)
    write_model_test_report(results, eval_rows, args)


def write_model_test_report(results, eval_rows, args):
    n = len(eval_rows)
    n_pos = sum(r["is_ontario_flood"] for r in eval_rows)
    L = ["# Model / config benchmark — hand-labelled set", "",
         f"_Generated {dt.datetime.now().isoformat(timespec='seconds')}._", "",
         f"- Evaluated on **{n}** labelled rows ({n_pos} Ontario floods, {n-n_pos} not).",
         f"- Scope: {'ALL rows (--eval-on-full)' if args.eval_on_full else f'held-out split (eval-frac={args.eval_frac})'}.",
         "- Ranked by **F2** (recall-weighted) → recall → cheaper.",
         f"- Corpus sizing: full = {args.corpus_rows:,} rows; "
         f"'fits?' = full-corpus cost ≤ ${args.budget*args.budget_safety:.0f}.", "",
         "| rank | model | strategy | config | recall | precision | acc | F1 | F2 | FN | FP | $/1k | full corpus $ | fits? |",
         "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|"]
    for i, m in enumerate(results, 1):
        L.append(f"| {i} | {m['model']} | {m['strategy']} | {m['config']} | "
                 f"{m['recall']:.3f} | {m['precision']:.3f} | {m['accuracy']:.3f} | "
                 f"{m['f1']:.3f} | {m['f2']:.3f} | {m['fn']} | {m['fp']} | "
                 f"${m['cost_per_1k_rows']:.3f} | ${m['full_corpus_cost']:.2f} | "
                 f"{'✅' if m['affordable'] else '❌'} |")
    best = results[0]
    L += ["", "## Best config", "",
          f"**{best['tag']}** — recall {best['recall']:.3f}, precision {best['precision']:.3f}, "
          f"F2 {best['f2']:.3f}, {best['fn']} missed floods, "
          f"~${best['cost_per_1k_rows']:.3f}/1k rows (≈ ${best['full_corpus_cost']:.2f} "
          f"for the full {args.corpus_rows:,}-row corpus).", "",
          "Per-row predictions for each config are saved alongside this report "
          "(`artifacts/labeled_eval__*.jsonl`); feed one to "
          "`--report --reuse-eval <path>` for its full FP/FN breakdown.", ""]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    md = ARTIFACTS_DIR / "model_test_report.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    (ARTIFACTS_DIR / "model_test_report.json").write_text(
        json.dumps({"eval_size": n, "eval_on_full": args.eval_on_full,
                    "corpus_rows": args.corpus_rows,
                    "results": [{k: v for k, v in m.items()} for m in results]},
                   indent=2), encoding="utf-8")
    print(f"\n[test] wrote {md} and .json  (best: {best['tag']})")


# ============================================================================
# Self-test: exercise the whole pipeline offline with a DummyLM (no network)
# ============================================================================
def _synthetic_data(tmp: Path):
    """Write tiny annotations + extracts CSVs and return (paths, markers->truth)."""
    on = [
        "the grand river overflowed its banks at galt flooding main street",
        "ice jam on the thames river backed up water into chatham ontario homes",
        "spring freshet swept through the don valley toronto washing out roads",
        "ottawa river burst flooding low lying parts of the city of ottawa",
        "flash flood on the speed river inundated guelph ontario overnight",
        "lake erie storm surge drove water into homes along the ontario shore",
    ]
    elsewhere = [
        "the red river flooded winnipeg manitoba forcing thousands to flee",
        "mississippi river flood devastates towns across louisiana this week",
        "severe flooding reported in calgary alberta after heavy mountain rain",
    ]
    notflood = [
        "the office was a flood of letters congratulating the new mayor today",
        "new flood lights installed at the arena for the evening hockey games",
        "a flood insurance policy advertisement ran on the classifieds page",
        "council debated hypothetical flood scenarios for a future zoning plan",
    ]
    rows, truth = [], {}
    idx = 0
    for group, flood, ont in ((on, 1, 1), (elsewhere, 1, 0), (notflood, 0, 0)):
        for txt in group:
            for rep in range(3):  # inflate to ~39 rows so splits have positives
                marker = f"ROW{idx:04d}"
                idx += 1
                rows.append({"id": marker, "date": "1950", "example": f"{marker} {txt}",
                             "changed": "True", "flood": flood, "ontario": ont})
                truth[marker] = {
                    "is_ontario_flood": bool(flood and ont),
                    "decision": ("ontario_flood" if flood and ont else
                                 "flood_not_ontario" if flood else "not_flood"),
                    "reason": f"synthetic reason for {marker}",
                    "flood_location": ("Ontario" if flood and ont else
                                       "elsewhere" if flood else "n/a"),
                    "not_flood_reason": ("n/a" if flood else "metaphor"),
                    "flood_type": ("river" if flood else "n/a"),
                }
    ann = tmp / "annotations_so_far.csv"
    with open(ann, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "date", "example", "changed", "flood", "ontario"])
        w.writeheader()
        w.writerows(rows)

    ext = tmp / "extracted_only_1.csv"
    with open(ext, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "date", "extracted_text"])
        w.writeheader()
        for i, r in enumerate(rows[:12]):
            eid = f"EXT{i:04d}"
            w.writerow({"id": eid, "date": "1950", "extracted_text": r["example"]})
            truth[eid] = truth[r["id"]]  # reuse marker (text starts with ROWxxxx)
    return ann, ext, truth


def run_self_test(args):
    import tempfile

    from dspy.utils.dummies import DummyLM, dotdict

    class _SelfTestLM(DummyLM):
        """DummyLM that returns a default answer for any unmatched prompt (e.g.
        the Phase-A model probe) instead of unparseable 'No more responses'."""

        def __init__(self, answers: dict, default: dict):
            # format answers as JSON so the JsonObjectAdapter can parse them
            super().__init__(answers, adapter=dspy.JSONAdapter())
            self._answers_map = answers
            self._default = default

        def forward(self, prompt=None, messages=None, **kwargs):
            messages = messages or [{"role": "user", "content": prompt}]
            content = messages[-1]["content"]
            chosen = next((v for k, v in self._answers_map.items() if k in content), self._default)
            text = self._format_answer_fields(chosen)
            return dotdict(
                choices=[dotdict(message=dotdict(content=text, tool_calls=None), finish_reason="stop")],
                usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model="dummy",
            )

        def dump_state(self):  # so program.save() works in the offline test
            return {"model": "dummy"}

    tmp = Path(tempfile.mkdtemp(prefix="tdm_selftest_"))
    ann, ext, truth = _synthetic_data(tmp)

    def answers_for(noise_seed: int):
        rng = random.Random(noise_seed)
        out = {}
        for marker, lab in truth.items():
            a = dict(lab)
            # inject a little per-"model" error so the leaderboard isn't all ties
            if noise_seed and rng.random() < 0.15:
                a["is_ontario_flood"] = not a["is_ontario_flood"]
            out[marker] = a
        return out

    seeds = {"fakeA": 0, "fakeB": 7}
    default = {"is_ontario_flood": False, "decision": "not_flood", "reason": "probe",
               "flood_location": "n/a", "not_flood_reason": "other", "flood_type": "n/a"}

    global _LM_FACTORY
    _LM_FACTORY = lambda model: _SelfTestLM(answers_for(seeds.get(model, 0)), default)  # noqa: E731

    dspy.configure(lm=_LM_FACTORY("fakeA"))
    configure_dspy()  # JsonObjectAdapter + track_usage (what the real run uses)

    args.models = ["fakeA", "fakeB"]
    args.strategies = ["zeroshot", "fewshot"]
    args.annotations = ann
    args.extracts = ext
    args.out = tmp / "preds.jsonl"
    args.out_csv = tmp / "preds.csv"
    args.ledger = tmp / "ledger.jsonl"
    args.budget = 100.0
    args.corpus_rows = 12
    args.max_rows = 12
    args.workers = 4
    args.eval_workers = 4
    args.price_in = 0.001
    args.price_out = 0.002
    args.optimize_budget = 50.0
    # new-mode knobs (defaults exist via argparse; set explicitly for the test)
    args.report_out = tmp / "full_report.md"
    args.reuse_eval = None
    args.report_model = None
    args.report_strategy = "best"
    args.fp_examples = 10
    args.fn_examples = 10
    args.eval_on_full = True
    args.reasoning_efforts = None
    args.temperatures = None
    args.continue_run = False
    args.ignore_budget = False

    print("=== SELF-TEST (DummyLM, no network) ===")
    program, model, spent = phase_optimize(args, api_key="dummy", remaining_budget=100.0)
    phase_run(program, model, spent, args, api_key="dummy")

    # assertions
    assert args.out_csv.exists(), "no output CSV written"
    with open(args.out_csv) as f:
        rdr = list(csv.DictReader(f))
    assert rdr, "output CSV empty"
    for col in OUTPUT_FIELDS:
        assert col in rdr[0], f"missing column {col}"
    assert (ARTIFACTS_DIR / "prompt_eval_report.md").exists(), "no report"
    assert (ARTIFACTS_DIR / "best_program.json").exists(), "no best program saved"
    assert args.ledger.exists(), "no usage ledger written"
    # daily-spend re-read works
    _a, _b, tot = today_spend(args)
    assert tot > 0, "today_spend did not pick up ledger + predictions"

    # ---- (3) --report : metrics + confusion + FP/FN + projections --------
    print("\n=== SELF-TEST: --report ===")
    run_report(args, api_key="dummy")
    assert args.report_out.exists(), "report markdown not written"
    assert args.report_out.with_suffix(".json").exists(), "report json not written"
    rpt = args.report_out.read_text()
    for needle in ("Confusion matrix", "False negatives", "projections", "Recall"):
        assert needle in rpt, f"report missing section: {needle!r}"
    rj = json.load(open(args.report_out.with_suffix(".json")))
    assert "projections" in rj and "labeled" in rj, "report json missing keys"

    # report can reuse the saved per-row predictions without any LM calls
    print("\n=== SELF-TEST: --report --reuse-eval ===")
    reuse_args = _clone_args(args, reuse_eval=Path(args._labeled_eval_saved),
                             report_out=tmp / "full_report_reuse.md")
    run_report(reuse_args, api_key="dummy")
    assert reuse_args.report_out.exists(), "reuse-eval report not written"

    # ---- (2) --test-models : multi-config benchmark on the 250 -----------
    print("\n=== SELF-TEST: --test-models ===")
    run_test_models(args, api_key="dummy")
    assert (ARTIFACTS_DIR / "model_test_report.md").exists(), "model test report not written"
    mt = json.load(open(ARTIFACTS_DIR / "model_test_report.json"))
    assert mt["results"], "model test produced no results"

    # ---- (1) --continue-run --ignore-budget : resume sizing path ---------
    print("\n=== SELF-TEST: --continue-run --ignore-budget (resume, corpus done) ===")
    cont_args = _clone_args(args, ignore_budget=True)
    phase_run(program, model, spent, cont_args, api_key="dummy")  # all done -> 0 new rows

    print(f"\nSELF-TEST PASSED ✅  ({len(rdr)} rows labelled; columns OK; "
          f"report+test-models+best+ledger saved)")
    print(f"  today_spend re-read: A=${_a:.4f} B=${_b:.4f} total=${tot:.4f}")
    print(f"  full report: {args.report_out}")
    print(f"  scratch dir: {tmp}")
    _LM_FACTORY = None


# ============================================================================
# CLI
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # data / io
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    p.add_argument("--extracts", type=Path, default=DEFAULT_EXTRACTS)
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "ontario_flood_predictions.jsonl")
    p.add_argument("--out-csv", type=Path, default=PROCESSED_DIR / "ontario_flood_predictions.csv")
    p.add_argument("--text-col", default="extracted_text")
    p.add_argument("--id-col", default="id")
    p.add_argument("--date-col", default="date")
    # proxy / auth
    p.add_argument("--base-url", default=TDM_BASE_URL)
    p.add_argument("--token-file", default=TDM_TOKEN_FILE)
    p.add_argument("--api-key", default=None, help="overrides --token-file")
    # models / prompts. Defaults are all cheap enough to label the whole corpus
    # in a day, and include the newer gpt-5 family (probed; skipped if the
    # account can't reach them). gpt-4.1 was dropped from defaults: at ~$4/1k
    # rows it cannot do 90k within $50.
    p.add_argument("--models", type=lambda s: [x for x in s.split(",") if x],
                   default=["gpt-4o-mini", "gpt-5-nano", "gpt-5-mini", "gpt-4.1-nano"],
                   help="comma-separated candidate models (accepts gpt_4o_mini or gpt-4o-mini)")
    p.add_argument("--strategies", type=lambda s: [x for x in s.split(",") if x],
                   default=["zeroshot", "fewshot"],
                   help="prompt strategies: zeroshot,fewshot,cot")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="output cap (reasoning models are auto-raised to >=16000)")
    p.add_argument("--reasoning-effort", default="minimal",
                   choices=["minimal", "low", "medium", "high", "none"],
                   help="effort for gpt-5/o-series; 'minimal' keeps them fast+cheap")
    p.add_argument("--num-retries", type=int, default=4)
    p.add_argument("--no-cache", action="store_true")
    # pricing (for budget estimation only; proxy's measured cost wins when present)
    p.add_argument("--price-in", type=float, default=None,
                   help="USD per 1K input tokens (override pricing tables for ALL models)")
    p.add_argument("--price-out", type=float, default=None,
                   help="USD per 1K output tokens (override pricing tables for ALL models)")
    # optimization
    p.add_argument("--eval-frac", type=float, default=0.30, help="held-out eval fraction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-demos", type=int, default=4, help="bootstrapped few-shot demos")
    p.add_argument("--max-rounds", type=int, default=1)
    p.add_argument("--optimize-budget", type=float, default=None,
                   help="USD cap on Phase A (default min($5, 25%% of remaining))")
    # budget / scale
    p.add_argument("--budget", type=float, default=50.0, help="daily USD cap (A+B), tracked across runs")
    p.add_argument("--budget-safety", type=float, default=0.90, help="fraction of remaining budget to commit")
    p.add_argument("--corpus-rows", type=int, default=91000,
                   help="total unlabelled rows; the chosen model must fit ALL of them in one day")
    p.add_argument("--max-rows", type=int, default=None, help="optional hard cap on rows labelled this run")
    p.add_argument("--workers", type=int, default=8, help="concurrent inference requests (Phase B)")
    p.add_argument("--eval-workers", type=int, default=8, help="concurrent eval requests (Phase A)")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER,
                   help="persistent daily usage log, re-read each run for the $/day cap")
    # flow control
    p.add_argument("--skip-optimize", action="store_true", help="skip Phase A; reuse saved best")
    p.add_argument("--reuse-best", action="store_true", help="load artifacts/best_program.json")
    p.add_argument("--force-optimize", action="store_true",
                   help="re-run Phase A even if today's prompt is already cached")
    p.add_argument("--optimize-only", action="store_true", help="run Phase A only")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--self-test", action="store_true", help="offline mechanical test (DummyLM)")
    p.add_argument("--verbose", action="store_true")

    # (1) continue labelling the leftover corpus -----------------------------
    p.add_argument("--continue-run", action="store_true",
                   help="(1) skip Phase A, load the saved best prompt, and keep labelling the "
                        "leftover unlabelled rows (resumes from the JSONL checkpoint)")
    p.add_argument("--ignore-budget", action="store_true",
                   help="bypass the daily $ cap and size Phase B to the remaining corpus "
                        "(use with --continue-run to finish all 91k in one go)")

    # (2) benchmark more models/configs on the labelled set ------------------
    p.add_argument("--test-models", action="store_true",
                   help="(2) evaluate --models x --strategies x configs on the hand-labelled "
                        "rows and write artifacts/model_test_report.{md,json}; no corpus run")
    p.add_argument("--eval-on-full", action="store_true",
                   help="for --test-models: score ALL labelled rows instead of a held-out split")
    p.add_argument("--reasoning-efforts", type=lambda s: [x for x in s.split(",") if x],
                   default=None, help="for --test-models: sweep these efforts on reasoning models "
                                      "(e.g. minimal,low,medium)")
    p.add_argument("--temperatures", type=lambda s: [float(x) for x in s.split(",") if x],
                   default=None, help="for --test-models: sweep these temperatures on non-reasoning models")

    # (3) full markdown report ----------------------------------------------
    p.add_argument("--report", action="store_true",
                   help="(3) write a full markdown report: metrics, confusion matrices, "
                        "distributions, FP/FN analysis on the labelled set, and cost/time "
                        "projections for the remaining corpus (folding in rows already done)")
    p.add_argument("--report-model", default=None,
                   help="model to score the labelled set for the report (default: saved best, "
                        "else first of --models)")
    p.add_argument("--report-strategy", default="best", choices=["best", "zeroshot", "cot"],
                   help="'best' reuses the compiled prompt actually labelling the corpus; "
                        "'zeroshot'/'cot' build a fresh program for a clean read")
    p.add_argument("--reuse-eval", type=Path, default=None,
                   help="for --report: load existing per-row labelled predictions (a "
                        "labeled_eval__*.jsonl from a prior --report/--test-models) instead "
                        "of re-scoring (no LM spend)")
    p.add_argument("--report-out", type=Path, default=ARTIFACTS_DIR / "full_report.md",
                   help="output path for the --report markdown")
    p.add_argument("--fp-examples", type=int, default=15,
                   help="max false-positive examples to list in the report")
    p.add_argument("--fn-examples", type=int, default=25,
                   help="max false-negative examples to list in the report")
    return p


def load_best_from_disk(args):
    meta_path = ARTIFACTS_DIR / "best_program.meta.json"
    prog_path = ARTIFACTS_DIR / "best_program.json"
    if not prog_path.exists():
        raise SystemExit(f"--reuse-best/--skip-optimize but {prog_path} not found. Run Phase A first.")
    model = canonical_model(args.models[0])
    strategy, meta = "fewshot", {}
    if meta_path.exists():
        meta = json.load(open(meta_path))
        model = canonical_model(meta.get("model", model))
        strategy = meta.get("strategy", strategy)
    program = make_program("cot" if strategy == "cot" else "predict")
    program.load(str(prog_path))
    args._best_cost_per_row = (meta.get("cost_per_1k_rows", 0.0) or 0.0) / 1000.0
    print(f"[reuse] loaded best program (model='{model}', strategy='{strategy}', "
          f"day={meta.get('day','?')}) from {prog_path}")
    return program, model


def _cached_prompt_is_today() -> bool:
    meta_path = ARTIFACTS_DIR / "best_program.meta.json"
    prog_path = ARTIFACTS_DIR / "best_program.json"
    if not (meta_path.exists() and prog_path.exists()):
        return False
    try:
        return json.load(open(meta_path)).get("day") == _today()
    except Exception:
        return False


def main():
    args = build_parser().parse_args()

    if args.self_test:
        run_self_test(args)
        return

    _load_tdm_pricing()
    configure_dspy()  # track usage + single-call json_object adapter

    # Read-only-ish modes that don't run the corpus: dispatch before the budget
    # gate so they work even when today's cap is already spent. The auth token is
    # read lazily so '--report --reuse-eval' regenerates a report fully offline.
    if args.test_models:
        run_test_models(args, read_api_key(args))
        return
    if args.report:
        run_report(args, None if args.reuse_eval else read_api_key(args))
        return

    api_key = read_api_key(args)

    # --continue-run = reuse the saved prompt and keep labelling the leftover.
    if args.continue_run:
        args.skip_optimize = True

    # Re-read the log: how much of today's budget is already gone (prior runs)?
    spent_A, spent_B, already = today_spend(args)
    remaining = max(0.0, args.budget - already)

    mode = ("CONTINUE leftover corpus" if args.continue_run else
            "RELABEL (reuse saved)" if (args.skip_optimize or args.reuse_best) else
            "OPTIMIZE + LABEL")
    print("=" * 78)
    print("TDM OVERNIGHT FLOOD LABELLER")
    print("=" * 78)
    print(f"  mode          : {mode}{'  (--ignore-budget)' if args.ignore_budget else ''}")
    print(f"  budget        : ${args.budget:.2f}/day   (today already spent ${already:.4f}: "
          f"A=${spent_A:.4f} B=${spent_B:.4f})")
    print(f"  remaining     : ${remaining:.4f}")
    print(f"  candidates    : {[canonical_model(m) for m in args.models]}")
    print(f"  strategies    : {args.strategies}   reasoning_effort={args.reasoning_effort}")
    print(f"  corpus        : {args.corpus_rows:,} rows  (goal: all in one day)")
    print(f"  annotations   : {args.annotations}")
    print(f"  extracts      : {args.extracts}")
    print(f"  proxy         : {args.base_url}")
    print(f"  ledger        : {args.ledger}")

    if remaining <= 0.01 and not args.ignore_budget:
        print("\n[stop] today's budget is exhausted. Re-run tomorrow (the ledger resets by day), "
              "or pass --ignore-budget (with --continue-run) to push through the rest now.")
        return

    spent_thisrun = 0.0
    if args.skip_optimize or args.reuse_best:
        program, model = load_best_from_disk(args)
        program.set_lm(build_lm(model, args, api_key))
    elif (not args.force_optimize) and _cached_prompt_is_today():
        print("\n[Phase A] reusing today's already-optimized prompt "
              "(use --force-optimize to re-run). Its cost is already in the ledger.")
        program, model = load_best_from_disk(args)
        program.set_lm(build_lm(model, args, api_key))
    else:
        program, model, spent_thisrun = phase_optimize(args, api_key, remaining)

    if args.optimize_only:
        print("\n[done] --optimize-only set; skipping Phase B.")
        return

    # spent_so_far for Phase B = everything already spent today + this run's Phase A
    phase_run(program, model, already + spent_thisrun, args, api_key)


if __name__ == "__main__":
    main()
