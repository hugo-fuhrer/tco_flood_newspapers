from pathlib import Path

import dspy

from ocr import OCRCorrection
from signatures import FloodExtraction, floodIdentification, isOntario

lm = dspy.LM("ollama/llama3.1:8b", api_base="http://127.0.0.1:11434", api_key="ollama")
dspy.configure(lm=lm)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _load_optimized(predictor, name):
    """Load a compiled program from artifacts/ if optimize.py has produced one."""
    path = ARTIFACTS_DIR / f"{name}_filter.json"
    if path.exists():
        predictor.load(str(path))
        print(f"[pipeline] loaded optimized {name} filter from {path}")
    return predictor

# Initialize all predictors
ocr_cleaner = dspy.Predict(OCRCorrection)
flood_identifier = _load_optimized(dspy.Predict(floodIdentification), "flood")
ontario_filter = _load_optimized(dspy.Predict(isOntario), "ontario")
extractor = dspy.Predict(FloodExtraction)

def process_article(raw_text: str, title: str = "", clean_ocr: bool = False) -> dict:
    # The filters were optimized on RAW OCR text, so they run on raw_text first.
    # Filtering before any cleaning keeps the filters' train/inference
    # distribution consistent and avoids extra work on rejected articles.
    #
    # OCR correction (Stage 2) is optional and OFF by default: it's the most
    # expensive call (a full-article generation) and extraction is usually fine
    # on the raw text. Pass clean_ocr=True to run it for accepted articles.

    # Stage 1a: Is it a real flood? (on raw text)
    flood_check = flood_identifier(article_text=raw_text, title=title)
    if not flood_check.flood_mentioned:
        return {"status": "rejected", "reason": "no real flood"}

    # Stage 1b: Is it Ontario? (on raw text)
    ontario_check = ontario_filter(article_text=raw_text, title=title)
    if not ontario_check.is_ontario:
        return {"status": "rejected", "reason": "not Ontario"}

    # Stage 2 (optional): Clean OCR for accepted articles.
    if clean_ocr:
        article_text = ocr_cleaner(raw_text=raw_text, title=title).corrected_text
    else:
        article_text = raw_text

    # Stage 3: Extract structured data.
    extraction = extractor(article=article_text)

    return {
        "status": "accepted",
        "date": extraction.date,
        "location": extraction.location,
        "intensity": extraction.intensity,
        "corrected_text": article_text if clean_ocr else None,
    }