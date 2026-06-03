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

def process_article(raw_text: str, title: str) -> dict:
    # Stage 1: Clean OCR
    cleaned = ocr_cleaner(raw_text=raw_text, title=title)
    
    # Stage 2a: Is it a real flood?
    flood_check = flood_identifier(article_text=cleaned.corrected_text, title=title)
    if not flood_check.flood_mentioned:
        return {"status": "rejected", "reason": "no real flood"}
    
    # Stage 2b: Is it Ontario?
    ontario_check = ontario_filter(article_text=cleaned.corrected_text, title=title)
    if not ontario_check.is_ontario:
        return {"status": "rejected", "reason": "not Ontario"}
    
    # Stage 3: Extract structured data
    extraction = extractor(article=cleaned.corrected_text)
    
    return {
        "status": "accepted",
        "date": extraction.date,
        "location": extraction.location,
        "intensity": extraction.intensity,
        "corrected_text": cleaned.corrected_text
    }