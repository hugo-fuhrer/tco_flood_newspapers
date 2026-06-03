import dspy


class OCRCorrection(dspy.Signature):
    """Clean OCR-extracted text from historical Canadian newspapers.
    
    CORRECT:
    - Spelling errors caused by OCR misreading (e.g., "tlie" → "the", "aud" → "and")
    - Spacing errors (e.g., "fl ood" → "flood", "yes terday" → "yesterday")
    - Punctuation errors introduced by OCR
    - Character substitutions (e.g., "l" for "1", "0" for "O")
    
    PRESERVE:
    - Historical and period-appropriate spelling (e.g., "colour", "to-day", "waggon")
    - Original sentence structure and content
    - Original formatting as much as possible
    
    DO NOT:
    - Modernize language or spelling
    - Add information not present in the original
    - Rewrite or paraphrase content
    """
    raw_text: str = dspy.InputField(desc="Raw OCR-extracted text from a historical Canadian newspaper")
    title: str = dspy.InputField(desc="Article headline/title if available, else empty string")
    corrected_text: str = dspy.OutputField(desc="Cleaned text with OCR errors corrected but historical language preserved")
    corrections_made: str = dspy.OutputField(desc="Brief summary of the types of corrections made, or 'none' if text was clean")