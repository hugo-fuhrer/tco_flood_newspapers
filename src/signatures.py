"""
Stage 2 Signatures: DSPy signatures for flood verification and Ontario filtering

These signatures determine which articles pass to Stage 3 for extraction.
The goal is to capture all articles that could provide evidence of historical
Ontario floods, including brief mentions in obituaries, anniversary articles, etc.
"""
import dspy



class floodIdentification(dspy.Signature):
    """Determine if an article describes or references a real flood event.

    INCLUDE articles that:
    - Describe a flood event (past, present, or imminent)
    - Mention a historical flood in context (e.g., "survived the 1954 flood")
    - Reference flood damage, flood victims, or flood recovery
    - Discuss flood prevention/mitigation related to a specific event

    EXCLUDE articles that:
    - Use "flood" only metaphorically ("flood of applications", "flooded with calls")
    - Discuss flooding only hypothetically without reference to real events
    - Are about non-water floods (flood lights, flood insurance policies without events)
    - Only mention flooding in a completely different country with no Ontario connection
    """

    article_text: str = dspy.InputField(desc="Full text of the news article")
    title: str = dspy.InputField(desc="Article headline/title")

    flood_mentioned: bool = dspy.OutputField(
        desc="True if article references any real flood event. False for metaphorical, hypothetical, or non-water 'flood' usage."
    )
    reasoning: str = dspy.OutputField(
        desc="Explain what flood event was referenced, or why this is metaphorical/not a real flood."
    )



class isOntario(dspy.Signature):
    """Determine if the flood event described occurred in Ontario, Canada.

    The article must describe a flood that OCCURRED IN Ontario, not just:
    - A flood mentioned by an Ontario newspaper but located elsewhere
    - A flood affecting Ontarians who were traveling elsewhere
    - A flood in another province (Manitoba, Quebec, etc.) or country

    INCLUDE if:
    - Flood location is an Ontario city, town, or region
    - Flood is on an Ontario river, lake, or watershed
    - Article explicitly states the flood was in Ontario

    EXCLUDE if:
    - Flood occurred in another Canadian province (even if article is from Ontario)
    - Flood occurred in another country
    - No specific location is mentioned for the flood
    - The only Ontario connection is the newspaper's location, not the flood's location
    """

    article_text: str = dspy.InputField(desc="Full text of the flood article")
    title: str = dspy.InputField(desc="Article headline/title")

    is_ontario: bool = dspy.OutputField(
        desc="True ONLY if the flood occurred in Ontario, Canada. False if flood was elsewhere or location unclear."
    )
    reasoning: str = dspy.OutputField(
        desc="Identify the flood location and explain why it is/isn't in Ontario."
    )


class FloodExtraction(dspy.Signature):
    """Extract structured data about an Ontario flood event from a news article.

    The article has already been verified to describe a real flood that occurred
    in Ontario. Extract ONLY what the article itself states. Do NOT invent,
    guess, or infer details from outside knowledge.

    Output rules (important):
    - Each field is a SHORT value only — no sentences, no explanations, no
      parenthetical notes, no markdown, no headings.
    - If the article does not state a field, output exactly: unknown
    - Never copy values from these instructions; read them from the article.
    """

    article: str = dspy.InputField(desc="Full text of an Ontario flood article")

    date: str = dspy.OutputField(
        desc="When the flood occurred, taken only from the article's own words "
             "(a year, a year-month like YYYY-MM, or a season+year). Do not infer "
             "from context. Output 'unknown' if the article gives no date."
    )
    location: str = dspy.OutputField(
        desc="Ontario place(s) the flood affected (city/town, river, or watershed) "
             "as named in the article. Comma-separated if several. 'unknown' if not stated."
    )
    intensity: str = dspy.OutputField(
        desc="A few words on the flood's severity/impact (e.g. deaths, damage, "
             "evacuations, water levels) as stated in the article. 'unknown' if not stated."
    )