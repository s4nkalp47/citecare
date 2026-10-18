"""Pure display helpers for the Streamlit app, kept out of app.py so they can be unit-tested.

`Citation.score` is never displayed: since Phase 4b it is a reranker sigmoid that is often near 0
even for correct sources. The only relevance hint is rank-based ("Top match" on ref 1).
"""

from __future__ import annotations

from src.core.schemas import Citation, Confidence

# label, Streamlit badge colour
CONFIDENCE_BADGES: dict[str, tuple[str, str]] = {
    "high": ("High confidence", "green"),
    "medium": ("Medium confidence", "orange"),
    "low": ("Low confidence", "red"),
    "insufficient_evidence": ("Insufficient evidence", "gray"),
}

SOURCE_TYPE_LABELS = {"guideline": "Guideline", "pubmed": "PubMed", "bootstrap": "Bootstrap"}


def confidence_badge(confidence: Confidence | str) -> tuple[str, str]:
    return CONFIDENCE_BADGES.get(confidence, (str(confidence).replace("_", " ").capitalize(),
                                              "gray"))


def rank_label(citation: Citation) -> str | None:
    """Context blocks are numbered in retrieval order, so ref 1 is the best-ranked chunk."""
    return "Top match" if citation.ref == 1 else None


def source_type_label(source_type: str) -> str:
    return SOURCE_TYPE_LABELS.get(source_type, source_type.replace("_", " ").capitalize())


def citation_heading(citation: Citation) -> str:
    """Markdown heading for one citation: [n] title (linked), page, source type, rank label."""
    title = citation.title or "Untitled source"
    title_md = f"[{_escape(title)}]({citation.url})" if citation.url else _escape(title)
    parts = [f"**[{citation.ref}]** {title_md}"]
    if citation.page is not None:
        parts.append(f"p. {citation.page}")
    parts.append(source_type_label(citation.source_type))
    heading = " · ".join(parts)
    label = rank_label(citation)
    return f"{heading} · :green-badge[{label}]" if label else heading


def _escape(text: str) -> str:
    """Stop titles from breaking the link markdown."""
    return text.replace("[", r"\[").replace("]", r"\]")
