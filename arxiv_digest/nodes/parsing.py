import logging
import re

import pymupdf
from curl_cffi import requests as curl_requests

from arxiv_digest.state import DigestState


SECTION_NAMES = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "approach",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "acknowledgments",
    "references",
)
HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(" + "|".join(re.escape(name) for name in SECTION_NAMES) + r")\s*\.?\s*$",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"Body": []}
    current = "Body"
    for line in text.splitlines():
        heading = HEADING_PATTERN.match(line)
        if heading:
            current = heading.group(1).title()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)
    return {name: content for name, lines in sections.items() if (content := "\n".join(lines).strip())}


def fetch_parse_node(state: DigestState) -> None:
    metadata = state["paper_metadata"]
    if metadata is None:
        raise ValueError("Cannot parse a paper before selection.")
    response = curl_requests.get(metadata["pdf_url"], timeout=60, impersonate="chrome")
    response.raise_for_status()
    try:
        with pymupdf.open(stream=response.content, filetype="pdf") as document:
            pages = [page.get_text("text") for page in document]
    except (pymupdf.FileDataError, RuntimeError, ValueError):
        pages = []

    extracted = "\n\n".join(pages).strip()
    text_density = sum(character.isalnum() for character in extracted) / max(len(pages), 1)
    word_count = len(re.findall(r"\b\w+\b", extracted))
    if text_density < 180 or word_count < 100:
        state["partial"] = True
        state["parsed_text"] = state["abstract"]
        state["sections"] = {"Abstract": state["abstract"]} if state["abstract"] else {}
        logger.warning(
            "PDF extraction is sparse (density=%d chars/page, words=%d); using abstract-only content",
            text_density,
            word_count,
        )
        return

    state["partial"] = False
    state["parsed_text"] = extracted
    state["sections"] = split_sections(extracted)
    if state["abstract"]:
        state["sections"].setdefault("Abstract", state["abstract"])
        logger.info("PDF parsed: %d pages, %d words, %d recognized sections", len(pages), word_count, len(state["sections"]))
