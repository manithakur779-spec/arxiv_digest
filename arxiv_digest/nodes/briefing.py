import json
import logging

from groq import Groq

from arxiv_digest.llm import json_completion
from arxiv_digest.schemas import BRIEFING_SCHEMA
from arxiv_digest.state import DigestState


BRIEFING_TEXT_BUDGET = 14000
PRIORITY_SECTIONS = (
    "Abstract",
    "Introduction",
    "Problem Formulation",
    "Method",
    "Methods",
    "Approach",
    "Experiments",
    "Evaluation",
    "Results",
    "Discussion",
    "Limitations",
    "Conclusion",
    "Conclusions",
)
logger = logging.getLogger(__name__)


def briefing_excerpt(state: DigestState) -> str:
    sections = state["sections"]
    selected: list[str] = []
    remaining = BRIEFING_TEXT_BUDGET
    for section_name in PRIORITY_SECTIONS:
        content = sections.get(section_name, "")
        if not content or remaining <= 0:
            continue
        excerpt = content[: min(2500, remaining)]
        selected.append(f"[{section_name}]\n{excerpt}")
        remaining -= len(excerpt)

    if selected:
        return "\n\n".join(selected)

    text = state["parsed_text"]
    if len(text) <= BRIEFING_TEXT_BUDGET:
        return text
    beginning_budget = BRIEFING_TEXT_BUDGET * 2 // 3
    ending_budget = BRIEFING_TEXT_BUDGET - beginning_budget
    return f"{text[:beginning_budget]}\n\n[Middle omitted]\n\n{text[-ending_budget:]}"


def summarize_node(state: DigestState, client: Groq) -> None:
    metadata = state["paper_metadata"]
    if metadata is None:
        raise ValueError("Cannot summarize a paper before selection.")
    prompt = (
        "Create an executive briefing using only the supplied paper excerpts and metadata. "
        "Do not infer missing details. For an abstract-only partial parse, say that unavailable methods/results "
        "are not stated rather than guessing. Limitations must never be empty: when none are explicitly stated, "
        "say so. The excerpts are section-prioritized and may omit content; do not imply they cover the full paper. "
        "Keep the output concise: summary under 100 words, problem statement under 60 words, up to four method "
        "bullets, five result bullets, three limitation bullets, and three follow-up questions. Summary must be "
        "one paragraph.\n"
        f"Partial parse: {state['partial']}\n"
        f"Metadata: {json.dumps(metadata)}\n"
        f"Paper excerpts: {briefing_excerpt(state)}"
    )
    briefing = json_completion(client, prompt, BRIEFING_SCHEMA)
    briefing.update(
        {
            "title": metadata["title"],
            "authors": [author.strip() for author in metadata["authors"].split(",") if author.strip()],
            "arxiv_id": metadata["arxiv_id"],
            "publish_date": metadata["published"],
            "link": metadata["link"],
            "partial": state["partial"],
        }
    )
    state["briefing"] = briefing
    logger.info("Structured briefing generated (partial=%s)", state["partial"])