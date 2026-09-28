import re

from groq import Groq

from arxiv_digest.llm import json_completion
from arxiv_digest.schemas import CLASSIFY_SCHEMA
from arxiv_digest.state import DigestState


ARXIV_ID_PATTERN = re.compile(
    r"(?:(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/|arxiv:)?"
    r"((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)


def query_understanding_node(state: DigestState, client: Groq) -> None:
    match = ARXIV_ID_PATTERN.search(state["query"])
    if match:
        state["query_kind"] = "arxiv_id"
        state["arxiv_id"] = match.group(1)
        return

    state["query_kind"] = "topic"
    state["arxiv_id"] = None
    looks_like_reference = "arxiv" in state["query"].casefold() or re.search(
        r"\b\d{4}\.\d{1,4}\b", state["query"]
    )
    if not looks_like_reference:
        return

    decision = json_completion(
        client,
        "Classify this input as a topic or an arXiv ID. If it contains a recoverable arXiv ID, normalize it. "
        f"Input: {state['query']}",
        CLASSIFY_SCHEMA,
    )
    if decision.get("kind") == "arxiv_id" and decision.get("arxiv_id"):
        state["query_kind"] = "arxiv_id"
        state["arxiv_id"] = str(decision["arxiv_id"])