from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PaperMetadata(TypedDict):
    arxiv_id: str
    title: str
    authors: str
    published: str
    summary: str
    link: str
    pdf_url: str
    categories: str


class Chunk(TypedDict):
    id: str
    section: str
    text: str


class DigestState(TypedDict):
    query: str
    query_kind: Literal["topic", "arxiv_id"]
    arxiv_id: str | None
    candidates: list[PaperMetadata]
    selection_note: str | None
    paper_metadata: PaperMetadata | None
    abstract: str
    parsed_text: str
    sections: dict[str, str]
    partial: bool
    chunks: list[Chunk]
    vector_store_path: str
    collection_name: str
    briefing: dict[str, object] | None
    current_question: str
    last_answer: dict[str, object] | None
    messages: Annotated[list[AnyMessage], add_messages]


def new_state(query: str, vector_store_path: str) -> DigestState:
    return {
        "query": query,
        "query_kind": "topic",
        "arxiv_id": None,
        "candidates": [],
        "selection_note": None,
        "paper_metadata": None,
        "abstract": "",
        "parsed_text": "",
        "sections": {},
        "partial": False,
        "chunks": [],
        "vector_store_path": vector_store_path,
        "collection_name": "",
        "briefing": None,
        "current_question": "",
        "last_answer": None,
        "messages": [],
    }