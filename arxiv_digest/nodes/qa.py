import logging

from groq import Groq
from langchain_core.messages import AnyMessage

from arxiv_digest.llm import json_completion
from arxiv_digest.schemas import ANSWER_SCHEMA
from arxiv_digest.state import DigestState


TOP_K = 4
SIMILARITY_THRESHOLD = 0.35
logger = logging.getLogger(__name__)


def _conversation_context(messages: list[AnyMessage]) -> str:
    recent = messages[-7:]
    return "\n".join(
        f"{message.type}: {message.content}"
        for message in recent
        if message.content
    )


def qa_node(state: DigestState, client: Groq, question: str) -> dict[str, object]:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    if not state["chunks"]:
        logger.warning("QA has no indexed paper chunks; returning not-found")
        return {"answer": "Not found in paper.", "citations": []}

    conversation = _conversation_context(state["messages"][:-1])
    retrieval_query = f"{conversation}\nCurrent question: {question}" if conversation else question

    store = chromadb.PersistentClient(path=state["vector_store_path"])
    collection = store.get_collection(
        name=state["collection_name"],
        embedding_function=DefaultEmbeddingFunction(),
    )
    results = collection.query(query_texts=[retrieval_query], n_results=min(TOP_K, len(state["chunks"])))
    chunk_by_id = {chunk["id"]: chunk for chunk in state["chunks"]}
    retrieved = [
        (chunk_by_id[chunk_id], 1.0 - float(distance))
        for chunk_id, distance in zip(results["ids"][0], results["distances"][0])
        if 1.0 - float(distance) >= SIMILARITY_THRESHOLD
    ]
    if not retrieved:
        logger.info("No QA retrieval result cleared the similarity threshold %.2f", SIMILARITY_THRESHOLD)
        return {"answer": "Not found in paper.", "citations": []}

    context = "\n\n".join(
        f"[{chunk['id']} | {chunk['section']} | similarity {similarity:.3f}]\n{chunk['text']}"
        for chunk, similarity in retrieved
    )
    generated = json_completion(
        client,
        "Use prior conversation only to resolve references in the current question; prior answers are not evidence. "
        "Answer only from retrieved paper chunks. If the chunks do not support an answer, return exactly "
        "'Not found in paper.' with no citations. Cite chunk IDs in the citations array; never use outside knowledge.\n"
        f"Recent conversation for reference resolution:\n{conversation or '(none)'}\n"
        f"Question: {question}\nRetrieved chunks:\n{context}",
        ANSWER_SCHEMA,
    )
    allowed = {chunk["id"] for chunk, _ in retrieved}
    citations = [str(value) for value in generated.get("citations", []) if value in allowed]
    answer_text = str(generated.get("answer", "")).strip()
    if not answer_text or (answer_text.casefold() != "not found in paper." and not citations):
        logger.warning("QA answer lacked valid chunk citations; returning not-found")
        answer_text = "Not found in paper."
        citations = []
    else:
        logger.info("QA answer produced with %d grounded citations", len(citations))
    return {"answer": answer_text, "citations": citations}