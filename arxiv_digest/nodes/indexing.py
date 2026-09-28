import logging
import re
import uuid

from arxiv_digest.state import Chunk, DigestState


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
logger = logging.getLogger(__name__)


def chunk_embed_node(state: DigestState) -> None:
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    chunks: list[Chunk] = []
    for section, content in state["sections"].items():
        start = 0
        while start < len(content):
            end = min(start + CHUNK_SIZE, len(content))
            if end < len(content):
                boundary = content.rfind(" ", start + CHUNK_SIZE - 400, end)
                if boundary > start:
                    end = boundary
            text = content[start:end].strip()
            if text:
                chunks.append({"id": f"c{len(chunks) + 1:04d}", "section": section, "text": text})
            if end >= len(content):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)

    state["chunks"] = chunks
    logger.info("Created %d chunks from paper sections", len(chunks))
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", state["arxiv_id"] or "paper")
    state["collection_name"] = f"paper_{safe_id}_{uuid.uuid4().hex[:8]}"
    store = chromadb.PersistentClient(path=state["vector_store_path"])
    collection = store.create_collection(
        name=state["collection_name"],
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    if chunks:
        collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[{"section": chunk["section"]} for chunk in chunks],
        )
    logger.info("Stored paper embeddings in Chroma collection %s", state["collection_name"])