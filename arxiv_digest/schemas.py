BRIEFING_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "arxiv_id": {"type": "string"},
        "publish_date": {"type": "string"},
        "link": {"type": "string"},
        "summary": {"type": "string"},
        "problem_statement": {"type": "string"},
        "method": {"type": "array", "items": {"type": "string"}},
        "key_results": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "suggested_followup_questions": {"type": "array", "items": {"type": "string"}},
        "partial": {"type": "boolean"},
    },
    "required": [
        "title",
        "authors",
        "arxiv_id",
        "publish_date",
        "link",
        "summary",
        "problem_statement",
        "method",
        "key_results",
        "limitations",
        "suggested_followup_questions",
        "partial",
    ],
}

RANK_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"ranked_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["ranked_ids"],
}

CLASSIFY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["topic", "arxiv_id"]},
        "arxiv_id": {"type": "string"},
    },
    "required": ["kind", "arxiv_id"],
}

ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "citations"],
}