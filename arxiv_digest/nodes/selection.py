import json

from groq import Groq

from arxiv_digest.llm import json_completion
from arxiv_digest.schemas import RANK_SCHEMA
from arxiv_digest.state import DigestState


TOP_N = 5


def selection_ranking_node(state: DigestState, client: Groq) -> None:
    if state["query_kind"] != "topic" or not state["candidates"]:
        return

    compact_candidates = [
        {"arxiv_id": item["arxiv_id"], "title": item["title"], "abstract": item["summary"][:700]}
        for item in state["candidates"]
    ]
    ranked = json_completion(
        client,
        "Rank these arXiv papers by relevance to the user's topic. Return every supplied arxiv_id once, "
        "most relevant first; do not invent IDs.\n"
        f"Topic: {state['query']}\nCandidates: {json.dumps(compact_candidates)}",
        RANK_SCHEMA,
    )
    valid_ids = {item["arxiv_id"] for item in state["candidates"]}
    ordered_ids = [str(value) for value in ranked.get("ranked_ids", []) if value in valid_ids]
    ordered_ids.extend(item["arxiv_id"] for item in state["candidates"] if item["arxiv_id"] not in ordered_ids)
    by_id = {item["arxiv_id"]: item for item in state["candidates"]}
    state["candidates"] = [by_id[arxiv_id] for arxiv_id in ordered_ids]


def select_paper_node(state: DigestState, input_fn=input, print_fn=print) -> None:
    if not state["candidates"]:
        return
    if state["query_kind"] == "arxiv_id" or len(state["candidates"]) == 1:
        selected = state["candidates"][0]
    else:
        visible = state["candidates"][:TOP_N]
        print_fn("\nRanked arXiv matches:")
        for index, paper in enumerate(visible, 1):
            print_fn(f"{index}. {paper['title']} ({paper['arxiv_id']})")
        if len(state["candidates"]) > TOP_N:
            print_fn(f"Showing top {TOP_N} of {len(state['candidates'])} candidates.")
        while True:
            choice = input_fn(f"Choose a paper [1-{len(visible)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(visible):
                selected = visible[int(choice) - 1]
                break
            print_fn("Enter one of the displayed candidate numbers.")
        state["selection_note"] = f"Selected ranked candidate {selected['arxiv_id']} from {len(state['candidates'])} results."

    state["arxiv_id"] = selected["arxiv_id"]
    state["paper_metadata"] = selected
    state["abstract"] = selected["summary"]