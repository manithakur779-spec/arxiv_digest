import json
import logging
from typing import Callable

from groq import Groq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from arxiv_digest.nodes.briefing import summarize_node
from arxiv_digest.nodes.indexing import chunk_embed_node
from arxiv_digest.nodes.parsing import fetch_parse_node
from arxiv_digest.nodes.qa import qa_node
from arxiv_digest.nodes.query import query_understanding_node
from arxiv_digest.nodes.retrieval import arxiv_retrieval_node
from arxiv_digest.nodes.selection import select_paper_node, selection_ranking_node
from arxiv_digest.settings import VECTOR_STORE_PATH
from arxiv_digest.state import DigestState, new_state


logger = logging.getLogger(__name__)


def retrieval_route(state: DigestState) -> str:
    if not state["candidates"]:
        return "no_results"
    if state["query_kind"] == "topic":
        return "rank"
    return "select"


def qa_route(state: DigestState) -> str:
    if state["current_question"].casefold() == "/exit":
        return "end"
    if not state["current_question"].strip():
        return "ask_again"
    return "answer"


def read_qa_question(input_fn: Callable[[str], str]) -> str:
    try:
        return input_fn("\nQ> ").strip()
    except EOFError:
        return "/exit"


def build_graph(
    client: Groq | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
):
    builder = StateGraph(DigestState)

    def require_client() -> Groq:
        if client is None:
            raise RuntimeError("A Groq client is required to execute the digest graph.")
        return client

    def understand(state: DigestState) -> dict[str, object]:
        logger.info("Graph node: query understanding")
        query_understanding_node(state, require_client())
        return {key: value for key, value in state.items() if key != "messages"}

    def retrieve(state: DigestState) -> dict[str, object]:
        logger.info("Graph node: arXiv retrieval")
        arxiv_retrieval_node(state)
        return {key: value for key, value in state.items() if key != "messages"}

    def rank(state: DigestState) -> dict[str, object]:
        logger.info("Graph node: selection ranking (%d candidates)", len(state["candidates"]))
        selection_ranking_node(state, require_client())
        return {key: value for key, value in state.items() if key != "messages"}

    def select(state: DigestState) -> dict[str, object]:
        logger.info("Graph node: paper selection")
        select_paper_node(state, input_fn=input_fn, print_fn=print_fn)
        return {key: value for key, value in state.items() if key != "messages"}

    def parse(state: DigestState) -> dict[str, object]:
        logger.info("Graph node: fetch and parse PDF")
        fetch_parse_node(state)
        return {key: value for key, value in state.items() if key != "messages"}

    def index(state: DigestState) -> dict[str, object]:
        logger.info("Graph node: chunk and embed (%d sections)", len(state["sections"]))
        chunk_embed_node(state)
        return {key: value for key, value in state.items() if key != "messages"}

    def summarize(state: DigestState) -> dict[str, object]:
        logger.info("Graph node: structured briefing")
        summarize_node(state, require_client())
        return {key: value for key, value in state.items() if key != "messages"}

    def report_no_results(state: DigestState) -> dict[str, object]:
        logger.warning("No paper selected: %s", state["selection_note"] or "no candidates found")
        print_fn(state["selection_note"] or "No arXiv paper was found.")
        return {}

    def show_briefing(state: DigestState) -> dict[str, object]:
        if state["briefing"] is not None:
            print_briefing(state["briefing"], print_fn=print_fn)
            print_fn("\nAsk questions about this paper. Enter /exit to finish.")
        return {}

    def ask_question(_: DigestState) -> dict[str, object]:
        question = read_qa_question(input_fn)
        if question == "/exit":
            logger.info("QA session ended by user or end of input")
        elif question:
            logger.info("Received QA question")
        update: dict[str, object] = {"current_question": question}
        if question and question.casefold() != "/exit":
            update["messages"] = [HumanMessage(content=question)]
        return update

    def answer_question(state: DigestState) -> dict[str, object]:
        logger.info("Graph node: retrieve context and answer QA question")
        answer = qa_node(state, require_client(), state["current_question"])
        citation_text = ", ".join(str(value) for value in answer["citations"])
        message_text = str(answer["answer"])
        if citation_text:
            message_text += f" [{citation_text}]"
        return {"last_answer": answer, "messages": [AIMessage(content=message_text)]}

    def show_answer(state: DigestState) -> dict[str, object]:
        answer = state["last_answer"] or {"answer": "Not found in paper.", "citations": []}
        citation_text = ", ".join(str(value) for value in answer["citations"])
        suffix = f" [{citation_text}]" if citation_text else ""
        print_fn(f"A> {answer['answer']}{suffix}")
        return {}

    builder.add_node("query_understanding", understand)
    builder.add_node("arxiv_retrieval", retrieve)
    builder.add_node("selection_ranking", rank)
    builder.add_node("paper_selection", select)
    builder.add_node("fetch_parse", parse)
    builder.add_node("chunk_embed", index)
    builder.add_node("summarize", summarize)
    builder.add_node("no_results", report_no_results)
    builder.add_node("display_briefing", show_briefing)
    builder.add_node("qa_prompt", ask_question)
    builder.add_node("qa_answer", answer_question)
    builder.add_node("display_answer", show_answer)

    builder.add_edge(START, "query_understanding")
    builder.add_edge("query_understanding", "arxiv_retrieval")
    builder.add_conditional_edges(
        "arxiv_retrieval",
        retrieval_route,
        {"no_results": "no_results", "rank": "selection_ranking", "select": "paper_selection"},
    )
    builder.add_edge("no_results", END)
    builder.add_edge("selection_ranking", "paper_selection")
    builder.add_edge("paper_selection", "fetch_parse")
    builder.add_edge("fetch_parse", "chunk_embed")
    builder.add_edge("chunk_embed", "summarize")
    builder.add_edge("summarize", "display_briefing")
    builder.add_edge("display_briefing", "qa_prompt")
    builder.add_conditional_edges(
        "qa_prompt",
        qa_route,
        {"end": END, "ask_again": "qa_prompt", "answer": "qa_answer"},
    )
    builder.add_edge("qa_answer", "display_answer")
    builder.add_edge("display_answer", "qa_prompt")
    return builder.compile()


def describe_graph(graph=None) -> tuple[str, list[str], list[str]]:
    compiled = graph or build_graph()
    graph_view = compiled.get_graph()
    nodes = list(graph_view.nodes)
    edges = [
        f"{edge.source} -> {edge.target}; conditional={edge.conditional}"
        for edge in graph_view.edges
    ]
    return graph_view.draw_mermaid(), nodes, edges


def run_graph(
    query: str,
    client: Groq,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> DigestState:
    logger.info("Compiling and invoking digest graph")
    graph = build_graph(client, input_fn=input_fn, print_fn=print_fn)
    initial_state = new_state(query, str(VECTOR_STORE_PATH))
    return graph.invoke(initial_state, config={"recursion_limit": 10000})


def print_briefing(briefing: dict[str, object], print_fn: Callable[..., None] = print) -> None:
    print_fn("\nExecutive briefing")
    print_fn(json.dumps(briefing, indent=2, ensure_ascii=False))