import argparse
import logging
from typing import get_type_hints

from arxiv_digest.graph import describe_graph, run_graph
from arxiv_digest.logging_config import configure_logging
from arxiv_digest.nodes.retrieval import ArxivSearchUnavailable
from arxiv_digest.settings import create_llm_client
from arxiv_digest.state import DigestState


logger = logging.getLogger(__name__)


def print_graph() -> None:
    mermaid, nodes, edges = describe_graph()
    print(mermaid)
    print("\nNodes:")
    for name in nodes:
        print(f"- {name}")
    print("\nEdges:")
    for edge in edges:
        print(f"- {edge}")
    print("\nShared state:")
    for name, value_type in get_type_hints(DigestState).items():
        print(f"- {name}: {value_type}")


def main() -> None:
    log_file = configure_logging()
    logger.info("Starting arXiv paper digest CLI; log file: %s", log_file)
    parser = argparse.ArgumentParser(description="Build an arXiv paper digest and answer grounded questions.")
    parser.add_argument("query", nargs="*", help="A topic, arXiv ID, or arXiv URL")
    parser.add_argument("--show-graph", action="store_true", help="Print the compiled graph, edges, and state schema")
    args = parser.parse_args()

    if args.show_graph:
        logger.info("Displaying compiled graph")
        print_graph()
        return
    if not args.query:
        parser.error("provide a topic/arXiv ID or use --show-graph")

    query = " ".join(args.query)
    logger.info("Starting digest run for query: %s", query)
    client = create_llm_client()
    try:
        run_graph(query, client)
    except ArxivSearchUnavailable as error:
        logger.warning("arXiv search unavailable: %s", error)
        parser.exit(1, f"Error: {error}\n")
    logger.info("Digest run finished")