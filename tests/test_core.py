import unittest
import xml.etree.ElementTree as ET
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from groq import Groq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from arxiv_digest.graph import build_graph, describe_graph, read_qa_question
from arxiv_digest.logging_config import configure_logging
from arxiv_digest.nodes.briefing import BRIEFING_TEXT_BUDGET, briefing_excerpt
from arxiv_digest.nodes.parsing import split_sections
from arxiv_digest.nodes.query import query_understanding_node
from arxiv_digest.nodes.retrieval import (
    ArxivSearchUnavailable,
    arxiv_retrieval_node,
    atom_search,
    parse_atom_entry,
)
from arxiv_digest.state import new_state


class QueryUnderstandingTests(unittest.TestCase):
    def test_arxiv_url_is_recognized_without_llm(self) -> None:
        state = new_state("https://arxiv.org/abs/2401.12345v2", ".store")
        query_understanding_node(state, client=None)
        self.assertEqual(state["query_kind"], "arxiv_id")
        self.assertEqual(state["arxiv_id"], "2401.12345v2")


class ParsingTests(unittest.TestCase):
    def test_sections_include_references(self) -> None:
        sections = split_sections(
            "1 Introduction\nProblem setup.\n2 Method\nOur approach.\nReferences\n[1] A citation."
        )
        self.assertEqual(sections["Introduction"], "Problem setup.")
        self.assertEqual(sections["Method"], "Our approach.")
        self.assertEqual(sections["References"], "[1] A citation.")

    def test_atom_entry_keeps_category(self) -> None:
        entry = ET.fromstring(
            """<entry xmlns="http://www.w3.org/2005/Atom">
            <id>http://arxiv.org/abs/2401.12345v2</id>
            <title>A useful paper</title>
            <summary>An abstract.</summary>
            <published>2024-01-01T00:00:00Z</published>
            <author><name>Researcher One</name></author>
            <category term="cs.AI"/>
            <category term="cs.LG"/>
            <link title="pdf" href="https://arxiv.org/pdf/2401.12345v2"/>
            </entry>"""
        )
        metadata = parse_atom_entry(entry)
        self.assertEqual(metadata["arxiv_id"], "2401.12345v2")
        self.assertEqual(metadata["categories"], "cs.AI, cs.LG")
        self.assertEqual(metadata["pdf_url"], "https://arxiv.org/pdf/2401.12345v2")

    @patch("arxiv_digest.nodes.retrieval.time.sleep")
    @patch("arxiv_digest.nodes.retrieval.curl_requests.get")
    def test_atom_search_retries_406_once_after_three_seconds(self, get, sleep) -> None:
        throttled = MagicMock(status_code=406, content=b"denied")
        success = MagicMock(status_code=200, content=b'<feed xmlns="http://www.w3.org/2005/Atom"/>')
        get.side_effect = [throttled, success]

        self.assertEqual(atom_search("all:transformer"), [])

        sleep.assert_called_once_with(3)
        self.assertEqual(get.call_count, 2)
        url = get.call_args.args[0]
        self.assertIn("search_query=all:transformer", url)
        self.assertEqual(get.call_args.kwargs["headers"]["User-Agent"], "arxiv-paper-digest/1.0")
        self.assertEqual(get.call_args.kwargs["impersonate"], "chrome")

    @patch("arxiv_digest.nodes.retrieval.curl_requests.get")
    def test_atom_search_uses_id_list_for_paper_ids(self, get) -> None:
        get.return_value = MagicMock(status_code=200, content=b'<feed xmlns="http://www.w3.org/2005/Atom"/>')

        self.assertEqual(atom_search(id_list="2401.12345", max_results=1), [])

        url = get.call_args.args[0]
        self.assertIn("id_list=2401.12345", url)
        self.assertNotIn("search_query", url)

    @patch("arxiv_digest.nodes.retrieval.atom_search")
    def test_id_lookup_reports_persistent_406_without_raising(self, search) -> None:
        search.side_effect = ArxivSearchUnavailable("406")
        state = new_state("2401.12345", ".store")
        state["query_kind"] = "arxiv_id"
        state["arxiv_id"] = "2401.12345"

        arxiv_retrieval_node(state)

        self.assertEqual(state["candidates"], [])
        self.assertIn("temporarily rejecting", state["selection_note"])
        search.assert_called_once_with(id_list="2401.12345", max_results=1)

    @patch("arxiv_digest.nodes.retrieval.atom_search")
    def test_topic_search_broadens_after_persistent_406(self, search) -> None:
        paper = {
            "arxiv_id": "2603.20397v1",
            "title": "KV cache review",
            "authors": "Author",
            "published": "2026-03-20",
            "summary": "Review of cache methods.",
            "link": "https://arxiv.org/abs/2603.20397v1",
            "pdf_url": "https://arxiv.org/pdf/2603.20397v1.pdf",
            "categories": "cs.CL",
        }
        search.side_effect = [ArxivSearchUnavailable("406"), [paper]]
        state = new_state("KV CACHE OPTIMIZATION", ".store")

        arxiv_retrieval_node(state)

        self.assertEqual(search.call_args.args[0], "all:KV")
        self.assertEqual(state["candidates"], [paper])
        self.assertIn("broadening once", state["selection_note"])

    @patch("arxiv_digest.nodes.retrieval.atom_search")
    def test_topic_search_returns_cleanly_when_broadened_query_also_406(self, search) -> None:
        search.side_effect = [ArxivSearchUnavailable("406"), ArxivSearchUnavailable("406")]
        state = new_state("ATTENTION MECHANISM", ".store")

        arxiv_retrieval_node(state)

        self.assertEqual(state["candidates"], [])
        self.assertIn("temporarily unavailable", state["selection_note"])
        self.assertIn("one broadened search", state["selection_note"])
        self.assertEqual(search.call_count, 2)

    @patch("arxiv_digest.nodes.retrieval.time.sleep")
    @patch("arxiv_digest.nodes.retrieval.curl_requests.get")
    def test_atom_search_reports_persistent_406(self, get, sleep) -> None:
        throttled = MagicMock(status_code=406, content=b"proxy rejection")
        get.side_effect = [throttled, throttled]

        with self.assertRaisesRegex(ArxivSearchUnavailable, "rejected the search twice"):
            atom_search("all:transformer")

        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once_with(3)


class StateTests(unittest.TestCase):
    def test_state_starts_without_selected_metadata(self) -> None:
        state = new_state("transformers", ".store")
        self.assertEqual(state["query_kind"], "topic")
        self.assertIsNone(state["paper_metadata"])
        self.assertEqual(state["messages"], [])

    def test_annotated_message_reducer_appends_both_roles(self) -> None:
        history = add_messages([], [HumanMessage(content="What is attention?")])
        history = add_messages(history, [AIMessage(content="The paper describes it. [c0001]")])
        self.assertEqual([message.type for message in history], ["human", "ai"])
        self.assertEqual(history[0].content, "What is attention?")
        self.assertIn("[c0001]", history[1].content)

    def test_briefing_excerpt_prioritizes_sections_within_budget(self) -> None:
        state = new_state("topic", ".store")
        state["sections"] = {
            "Body": "body " * 10000,
            "Abstract": "abstract details",
            "Results": "reported results",
            "Conclusion": "paper conclusion",
        }
        excerpt = briefing_excerpt(state)
        self.assertLessEqual(len(excerpt), BRIEFING_TEXT_BUDGET + 100)
        self.assertIn("[Abstract]", excerpt)
        self.assertIn("[Results]", excerpt)
        self.assertIn("[Conclusion]", excerpt)
        self.assertNotIn("body body", excerpt)

    def test_run_logger_writes_info_warning_to_unique_timestamped_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "logs"
            logger = logging.getLogger("arxiv_digest.test_logging")
            try:
                first_log = configure_logging(log_dir)
                logger.info("first run info")
                logger.warning("first run warning")
                for handler in logging.getLogger().handlers:
                    handler.flush()
                first_run = first_log.read_text(encoding="utf-8")
                self.assertIn("INFO", first_run)
                self.assertIn("WARNING", first_run)
                self.assertIn("first run info", first_run)
                self.assertRegex(first_log.name, r"^log_\d{8}_\d{6}_\d{6}\.txt$")

                second_log = configure_logging(log_dir)
                logger.warning("second run only")
                for handler in logging.getLogger().handlers:
                    handler.flush()
                second_run = second_log.read_text(encoding="utf-8")
                self.assertNotEqual(first_log, second_log)
                self.assertTrue(first_log.exists())
                self.assertIn("first run info", first_log.read_text(encoding="utf-8"))
                self.assertNotIn("first run info", second_run)
                self.assertIn("second run only", second_run)
                self.assertEqual(len(list(log_dir.glob("log_*.txt"))), 2)
            finally:
                root_logger = logging.getLogger()
                for handler in root_logger.handlers[:]:
                    if getattr(handler, "_arxiv_digest_handler", False):
                        root_logger.removeHandler(handler)
                        handler.close()


class LangGraphTests(unittest.TestCase):
    def test_qa_prompt_treats_eof_as_exit(self) -> None:
        def end_of_input(_: str) -> str:
            raise EOFError

        self.assertEqual(read_qa_question(end_of_input), "/exit")

    def test_compiled_graph_exposes_nodes_and_conditional_edges(self) -> None:
        compiled = build_graph()
        mermaid, nodes, edges = describe_graph(compiled)
        self.assertIn("query_understanding", nodes)
        self.assertIn("qa_answer", nodes)
        self.assertIn("arxiv_retrieval -.-> no_results", mermaid)
        self.assertTrue(any("conditional=True" in edge for edge in edges))

    def test_no_results_routes_to_end(self) -> None:
        output: list[str] = []

        def no_results(state) -> None:
            state["candidates"] = []
            state["selection_note"] = "No matching paper."

        with patch("arxiv_digest.graph.arxiv_retrieval_node", side_effect=no_results):
            compiled = build_graph(client=Groq(api_key="test-key"), print_fn=output.append)
            final_state = compiled.invoke(new_state("transformers", ".store"))

        self.assertEqual(output, ["No matching paper."])
        self.assertIsNone(final_state["briefing"])


if __name__ == "__main__":
    unittest.main()