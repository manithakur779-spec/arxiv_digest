import logging
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from curl_cffi import requests as curl_requests

from arxiv_digest.nodes.query import ARXIV_ID_PATTERN
from arxiv_digest.state import DigestState, PaperMetadata


ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
logger = logging.getLogger(__name__)


class ArxivSearchUnavailable(RuntimeError):
    """The arXiv Atom search endpoint rejected a query after one retry."""


def parse_atom_entry(entry: ET.Element) -> PaperMetadata:
    entry_id = entry.findtext(f"{ATOM}id", default="")
    match = ARXIV_ID_PATTERN.search(entry_id)
    if match is None:
        raise ValueError(f"arXiv returned an unrecognized paper ID: {entry_id}")

    pdf_link = next(
        (link.attrib["href"] for link in entry.findall(f"{ATOM}link") if link.attrib.get("title") == "pdf"),
        f"https://arxiv.org/pdf/{match.group(1)}.pdf",
    )
    authors = [author.findtext(f"{ATOM}name", default="") for author in entry.findall(f"{ATOM}author")]
    categories = [category.attrib.get("term", "") for category in entry.findall(f"{ATOM}category")]
    return {
        "arxiv_id": match.group(1),
        "title": " ".join(entry.findtext(f"{ATOM}title", default="").split()),
        "authors": ", ".join(authors),
        "published": entry.findtext(f"{ATOM}published", default="")[:10],
        "summary": " ".join(entry.findtext(f"{ATOM}summary", default="").split()),
        "link": f"https://arxiv.org/abs/{match.group(1)}",
        "pdf_url": pdf_link,
        "categories": ", ".join(category for category in categories if category),
    }


def atom_search(
    search_query: str | None = None,
    id_list: str | None = None,
    max_results: int = 10,
) -> list[PaperMetadata]:
    if not search_query and not id_list:
        raise ValueError("Provide either search_query or id_list.")
    if search_query and id_list:
        raise ValueError("Provide search_query or id_list, not both.")

    params: dict[str, str | int] = {}
    if search_query:
        params["search_query"] = search_query
    if id_list:
        params["id_list"] = id_list
    if max_results != 10:
        params["max_results"] = max_results
    request_url = f"{ARXIV_API}?{urlencode(params, safe=':')}"
    headers = {
        "Accept": "application/atom+xml",
        "User-Agent": "arxiv-paper-digest/1.0",
    }
    logger.info(
        "Calling arXiv Atom API (%s=%s, max_results=%d)",
        "search_query" if search_query else "id_list",
        search_query or id_list,
        max_results,
    )
    for attempt in range(2):
        response = curl_requests.get(
            request_url,
            headers=headers,
            timeout=30,
            impersonate="chrome",
        )
        if response.status_code == 406:
            body = response.content[:500]
            logger.warning("arXiv HTTP 406 on attempt %d; response body: %r", attempt + 1, body)
            if attempt == 0:
                logger.warning("Retrying arXiv request once after 3 seconds")
                time.sleep(3)
            else:
                raise ArxivSearchUnavailable(
                    "arXiv rejected the search twice with HTTP 406. Wait briefly and run the topic search again."
                )
            continue
        response.raise_for_status()
        content = response.content
        break

    root = ET.fromstring(content)
    papers = [parse_atom_entry(entry) for entry in root.findall(f"{ATOM}entry")]
    logger.info("arXiv Atom API returned %d candidates", len(papers))
    return papers


def arxiv_retrieval_node(state: DigestState) -> None:
    if state["query_kind"] == "arxiv_id":
        try:
            state["candidates"] = atom_search(id_list=state["arxiv_id"], max_results=1)
        except ArxivSearchUnavailable:
            state["selection_note"] = "arXiv is temporarily rejecting requests (HTTP 406). Try again shortly."
            state["candidates"] = []
            logger.warning("arXiv ID lookup unavailable for %s", state["arxiv_id"])
        return

    phrase = state["query"].replace('"', " ").strip()
    try:
        candidates = atom_search(f'all:"{phrase}"', max_results=10)
    except ArxivSearchUnavailable:
        primary_term = next(iter(re.findall(r"[A-Za-z0-9-]+", phrase)), "")
        if not primary_term:
            state["selection_note"] = "arXiv search is temporarily unavailable (HTTP 406). Try again later."
            state["candidates"] = []
            return
        state["selection_note"] = (
            f"The exact topic search was rejected; broadening once to all:{primary_term} for ranking."
        )
        logger.warning("Exact topic query was rejected; broadening once to first term '%s'", primary_term)
        try:
            candidates = atom_search(f"all:{primary_term}", max_results=10)
        except ArxivSearchUnavailable:
            state["selection_note"] = (
                "arXiv search is temporarily unavailable (HTTP 406), including after one broadened search. "
                "Try again later."
            )
            logger.warning("arXiv rejected the broadened topic query as well")
            state["candidates"] = []
            return

    if not candidates:
        if state["selection_note"]:
            state["selection_note"] = "No papers matched the broadened topic search."
            state["candidates"] = []
            return
        terms = re.findall(r"[A-Za-z0-9-]+", phrase)[:8]
        broadened_query = " OR ".join(f"all:{term}" for term in terms)
        if broadened_query:
            logger.info("No exact topic matches; issuing one broadened search")
            time.sleep(3)
            candidates = atom_search(broadened_query, max_results=10)
            state["selection_note"] = "The exact topic had no matches; a broadened topic search was used."
        else:
            candidates = []
        if not candidates:
            state["selection_note"] = "No papers matched after one broadened topic search."
            logger.warning("No candidates found after the single topic broadening attempt")
    state["candidates"] = candidates