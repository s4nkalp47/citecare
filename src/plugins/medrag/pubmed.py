"""PubMedAbstractReader: a LlamaIndex reader for PubMed abstracts via NCBI E-utilities.

Replaces `llama_index.readers.papers.PubmedReader`, which searches PMC without relevance
sorting (returns the newest, unrelated papers), and dumps every XML element (references,
affiliations, error pages) into the text.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET

import httpx
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedAbstractReader(BaseReader):
    def __init__(self, api_key: str = "", client: httpx.Client | None = None,
                 request_delay: float = 0.4):
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=30)
        # NCBI allows 3 requests/s without a key, 10/s with one.
        self.request_delay = 0.11 if api_key else request_delay

    def load_data(self, query: str, max_results: int = 5, extra_filter: str = "") -> list[Document]:
        term = f"({query}) {extra_filter}".strip()
        pmids = self._search(term, max_results)
        if not pmids:
            logger.info("No PubMed results for %r", term)
            return []
        return [self._to_document(article) for article in self._fetch(pmids)]

    # ---------- E-utilities calls ----------

    def _get(self, endpoint: str, params: dict) -> httpx.Response:
        params = {**params, "tool": "medrag"}
        if self.api_key:
            params["api_key"] = self.api_key
        time.sleep(self.request_delay)
        resp = self.client.get(f"{EUTILS}/{endpoint}", params=params)
        resp.raise_for_status()
        return resp

    def _search(self, term: str, max_results: int) -> list[str]:
        resp = self._get("esearch.fcgi", {
            "db": "pubmed", "term": term, "retmax": max_results,
            "sort": "relevance", "retmode": "json",
        })
        return resp.json()["esearchresult"]["idlist"]

    def _fetch(self, pmids: list[str]) -> list[ET.Element]:
        resp = self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        return ET.fromstring(resp.content).findall(".//PubmedArticle")

    # ---------- XML -> Document ----------

    @staticmethod
    def _to_document(article: ET.Element) -> Document:
        pmid = article.findtext(".//PMID", default="")
        title = _text(article.find(".//ArticleTitle")) or "Untitled PubMed article"
        journal = article.findtext(".//Journal/Title", default="")
        year = (article.findtext(".//PubDate/Year")
                or (article.findtext(".//PubDate/MedlineDate", default="")[:4]))
        pub_types = [_text(pt) for pt in article.findall(".//PublicationType")]

        sections = []
        for part in article.findall(".//Abstract/AbstractText"):
            label = part.get("Label")
            body = _text(part)
            if body:
                sections.append(f"{label}: {body}" if label else body)
        abstract = "\n".join(sections)

        header = f"{title}\n{journal} ({year})" if journal else title
        return Document(
            text=f"{header}\n\n{abstract}".strip(),
            metadata={
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "year": year,
                "publication_types": ", ".join(pub_types),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            },
        )


def _text(element: ET.Element | None) -> str:
    """All text inside an element, including nested tags like <i> or <sup>."""
    return " ".join("".join(element.itertext()).split()) if element is not None else ""
