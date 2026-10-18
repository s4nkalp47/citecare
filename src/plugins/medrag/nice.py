"""Recommendation-aware chunking for NICE guidelines (Phase 4b).

Page-sized chunks split recommendations at page breaks ("1.4.32 Offer an ACE inhibitor or an
ARB to adults starting step 1 antihypertensive" on p19, "treatment who: have type 2 diabetes..."
on p20), so no chunk held the whole recommendation. NICE guidelines are built from numbered
recommendations, so they are the chunk unit here:

- the ingestor joins a guideline's pages into one Document with page markers (`join_pages`);
- `NiceGuidelineParser` stitches pages back together (mid-sentence breaks become a space), strips
  the <u>/** markup LlamaParse adds, flattens HTML tables to "cell | cell" rows, and cuts at
  headings and recommendation numbers;
- each chunk starts with its heading path ("NICE NG136 Hypertension > 1.4 Treating and
  monitoring hypertension > Step 1 treatment"), so short recommendations keep their context;
- chunks over MAX_CHUNK_TOKENS are split with SentenceSplitter (prefix repeated on each piece),
  fragments under MIN_CHUNK_TOKENS (orphan "[2019]" tags) are merged into the previous chunk;
- each chunk gets `page` (where it starts), `page_end`, `rec_ids` and `section` (the heading
  path, used by neighbour expansion to stay within a section).

Anything else (non-NICE PDFs, PubMed abstracts, bootstrap docs) goes to a plain SentenceSplitter.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.utils import get_tokenizer

DOC_FORMAT = "nice_guideline"  # Document.metadata["doc_format"] for joined NICE guidelines
MAX_CHUNK_TOKENS = 400
MIN_CHUNK_TOKENS = 20
# Node metadata the parser adds; used for retrieval checks, not for embedding or the LLM.
PARSER_KEYS = ["doc_format", "page_end", "rec_ids", "section"]

_PAGE_MARKER = "<!-- page {page} -->"
_PAGE_MARKER_RE = re.compile(r"<!-- page (\d+) -->")
_MARKUP = re.compile(r"</?u>|\*\*")
_CELL_BREAK = re.compile(r"</t[dh]>\s*<t[dh][^>]*>")
_ROW_END = re.compile(r"</tr>")
_LINE_BREAK = re.compile(r"<br\s*/?>")
_TAG = re.compile(r"<(?!!--)[^>]+>")  # any HTML tag except our page-marker comments
# A recommendation starts a line, sometimes as a heading: "1.4.32 Offer...", "## 1.1.4".
_REC = re.compile(r"^(?:#{1,6}[ \t]*)?(\d+\.\d+\.\d+)\b[ \t]*")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_SECTION = re.compile(r"^\d+\.\d+\b")  # "1.4 Treating and monitoring hypertension"
_SUBHEADING_WORDS = ("table", "box", "notes for table")  # not new sections
_SKIP_HEADINGS = {"your responsibility"}  # legal boilerplate on every guideline
_ENDS_BLOCK = re.compile(r"[.!?:;)\]>|]\s*$")
_STARTS_BLOCK = re.compile(r"#|\d+\.\d+(\.\d+)?\b|[|<-]")
_TOC_LINE = re.compile(r"(\.{4,}|\s)\d{1,3}\s*$")


def is_nice_guideline(pages: list[dict]) -> bool:
    """Parsed pages of a NICE guideline: 'NICE guideline' up front and numbered recommendations."""
    if not pages:
        return False
    front = " ".join(p["markdown"] for p in pages[:2])
    recs = sum(1 for p in pages for line in p["markdown"].splitlines() if _REC.match(line))
    return "NICE guideline" in _MARKUP.sub("", front) and recs >= 10


def join_pages(pages: list[dict]) -> str:
    """One text for the whole guideline, with a marker before each page."""
    return "\n\n".join(f"{_PAGE_MARKER.format(page=p['page'])}\n\n{p['markdown']}" for p in pages)


@dataclass
class _Unit:
    start: int  # offsets into the stitched text
    end: int
    path: tuple[str, ...]
    rec_ids: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _flatten_html(text: str) -> str:
    """LlamaParse tables are HTML: '<tr><td>a</td><td>b</td></tr>' -> 'a | b' per line."""
    text = _LINE_BREAK.sub("; ", _CELL_BREAK.sub(" | ", text))
    text = _TAG.sub(" ", _ROW_END.sub("\n", text))  # spaces collapsed below
    lines = (" ".join(line.split()) for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def _stitch(text: str) -> tuple[str, list[int], list[int]]:
    """Remove page markers and markup; return (text, page start offsets, page numbers)."""
    parts = _PAGE_MARKER_RE.split(text)  # [before, page, text, page, text, ...]
    out, offsets, pages = "", [], []
    for page, chunk in zip(parts[1::2], parts[2::2]):
        chunk = _flatten_html(_MARKUP.sub("", chunk)).strip()
        if not chunk:
            continue
        if out:
            # A page that ends mid-sentence continues on the next one, unless that one starts a
            # new block (heading, recommendation, table row).
            continues = not _ENDS_BLOCK.search(out) and not _STARTS_BLOCK.match(chunk)
            out += " " if continues else "\n\n"
        offsets.append(len(out))
        pages.append(int(page))
        out += chunk
    return out, offsets, pages


def _clean_heading(text: str) -> str:
    return " ".join(text.replace("#", " ").split())


def _is_toc(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 3 and sum(bool(_TOC_LINE.search(ln)) for ln in lines) / len(lines) > 0.5


class NiceGuidelineParser(NodeParser):
    """NICE guidelines -> one chunk per recommendation; everything else -> SentenceSplitter."""

    chunk_size: int = 256  # for the SentenceSplitter fallback
    chunk_overlap: int = 32
    max_chunk_tokens: int = MAX_CHUNK_TOKENS
    min_chunk_tokens: int = MIN_CHUNK_TOKENS

    _fallback: SentenceSplitter = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._fallback = SentenceSplitter(chunk_size=self.chunk_size,
                                          chunk_overlap=self.chunk_overlap)
        self._tokenizer = get_tokenizer()

    @classmethod
    def class_name(cls) -> str:
        return "NiceGuidelineParser"

    def _parse_nodes(self, nodes: Sequence[BaseNode], show_progress: bool = False,
                     **kwargs: Any) -> list[BaseNode]:
        out: list[BaseNode] = []
        for node in nodes:
            if node.metadata.get("doc_format") == DOC_FORMAT:
                out += self._parse_guideline(node)
            else:
                out += self._fallback.get_nodes_from_documents([node])
        return out

    # ---------- NICE guidelines ----------

    def _tokens(self, text: str) -> int:
        return len(self._tokenizer(text))

    def _units(self, text: str, title: str) -> list[_Unit]:
        units: list[_Unit] = []
        section = subsection = table = ""  # table: a "Table n"/"Box n" heading, until the next rec
        skipping = False
        current: _Unit | None = None
        offset = 0
        for line in text.split("\n"):
            start, offset = offset, offset + len(line) + 1
            rec = _REC.match(line)
            heading = None if rec else _HEADING.match(line)
            if heading:
                name = _clean_heading(heading.group(2))
                if not name:
                    continue
                current = None
                if name.lower() in _SKIP_HEADINGS:
                    skipping = True
                elif name.lower().startswith(_SUBHEADING_WORDS):
                    table = name
                elif _SECTION.match(name) or len(heading.group(1)) == 1:
                    section, subsection, table, skipping = name, "", "", False
                else:
                    subsection, table = name, ""
                continue
            if skipping:
                continue
            if rec:
                table = ""
            path = tuple(p for p in (title, section, subsection, table) if p)
            if rec:
                line = line[rec.start(1):]  # drop a leading "## " on heading-style rec numbers
                current = _Unit(start=start, end=offset, path=path, rec_ids=[rec.group(1)])
                units.append(current)
            elif current is None:
                if not line.strip():
                    continue
                current = _Unit(start=start, end=offset, path=path)
                units.append(current)
            current.lines.append(line)
            current.end = offset
        return [u for u in units if u.text and not _is_toc(u.text)]

    def _merge_short(self, units: list[_Unit]) -> list[_Unit]:
        merged: list[_Unit] = []
        for unit in units:
            if merged and self._tokens(unit.text) < self.min_chunk_tokens:
                prev = merged[-1]
                prev.lines += unit.lines
                prev.end = unit.end
                prev.rec_ids += [r for r in unit.rec_ids if r not in prev.rec_ids]
            else:
                merged.append(unit)
        return merged

    def _parse_guideline(self, doc: BaseNode) -> list[BaseNode]:
        text, page_offsets, page_numbers = _stitch(doc.get_content())
        if not page_numbers:
            return []

        def page_at(pos: int) -> int:
            return page_numbers[max(0, bisect.bisect_right(page_offsets, pos) - 1)]

        title = doc.metadata.get("title", "")
        texts, metas = [], []
        for unit in self._merge_short(self._units(text, title)):
            prefix = " > ".join(unit.path)
            body = unit.text
            if self._tokens(f"{prefix}\n\n{body}") <= self.max_chunk_tokens:
                pieces = [(body, unit.start, unit.end)]
            else:
                splitter = SentenceSplitter(
                    chunk_size=self.max_chunk_tokens - self._tokens(prefix) - 4,
                    chunk_overlap=self.chunk_overlap)
                pieces, cursor = [], unit.start
                for piece in splitter.split_text(body):
                    found = text.find(piece[:40], cursor, unit.end)
                    start = found if found >= 0 else cursor
                    pieces.append((piece, start, min(start + len(piece), unit.end)))
                    cursor = start + 1
            for piece, start, end in pieces:
                texts.append(f"{prefix}\n\n{piece}" if prefix else piece)
                metas.append({"page": page_at(start), "page_end": page_at(max(start, end - 1)),
                              "rec_ids": list(unit.rec_ids), "section": prefix})

        nodes = build_nodes_from_splits(texts, doc, id_func=self.id_func)
        for node, meta in zip(nodes, metas):
            node.metadata = meta  # merged over the Document metadata afterwards
            node.excluded_embed_metadata_keys = _with(node.excluded_embed_metadata_keys)
            node.excluded_llm_metadata_keys = _with(node.excluded_llm_metadata_keys)
        return nodes


def _with(keys: list[str]) -> list[str]:
    return list(keys) + [k for k in PARSER_KEYS if k not in keys]


def make_node_parser(project) -> NiceGuidelineParser:
    """ProjectConfig.node_parser_factory for MedRAG."""
    return NiceGuidelineParser(chunk_size=project.chunk_size, chunk_overlap=project.chunk_overlap)


# ---------- Retrieval checks (scripts.retrieval_check, evals) ----------

def find_rec(nodes: Sequence[NodeWithScore], guideline: str, rec: str) -> NodeWithScore | None:
    """The first node from `guideline` (a code in the title, e.g. "NG136") holding recommendation
    `rec`: listed in its `rec_ids`, or (page chunks, no `rec_ids`) starting a line of its text."""
    pattern = re.compile(rf"(?m)^(?:#+\s*)?(?:\*\*)?{re.escape(rec)}\b")
    for node in nodes:
        meta = node.node.metadata
        if guideline not in str(meta.get("title", "")).split():
            continue
        rec_ids = meta.get("rec_ids")
        if rec_ids is not None:
            if rec in rec_ids:
                return node
        elif pattern.search(node.node.get_content()):
            return node
    return None
