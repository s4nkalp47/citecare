"""Streamlit chat UI for MedRAG. Every request goes through the HTTP API (and so its guardrails).

Run from the project root, with the API already running:
    uvicorn src.api.main:app
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import streamlit as st

from src.api.schemas import BlockedResponse
from src.core.schemas import RAGResponse, SourceInfo
from src.ui.client import APIError, APIUnavailableError, MedRAGClient
from src.ui.formatting import citation_heading, confidence_badge, source_type_label

ALL_SOURCES = "All sources"

st.set_page_config(page_title="MedRAG", page_icon=":material/medical_services:", layout="wide")


@st.cache_resource
def get_client() -> MedRAGClient:
    return MedRAGClient()


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("uploader_key", 0)
    st.session_state.setdefault("pending_delete", None)


# ---------- Rendering ----------

def render_answer(response: RAGResponse) -> None:
    label, color = confidence_badge(response.confidence)
    st.badge(label, color=color)
    st.markdown(response.answer)
    if response.disclaimer:
        st.caption(response.disclaimer)
    if response.citations:
        with st.expander(f"Sources ({len(response.citations)})"):
            for citation in response.citations:
                # Citation.score is deliberately not shown (see src/ui/formatting.py).
                st.markdown(citation_heading(citation))
                if citation.snippet:
                    st.caption(citation.snippet)
    st.caption(f"Answered in {response.latency_ms / 1000:.1f} s")


def render_blocked(blocked: BlockedResponse) -> None:
    title = "Question blocked" if blocked.stage == "input" else "Answer withheld"
    st.warning(f"**{title}.** {blocked.message}", icon=":material/block:")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif isinstance(message.get("response"), BlockedResponse):
            render_blocked(message["response"])
        elif isinstance(message.get("response"), RAGResponse):
            render_answer(message["response"])
        else:
            st.error(message.get("error") or "Something went wrong.", icon=":material/error:")


# ---------- Sidebar ----------

def sidebar_health(client: MedRAGClient) -> None:
    st.subheader("API")
    try:
        health = client.health()
    except APIError as exc:
        st.error(f"{exc}", icon=":material/cloud_off:")
        return
    if health.qdrant_reachable:
        st.success(f"Connected · {health.points or 0} chunks indexed", icon=":material/check:")
    else:
        st.warning("API up, but Qdrant is unreachable", icon=":material/warning:")
    st.caption(f"Project `{health.project}` · LLM `{health.models.llm}`")


def sidebar_upload(client: MedRAGClient) -> None:
    st.subheader("Add a guideline")
    uploaded = st.file_uploader("PDF", type=["pdf"], key=f"upload_{st.session_state.uploader_key}",
                                label_visibility="collapsed")
    if uploaded is not None and st.button("Upload and index", icon=":material/upload:",
                                          width="stretch"):
        with st.spinner("Parsing and indexing (this can take a few minutes)..."):
            try:
                source = client.upload_source(uploaded.name, uploaded.getvalue())
            except APIError as exc:
                st.error(f"Upload failed: {exc}")
                return
        st.session_state.uploader_key += 1  # clears the file uploader
        st.toast(f"Indexed {source.title} ({source.chunks} chunks)", icon=":material/check:")
        st.rerun()


def sidebar_sources(sources: list[SourceInfo] | None, client: MedRAGClient) -> None:
    st.subheader("Sources")
    if sources is None:
        st.caption("Unavailable while the API is unreachable.")
        return
    if not sources:
        st.caption("No sources indexed yet.")
        return
    for source in sources:
        with st.container(border=True):
            st.markdown(f"**{source.title}**")
            st.caption(f"{source_type_label(source.source_type)} · {source.chunks} chunks")
            if st.session_state.pending_delete == source.source_id:
                st.caption("Delete this source from the index?")
                confirm, cancel = st.columns(2)
                if confirm.button("Delete", key=f"confirm_{source.source_id}", type="primary"):
                    _delete(client, source)
                if cancel.button("Cancel", key=f"cancel_{source.source_id}"):
                    st.session_state.pending_delete = None
                    st.rerun()
            elif st.button("Delete", key=f"delete_{source.source_id}",
                           icon=":material/delete:"):
                st.session_state.pending_delete = source.source_id
                st.rerun()


def _delete(client: MedRAGClient, source: SourceInfo) -> None:
    st.session_state.pending_delete = None
    try:
        result = client.delete_source(source.source_id)
    except APIError as exc:
        st.error(f"Delete failed: {exc}")
        return
    st.toast(f"Deleted {source.title} ({result.chunks_removed} chunks)", icon=":material/delete:")
    st.rerun()


def load_sources(client: MedRAGClient) -> list[SourceInfo] | None:
    try:
        return client.list_sources()
    except APIError:
        return None


# ---------- Page ----------

def main() -> None:
    _init_state()
    client = get_client()
    sources = load_sources(client)

    with st.sidebar:
        st.title("MedRAG")
        sidebar_health(client)
        source_types = sorted({s.source_type for s in sources or []})
        source_filter = st.selectbox(
            "Search in", [ALL_SOURCES, *source_types],
            format_func=lambda t: t if t == ALL_SOURCES else source_type_label(t))
        st.divider()
        sidebar_upload(client)
        st.divider()
        sidebar_sources(sources, client)
        st.divider()
        if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    st.title("Clinical guideline assistant")
    st.caption("Answers are drawn from indexed guidelines and abstracts. "
               "Decision support only: not a substitute for clinical judgement.")

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Ask a clinical question")
    if not question:
        return
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    reply: dict = {"role": "assistant"}
    with st.spinner("Searching the guidelines..."):
        try:
            reply["response"] = client.ask(
                question, source_type=None if source_filter == ALL_SOURCES else source_filter)
        except APIUnavailableError as exc:
            reply["error"] = f"{exc}. Is `uvicorn src.api.main:app` running?"
        except APIError as exc:
            reply["error"] = f"The API returned an error: {exc}"
    st.session_state.messages.append(reply)
    render_message(reply)


main()
