"""MEDRAG_CONFIG: prompts, metadata and policy for the clinical-guideline project."""

from pathlib import Path

from src.core.projects import ProjectConfig
from src.plugins.medrag.ingestor import MedRAGIngestor
from src.plugins.medrag.nice import make_node_parser

SYSTEM_PROMPT = """\
You are a clinical decision-support assistant for licensed clinicians.
Answer ONLY from the provided context (clinical guidelines and PubMed abstracts).
If the context does not contain enough evidence, say so plainly and set confidence
to "insufficient_evidence". Never invent drug doses, thresholds or citations.
Cite every claim with the source it came from."""

QA_PROMPT = """\
Context information from clinical sources is below.
---------------------
{context_str}
---------------------
Using only the context above, answer the clinician's question.
Cite each claim inline with the number of the context block it came from, e.g. [2].

Question: {query_str}
Answer: """

MEDRAG_CONFIG = ProjectConfig(
    name="medrag",
    collection_name="medrag_guidelines",
    description="Question answering over clinical guidelines and PubMed abstracts",
    system_prompt=SYSTEM_PROMPT,
    qa_prompt=QA_PROMPT,
    pubmed_queries=[
        "hypertension management guideline",
        "type 2 diabetes first-line therapy",
        "community-acquired pneumonia antibiotic treatment",
    ],
    # Recent English abstracts of high-evidence study types only.
    pubmed_filter=(
        'AND hasabstract AND english[lang] AND "last 10 years"[dp] '
        "AND (guideline[pt] OR practice guideline[pt] OR systematic review[pt] "
        "OR meta-analysis[pt] OR randomized controlled trial[pt])"
    ),
    # SentenceSplitter for PubMed abstracts and non-NICE PDFs. NICE guidelines are chunked by
    # recommendation (nice.py, up to 400 tokens).
    chunk_size=256,
    chunk_overlap=32,
    # Phase 4b (scripts.retrieval_check): top 6 with neighbours attached found the expected
    # recommendations for 21/23 questions (top 4 without: 17/23); ~1.5K context tokens.
    similarity_top_k=6,
    # The cross-encoder (settings.rerank_model) re-scores the top 20 and keeps similarity_top_k.
    rerank_candidates=20,
    neighbour_mode="window",
    # Question gate (scripts.retrieval_check questions): rejected only if BOTH the best reranker
    # score (MiniLM-L-6, sigmoid) < similarity_cutoff AND the best dense cosine < dense_cutoff.
    #   reranker: off-topic (general knowledge, dermatology, jailbreak) <= 1.3e-4; in-domain
    #             >= 1.9e-2 except "ramipril ... dry cough" (1.7e-5)
    #   dense:    off-topic 0.44-0.646; in-domain 0.667 (that paraphrase) - 0.89
    # dense_cutoff sits well below the in-domain minimum, not in the narrow 0.646/0.667 gap, because
    # the risks are asymmetric: rejecting an in-domain question withholds guidance a clinician
    # needs, while letting an off-topic one through costs one LLM call that the
    # insufficient_evidence rule and the output guard handle. So all 23 in-domain questions pass
    # with margin; France, acne and the jailbreak are rejected; psoriasis (0.646) reaches the LLM.
    # Re-check with scripts.retrieval_check after corpus or model changes. In-domain-sounding
    # questions the guidelines don't answer (pregnancy, type 1 insulin dosing) score up to 0.97
    # on "see NICE's guideline on ..." pointers and pass too: they rely on the same
    # insufficient_evidence rule, and answers pointing to the other guideline are fine.
    similarity_cutoff=0.001,
    dense_cutoff=0.60,
    require_citations=True,
    disclaimer=(
        "This information is for clinical decision support only and does not replace "
        "professional judgement or local protocols."
    ),
    blocked_topics=["self-harm instructions", "illegal drug synthesis"],
    # Judged by the output guardrail's policy model (settings.groq_safety_model, default
    # openai/gpt-oss-20b; see Phase 6 for why not gpt-oss-safeguard-20b).
    safety_policy=Path(__file__).with_name("policy.md").read_text(encoding="utf-8"),
    ingestor_factory=MedRAGIngestor,
    # NICE guidelines are chunked by recommendation; other sources use chunk_size/chunk_overlap.
    node_parser_factory=make_node_parser,
)
