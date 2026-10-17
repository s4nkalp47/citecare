"""MEDRAG_CONFIG: prompts, metadata and policy for the clinical-guideline project."""

from pathlib import Path

from src.core.projects import ProjectConfig
from src.plugins.medrag.ingestor import MedRAGIngestor

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
    chunk_size=512,
    chunk_overlap=64,
    similarity_top_k=4,
    # Calibrated on bge-small-en-v1.5 over NG136/NG28/NG250: in-domain questions scored
    # 0.73-0.86, off-topic ones (general knowledge, dermatology, jailbreak) 0.42-0.64.
    # Set at the low end of the gap; the prompt's insufficient_evidence rule backs it up.
    similarity_cutoff=0.65,
    require_citations=True,
    disclaimer=(
        "This information is for clinical decision support only and does not replace "
        "professional judgement or local protocols."
    ),
    blocked_topics=["self-harm instructions", "illegal drug synthesis"],
    # Judged by the output guardrail's safeguard model (gpt-oss-safeguard).
    safety_policy=Path(__file__).with_name("policy.md").read_text(encoding="utf-8"),
    ingestor_factory=MedRAGIngestor,
)
