"""MEDRAG_CONFIG: prompts, metadata and policy for the clinical-guideline project."""

from src.core.projects import ProjectConfig

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
Include citations (source name and a short supporting snippet) for each claim.

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
    chunk_size=512,
    chunk_overlap=64,
    similarity_top_k=4,
    require_citations=True,
    disclaimer=(
        "This information is for clinical decision support only and does not replace "
        "professional judgement or local protocols."
    ),
    blocked_topics=["self-harm instructions", "illegal drug synthesis"],
)
