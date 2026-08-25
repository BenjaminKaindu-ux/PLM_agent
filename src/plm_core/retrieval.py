"""Retrieval Agent — one smolagents agent, three Chroma collections (math223/chem241/arh).

Single tool per ARCHITECTURE.md: `retrieve_course_context(course, query)`. Semantic-only
via bge-m3 embeddings (HF Inference API); BM25+RRF is deferred until real retrieval misses
are observed. Grounds concepts only — never paraphrased into item content or answer keys
(that stays the PLM Agent's job, backed by code oracles).
"""

import chromadb
from smolagents import CodeAgent, InferenceClientModel, Tool

from src.config import CHROMA_COLLECTIONS, CHROMA_PATH, PLM_AGENT_MODEL
from src.plm_core.embeddings import embed_query

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def retrieve_course_context(course: str, query: str, k: int = 5) -> list[dict]:
    """Semantic search over one course's textbook chunks. Returns [] if the
    collection is empty (e.g. ARH — no local corpus yet)."""
    if course not in CHROMA_COLLECTIONS:
        raise ValueError(f"unknown course {course!r}; expected one of {list(CHROMA_COLLECTIONS)}")
    coll = _get_client().get_collection(CHROMA_COLLECTIONS[course])
    if coll.count() == 0:
        return []
    result = coll.query(query_embeddings=[embed_query(query)], n_results=k)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"]}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]


class RetrieveCourseContextTool(Tool):
    name = "retrieve_course_context"
    description = (
        "Semantic search over a course's textbook for grounding context on a concept. "
        "Use this to check terminology, definitions, and canonical examples before generating "
        "a PLM item — never invent facts the textbook can confirm."
    )
    inputs = {
        "course": {"type": "string", "description": "One of: MATH223, CHEM241A, ARH"},
        "query": {"type": "string", "description": "The concept or term to look up"},
    }
    output_type = "array"

    def forward(self, course: str, query: str) -> list[dict]:
        return retrieve_course_context(course, query)


def build_retrieval_agent() -> CodeAgent:
    """CodeAgent, not ToolCallingAgent: the HF Inference providers currently serving
    Qwen2.5-Coder-32B-Instruct (nscale/featherless-ai) reject the `tools`/`tool_choice`
    request params outright (422 UNSUPPORTED_OPENAI_PARAMS) — native function-calling
    isn't available for this model on the free API. CodeAgent drives tools via generated
    Python instead, which only needs plain chat completion."""
    return CodeAgent(
        tools=[RetrieveCourseContextTool()],
        model=InferenceClientModel(model_id=PLM_AGENT_MODEL),
        name="retrieval_agent",
        description=(
            "Answers concept-grounding questions for MATH223, CHEM241A, or ARH by searching "
            "the course textbook. Give it a course and a specific concept/term."
        ),
    )
