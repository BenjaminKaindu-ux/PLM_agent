"""Research Agent — web search to deepen concept understanding (ARCHITECTURE.md).

Separate from the Retrieval Agent, with different caching behavior: this runs once per
topic (not per item) and caches a concept brief to disk, keyed on (course, topic). The
textbook (Retrieval Agent) is ground truth; this is supplementary context only — never
a source of answer keys, and any web/textbook disagreement must be flagged, not silently
resolved by picking one.
"""

import json
import re
from pathlib import Path

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, VisitWebpageTool

from src.config import RESEARCH_AGENT_MODEL

CACHE_DIR = Path("data/research_cache")


def _slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def _cache_path(course: str, topic: str) -> Path:
    return CACHE_DIR / course / f"{_slug(topic)}.json"


def build_research_agent() -> CodeAgent:
    return CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=InferenceClientModel(model_id=RESEARCH_AGENT_MODEL),
        name="research_agent",
        description=(
            "Researches a course concept on the web and produces a concise, cited brief. "
            "Never used as a source of answer keys — grounding/context only."
        ),
    )


def get_concept_brief(course: str, topic: str, force_refresh: bool = False) -> dict:
    """Cached per (course, topic) — the whole point is NOT re-researching per item."""
    cache_path = _cache_path(course, topic)
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text())

    agent = build_research_agent()
    brief = agent.run(
        f"Research the concept '{topic}' as it's taught in a {course} course. Search the web "
        "and visit a couple of the most relevant, reputable pages (educational sites, "
        "reference sites — not forums). Then call final_answer with a JSON-serializable dict "
        "with exactly these keys: "
        "'key_facts' (list of 3-5 short factual bullet strings), "
        "'misconceptions' (list of 1-3 common student misconceptions about this concept), "
        "'canonical_examples' (list of 1-2 short canonical example descriptions), "
        "'disagreements' (list of strings — note here if different sources describe the "
        "concept inconsistently; empty list if none), "
        "'sources' (list of the URLs you actually visited). "
        "Do not include any keys other than these five. Be concise — bullets, not paragraphs."
    )
    if not isinstance(brief, dict):
        raise RuntimeError(f"Research Agent returned non-dict final answer: {brief!r}")

    brief = {"course": course, "topic": topic, **brief}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(brief, indent=2))
    return brief
