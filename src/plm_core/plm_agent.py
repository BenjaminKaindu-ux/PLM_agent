"""PLM Agent — single CodeAgent, course passed as a parameter (ARCHITECTURE.md).

CoT decides WHICH category / WHAT difficulty / WHAT instance — it never generates raw
item content itself. Every item comes from one consolidated, course-specific tool that
wraps a deterministic generator (sympy for MATH223, RDKit for CHEM241A, WikiArt-label
lookup for ARH). That tool call is the only sanctioned path to a persisted image + answer key.
"""

import re
from pathlib import Path

from smolagents import CodeAgent, InferenceClientModel, Tool

from src.config import PLM_AGENT_MODEL
from src.generators import arh as arh_gen
from src.generators import chem241a as chem241a_gen
from src.generators import math223 as math223_gen
from src.plm_core.retrieval import build_retrieval_agent

ITEMS_DIR = Path("data/items")


def _persist_item(item: dict, course: str) -> dict:
    """Save the stimulus image to disk and return everything else as the answer key —
    the (image_path, answer_key) contract from ARCHITECTURE.md, flattened into one dict.

    Includes `correct_answer` (the resolved choice text, not just the `correct` index):
    an agent doing its own `choices[correct]` arithmetic in generated code can get the
    off-by-one wrong even when the index itself came straight from the sanctioned tool —
    observed this exact failure in testing. Pre-resolving removes that arithmetic entirely."""
    out_dir = ITEMS_DIR / course
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{item['id']}.png"
    item["stimulus"]["image"].save(img_path)
    answer_key = {k: v for k, v in item.items() if k != "stimulus"}
    answer_key["image_path"] = str(img_path)
    answer_key["correct_answer"] = item["choices"][item["correct"]]
    return answer_key


class GenerateMath223ItemTool(Tool):
    name = "generate_math223_item"
    description = (
        "The ONLY way to create a MATH223 FieldSense item. Never invent a divergence/curl "
        "answer yourself — this computes it symbolically with sympy and returns a saved "
        "stimulus image plus the verified answer key. Read the returned `correct_answer` "
        "field directly (already resolved to the right choice text) — don't index "
        "`choices` with `correct` yourself, that's an easy off-by-one to get wrong."
    )
    inputs = {
        "concept": {"type": "string", "description": "One of: divergence_sign, curl_sign"},
        "difficulty": {
            "type": "integer",
            "description": "Difficulty tier, reserved for future use — pass 1",
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(self, concept: str, difficulty: int = 1) -> dict:
        item = math223_gen.make_item(concept, difficulty=difficulty)
        return _persist_item(item, "MATH223")


class GenerateArhItemTool(Tool):
    name = "generate_arh_item"
    description = (
        "The ONLY way to create an ARH StyleSense item. Never assert a painting's style "
        "yourself — this crops a real WikiArt exemplar and returns the dataset's own "
        "curated style label as the verified answer key. Read the returned `correct_answer` "
        "field directly (already resolved to the right choice text) — don't index "
        "`choices` with `correct` yourself, that's an easy off-by-one to get wrong. Call "
        "arh_active_categories first if you don't already know which styles are available."
    )
    inputs = {
        "concept": {
            "type": "string",
            "description": "A style name from arh_active_categories, e.g. Baroque",
        },
        "difficulty": {
            "type": "integer",
            "description": "Difficulty tier, reserved for future use — pass 1",
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(self, concept: str, difficulty: int = 1) -> dict:
        item = arh_gen.make_item(concept, difficulty=difficulty)
        return _persist_item(item, "ARH")


class GenerateChem241ItemTool(Tool):
    name = "generate_chem241_item"
    description = (
        "The ONLY way to create a CHEM241A item. Never assert a functional group or CIP "
        "(R/S) configuration yourself — this uses RDKit's own maintained fragment counters "
        "and CIP algorithm and returns the verified answer key. Read the returned "
        "`correct_answer` field directly (already resolved to the right choice text) — "
        "don't index `choices` with `correct` yourself, that's an easy off-by-one to get wrong."
    )
    inputs = {
        "concept": {
            "type": "string",
            "description": "One of: functional_group_id, stereo_center_config",
        },
        "difficulty": {
            "type": "integer",
            "description": "Difficulty tier, reserved for future use — pass 1",
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(self, concept: str, difficulty: int = 1) -> dict:
        item = chem241a_gen.make_item(concept, difficulty=difficulty)
        return _persist_item(item, "CHEM241A")


class ArhActiveCategoriesTool(Tool):
    name = "arh_active_categories"
    description = "Lists which ARH style categories currently have cached exemplars and can be generated."
    inputs = {}
    output_type = "array"

    def forward(self) -> list[str]:
        return arh_gen.active_categories()


_GUARDED_TOOLS = ("generate_math223_item", "generate_arh_item", "generate_chem241_item")
_ASSERTION_PATTERN = re.compile(r"\b(correct|answer_key|ground_truth|is_correct)\s*=")


def guard_ground_truth(memory_step, agent) -> None:
    """Lightweight first line of defense (ARCHITECTURE.md): flags steps whose code looks
    like it's asserting an answer key directly instead of calling a sanctioned generator
    tool. Not a hard block — Phoenix tracing is the full review, once wired."""
    code = getattr(memory_step, "code_action", None)
    if not code:
        return
    calls_sanctioned_tool = any(name in code for name in _GUARDED_TOOLS)
    if _ASSERTION_PATTERN.search(code) and not calls_sanctioned_tool:
        agent.logger.log(
            f"[guard_ground_truth] step {memory_step.step_number}: code assigns an "
            f"answer/ground-truth-looking variable without calling {_GUARDED_TOOLS} — "
            "verify this isn't an LLM-asserted answer key:\n" + code,
            level=1,  # LogLevel.INFO — visible but non-blocking
        )


def build_plm_agent(course: str) -> CodeAgent:
    """One PLM Agent, course passed as a parameter — not one agent per course."""
    if course == "MATH223":
        tools = [GenerateMath223ItemTool()]
    elif course == "ARH":
        tools = [GenerateArhItemTool(), ArhActiveCategoriesTool()]
    elif course == "CHEM241A":
        tools = [GenerateChem241ItemTool()]
    else:
        raise ValueError(f"no generator wired for course {course!r}")

    return CodeAgent(
        tools=tools,
        model=InferenceClientModel(model_id=PLM_AGENT_MODEL),
        managed_agents=[build_retrieval_agent()],
        step_callbacks=[guard_ground_truth],
        name=f"plm_agent_{course.lower()}",
        description=f"Generates perceptual learning items for {course}.",
    )
