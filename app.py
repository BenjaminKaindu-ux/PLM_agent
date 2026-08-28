"""PLM Factory — multi-course student drill + a constrained live debug tab for the
multi-agent PLM pipeline (Retrieval/PLM/Research agents; MATH223, CHEM241A, ARH).

Timed, interleaved classification trials with ARTS adaptive sequencing. The student
drill calls each course's generator directly (sympy/RDKit/WikiArt-label oracle) with NO
agent/LLM in the per-trial loop — perceptual learning requires seconds-fast trials, and
the PLM Agent's LLM round trips (2-30s observed) would break that. The agent's LLM
reasoning belongs in the separate debug tab / offline item pre-generation, not live
per-trial sequencing.

The agent debug tab deliberately does NOT use smolagents' built-in `GradioUI` (a raw
free-text chat into a `CodeAgent`): on a public Space that would let any visitor prompt
the agent into executing arbitrary generated Python in the container. Fixed
course/concept dropdowns keep the same live agentic pipeline (and its code trace)
visible without that open-ended attack surface.
"""

import json
import re
import time

import gradio as gr
from smolagents import ActionStep

from src.generators import arh, chem241a, math223
from src.plm_core.arts import ArtsTracker, CategoryState
from src.plm_core.plm_agent import build_plm_agent

MAX_CHOICES = 5  # widest choice set across courses (CHEM241A functional groups, ARH styles)

# Shared by both the student drill and the agent debug tab below.
# ARH categories are re-derived once at import time, not per-access, since
# active_categories() reads manifest.json on every call — cache the result.
_ACTIVE_ARH_CATEGORIES = None


def _get_active_arh_categories():
    global _ACTIVE_ARH_CATEGORIES
    if _ACTIVE_ARH_CATEGORIES is None:
        from src.generators import arh as arh_gen
        _ACTIVE_ARH_CATEGORIES = arh_gen.active_categories()
    return _ACTIVE_ARH_CATEGORIES


COURSES = {
    "MATH223": {
        "make_item": math223.make_item,
        "categories": lambda: math223.CATEGORIES,
        "blurb": (
            "**FieldSense** — judge **divergence**/**curl** sign from the field pattern at "
            "the marked point, in seconds, by eye. Answer keys are sympy-verified."
        ),
    },
    "CHEM241A": {
        "make_item": chem241a.make_item,
        "categories": lambda: chem241a.CATEGORIES,
        "blurb": (
            "**GroupSense / StereoSense** — spot the functional group, or the CIP (R/S) "
            "configuration at a highlighted stereocenter, at a glance. Answer keys are "
            "RDKit-verified."
        ),
    },
    "ARH": {
        "make_item": arh.make_item,
        "categories": _get_active_arh_categories,
        "blurb": (
            "**StyleSense** — classify the art style/period from a cropped detail. Answer "
            "keys are WikiArt's own curated style labels."
        ),
    },
},


def new_session(course: str) -> ArtsTracker:
    categories = COURSES[course]["categories"]()
    return ArtsTracker([CategoryState(name=n, rt_threshold_s=s["rt_threshold_s"]) for n, s in categories.items()])


def next_trial(course: str, tracker: ArtsTracker | None):
    if tracker is None:
        tracker = new_session(course)
    if tracker.all_retired() and tracker.trial > 0:
        done = "🎉 **All categories retired — session mastered!** Press *New session* to restart."
        return tracker, None, gr.Image(visible=False), done, *[gr.Button(visible=False)] * MAX_CHOICES, tracker.summary(), None
    cat = tracker.next_category()
    item = COURSES[course]["make_item"](cat)
    buttons = [
        gr.Button(value=item["choices"][i], visible=True, interactive=True) if i < len(item["choices"])
        else gr.Button(visible=False)
        for i in range(MAX_CHOICES)
    ]
    return (
        tracker,
        item,
        gr.Image(value=item["stimulus"]["image"], visible=True),
        f"### {item['prompt']}",
        *buttons,
        tracker.summary(),
        time.time(),
    )


def answer(idx: int, course: str, tracker: ArtsTracker, item: dict, t0: float):
    if item is None or t0 is None:
        return tracker, "Press **Next trial** to begin.", tracker.summary() if tracker else []
    rt = time.time() - t0
    correct = idx == item["correct"]
    tracker.record(item["subcategory"], correct, rt)
    verdict = "✅ **Correct**" if correct else f"❌ **Incorrect** — answer: *{item['choices'][item['correct']]}*"
    threshold = COURSES[course]["categories"]()[item["subcategory"]]["rt_threshold_s"]
    speed = "⚡ fast enough to count toward mastery" if rt < threshold else f"🐢 over the {threshold:.0f}s fluency threshold"
    msg = f"{verdict}  ·  {rt:.1f}s ({speed})\n\n{item['feedback']}\n\n*Press **Next trial** to continue.*"
    return tracker, msg, tracker.summary()


def _switch_course(course: str):
    tr, item, img_u, prompt_u, *btns, stats_u, t0_u = next_trial(course, new_session(course))
    return COURSES[course]["blurb"], tr, item, img_u, prompt_u, *btns, stats_u, t0_u


with gr.Blocks() as student_demo:
    gr.Markdown("# 🧠 PLM Factory — Student Drill")
    course_dd = gr.Dropdown(choices=list(COURSES), value="MATH223", label="Course")
    blurb_md = gr.Markdown(COURSES["MATH223"]["blurb"])
    gr.Markdown(
        "Categories retire after 4 consecutive fast-and-correct answers (ARTS adaptive "
        "sequencing — accuracy **and** response time; Kellman, Massey & Son 2010)."
    )
    tracker_s, item_s, t0_s = gr.State(None), gr.State(None), gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            img = gr.Image(visible=False, show_label=False, type="pil")
            prompt_md = gr.Markdown("Press **Next trial** to begin.")
            with gr.Row():
                btns = [gr.Button(visible=False) for _ in range(MAX_CHOICES)]
            feedback_md = gr.Markdown("")
            with gr.Row():
                next_btn = gr.Button("Next trial ▶", variant="primary")
                reset_btn = gr.Button("New session ⟳")
        with gr.Column(scale=2):
            gr.Markdown("### Session progress")
            stats = gr.Dataframe(interactive=False)

    trial_outputs = [tracker_s, item_s, img, prompt_md, *btns, stats, t0_s]

    next_btn.click(next_trial, [course_dd, tracker_s], trial_outputs).then(lambda: "", None, feedback_md)
    reset_btn.click(lambda c: next_trial(c, new_session(c)), [course_dd], trial_outputs).then(lambda: "", None, feedback_md)
    course_dd.change(_switch_course, [course_dd], [blurb_md, *trial_outputs]).then(lambda: "", None, feedback_md)
    for i, b in enumerate(btns):
        b.click(
            lambda tr, c, it, t0, i=i: answer(i, c, tr, it, t0),
            [tracker_s, course_dd, item_s, t0_s],
            [tracker_s, feedback_md, stats],
        )


# ---------------------------------------------------------------------------
# Agent debug tab — live PLM Agent runs, gated to fixed dropdowns (see module
# docstring for why this isn't a free-text chat).
# ---------------------------------------------------------------------------

_COURSE_CONCEPTS = {course: list(cfg["categories"]().keys()) for course, cfg in COURSES.items()}

# Built once at Space startup, not per-click: each build_plm_agent() call constructs a
# fresh CodeAgent + retrieval sub-agent, which is wasted work if repeated every request.
_PLM_AGENTS = {course: build_plm_agent(course) for course in _COURSE_CONCEPTS}


def _update_concept_choices(course: str):
    concepts = _COURSE_CONCEPTS[course]
    return gr.Dropdown(choices=concepts, value=concepts[0])


def _summarize_trace(agent) -> str:
    lines = []
    for step in agent.memory.steps:
        if isinstance(step, ActionStep) and step.code_action:
            lines.append(f"--- step {step.step_number} ---\n{step.code_action}")
    return "\n\n".join(lines) or "(no steps captured)"


_IMAGE_PATH_RE = re.compile(r"data/items/\S+?\.png")


def run_plm_agent(course: str, concept: str):
    agent = _PLM_AGENTS[course]
    task = (
        f"Generate one {course} item for concept {concept!r}. Call the sanctioned "
        "generator tool, then report only image_path and correct_answer."
    )
    result = agent.run(task, reset=True)  # reset=True: no cross-visitor memory buildup
    trace = _summarize_trace(agent)

    # The final packaging step's exact shape isn't guaranteed by the prompt alone — observed
    # a dict, a plain [image_path, correct_answer] list, AND a free-form descriptive string
    # across different live runs/courses. Regexing the path out of the stringified result is
    # robust to all three rather than trying to enumerate every container shape.
    result_str = result if isinstance(result, str) else json.dumps(result, default=str)
    match = _IMAGE_PATH_RE.search(result_str)
    image_path = match.group(0) if match else None
    answer_json = result_str if isinstance(result, str) else json.dumps(result, indent=2, default=str)
    return image_path, answer_json, trace


with gr.Blocks() as agent_debug:
    gr.Markdown(
        "# 🤖 PLM Agent — live debug\n"
        "Each click runs a real `smolagents` `CodeAgent` (Qwen2.5-Coder-32B-Instruct via the "
        "HF Inference API) that decides the item parameters and calls the course's sanctioned "
        "generator tool — sympy (MATH223), RDKit (CHEM241A), or a curated WikiArt exemplar "
        "(ARH) computes the actual answer key. The code trace below shows exactly what the "
        "agent ran, so you can verify it never asserts an answer itself."
    )
    with gr.Row():
        course_dd = gr.Dropdown(choices=list(_COURSE_CONCEPTS), value="MATH223", label="Course")
        concept_dd = gr.Dropdown(
            choices=_COURSE_CONCEPTS["MATH223"], value=_COURSE_CONCEPTS["MATH223"][0], label="Concept"
        )
    run_btn = gr.Button("Generate item ▶", variant="primary")
    with gr.Row():
        with gr.Column():
            out_img = gr.Image(label="Generated stimulus", type="filepath")
            out_json = gr.Code(label="Answer key", language="json")
        out_trace = gr.Code(label="Agent code trace", language="python")

    course_dd.change(_update_concept_choices, [course_dd], [concept_dd])
    run_btn.click(run_plm_agent, [course_dd, concept_dd], [out_img, out_json, out_trace])


with gr.Blocks(title="PLM Factory") as demo:
    gr.Markdown("# PLM Factory")
    with gr.Tab("MATH223 Student Demo"):
        student_demo.render()
    with gr.Tab("Agent Debug (dev)"):
        agent_debug.render()

if __name__ == "__main__":
    demo.launch()
