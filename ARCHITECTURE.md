# PLM Factory — Confirmed Architecture (v2, 2026-07-20)

**This supersedes PROPOSAL.md's 3-CodeAgent design.** Confirmed shape:
**1 Retrieval Agent + 1 Research Agent + 1 PLM Agent** (single agent, course passed as a
parameter). Carried over from v1: item schema, ARTS specifics, ground-truth oracle policy,
museum API option, Qwen2.5-Coder-32B model choice.

## Courses & perceptual skills
- **MATH 223** — vector field plots → instant judgment of div/curl/flux/gradient sign & behavior
- **ARH 201/202** — classify style/period/artist from a fragment or detail (292 does not exist)
- **CHEM 241A** — recognize functional groups, stereochemistry, mechanism patterns at a glance

Sessions: 10–15 min, many short trials (~60/session target), interleaved, immediate feedback.

## Non-negotiable PLM rules (Kellman, Massey & Son 2010; Kellman & Massey 2013)
Classification not problem-solving · high exemplar variability · interleaving never blocking ·
ARTS sequencing (accuracy + RT; retire at ~4 consecutive correct under category RT threshold —
~8 s simple visual, ~15 s multi-representation) · fluency = accuracy + speed (RT logged
first-class) · held-out transfer set.

## Data sources
- RAG corpus (textbooks): MATH 223 → OpenStax Calculus Vol 3 —
  **`~/Downloads/calculus-volume-3_-_WEB.pdf` (located)**; CHEM 241 → OpenStax Organic Chemistry
  (McMurry 10e) — **`~/Downloads/organic-chemistry_-_WEB.pdf` (located)**; ARH 201 → ASCCC OERI
  Intro to Art History I (2022); ARH 202 → Smarthistory period guides.
  Bonus found: `~/Downloads/vector calculus practice exam 1 MIT OCW.pdf`.
- HF datasets: CHEM → jablonkagroup/MaCBench, InnovatorLab/OpenRxn (validation/fallback only);
  ARH → huggan/wikiart (81k, artist/style/genre), Artificio/WikiArt (27 style categories) —
  primary ARH item source. MATH → none; procedural via sympy.
- Optional ARH supplement: Met / AIC / Rijksmuseum / Wikimedia open APIs — attribution must come
  from API record cross-checked vs a second source; optional CLIP outlier gate (transformers).

## Item engines (deterministic ground truth — NEVER LLM-generated)
- MATH 223: `sympy.vector` div/curl/gradient — exact keys, infinite instances
- CHEM 241: `rdkit` — SMILES→2D render, SMARTS functional-group detection, CIP stereo assignment
- ARH: WikiArt (+ optional museum APIs) + pillow crops/composites of diagnostic regions

**Hard rule:** answer keys always from a code oracle (sympy / RDKit / verified metadata),
never LLM assertion. Most critical for CHEM mechanisms.

## The three agents
1. **Retrieval Agent** — one agent, three Chroma collections (`math223`, `arh`, `chem241`);
   semantic-only first, add rank_bm25 + Reciprocal Rank Fusion ONLY after observing real
   retrieval misses on exact-term queries. One tool: `retrieve_course_context(course, query)`.
   Chunking via langchain-text-splitters; Chroma via langchain-community (sunsetting upstream —
   keep rank_bm25 directly callable without the wrapper). Grounds concepts only; never
   paraphrased into item content or answer keys.
2. **Research Agent** (separate from Retrieval — different caching behavior) — web search to
   deepen concept understanding; runs **once per topic, not per item**; caches concept briefs
   (key facts, misconceptions, canonical examples) to disk/Chroma; HTML→Markdown via
   markdownify; textbook is ground truth, web is supplementary — flag disagreements.
3. **PLM Agent** (single CodeAgent, course as parameter) — receives retrieval context + cached
   research brief + adaptive sequencing state; CoT decides WHICH category / WHAT difficulty /
   WHAT instance — never generates raw content; hands parameters to one consolidated tool per
   course (smolagents "simplify the workflow" guidance):
   `generate_math223_item(concept, difficulty)` / `generate_chem241_item(...)` /
   `generate_arh_item(...)` → `(image_path, answer_key)`.

### Runtime guardrail — step_callbacks
`guard_ground_truth(memory_step: ActionStep, agent)` registered via `step_callbacks` at agent
creation; flags steps that look like direct content/answer-key generation without calling a
unified tool. **Verified on this machine (smolagents 1.25.0): `ActionStep.code_action` exists**
(fields: step_number, timing, model_input_messages, tool_calls, error, model_output_message,
model_output, code_action, observations, observations_images, action_output, token_usage,
is_final_answer). Lightweight first line of defense; Phoenix is the full trace review.

## Adaptive sequencing (NOT a 4th agent)
Deterministic external tracker (dict/small DB). Priority ↑ with error rate and slow-but-correct;
recent presentation suppresses (spacing). Retirement ~4 consecutive correct under RT threshold.
PLM agent reads/writes state each trial; no LLM reasoning in the loop.

## Item schema
Same as v1 (see PROPOSAL.md §6): id, course, category, subcategory, stimulus, prompt, choices,
correct, feedback, ground_truth_method, difficulty, transfer, provenance{generator, seed}.
ARH adds provenance.source (wikiart|museum), .object_id, .verified_by.

## Delivery — Gradio, two surfaces
Dev/debug: smolagents built-in GradioUI trace. Student: separate minimal Gradio — image + timed
response buttons only, no agent reasoning visible.

## Model & runtime
`InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")` (HF Inference API).
**Blocker: HF token still not configured — `hf auth login` required before live runs.**
Local executor to start; Docker/E2B only if generation code grows risky.

## Observability — Arize Phoenix (use AFTER agent is built)
Installed on this machine: **arize-phoenix 19.2.0, arize-phoenix-otel 0.16.1**, and (added
2026-07-20) **openinference-instrumentation-smolagents 0.1.32**. Integration pattern:
```python
from phoenix.otel import register
tracer_provider = register(project_name="plm-factory", auto_instrument=True)
# or explicitly:
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)
```
Run `phoenix serve` → UI at http://localhost:6006 (collector gRPC :4317); fully local, no data
leaves the machine. Docs: https://arize.com/docs/phoenix/integrations/python/hugging-face-smolagents/smolagents-tracing
and https://huggingface.co/blog/smolagents-phoenix

## Stack (verified installed 2026-07-20)
smolagents 1.25.0, numpy, matplotlib, pillow, sympy, rdkit 2026.3.4, datasets 5.0.0,
transformers 5.12.1, **chromadb 1.5.9 (installed today — smoke-tested: collection + semantic
query OK, default embedder all-MiniLM-L6-v2 cached)**, rank_bm25, langchain-text-splitters,
langchain-community, markdownify, gradio 6.20, arize-phoenix 19.2.0,
openinference-instrumentation-smolagents 0.1.32. Not needed: edgartools/yfinance/pandas.

## Explicitly rejected
- 3 per-course CodeAgents (superseded — do not build)
- Canvas/freeform drawing tools (items are deterministic single-shot renders)
- Web search folded into Retrieval Agent (Research Agent stays separate — different caching)

## Build order
1. Retrieval Agent (Chroma ×3, semantic-only)
2. MATH 223 generation (sympy)
3. CHEM 241 generation (RDKit; hand-validate early vs MaCBench)
4. ARH generation (WikiArt + Pillow; museum APIs optional)
5. PLM Agent wired around all three tools (course as parameter)
6. Adaptive sequencing tracker
7. Research Agent + markdownify caching
8. Student-facing Gradio surface (last)
