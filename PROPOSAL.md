# PLM Factory — Multi-Agent System Proposal

> **⚠️ SUPERSEDED (2026-07-20):** the 3-CodeAgent architecture below was replaced by
> **1 Retrieval Agent + 1 Research Agent + 1 PLM Agent (course as parameter)** — see
> `ARCHITECTURE.md` in this directory for the confirmed design. Item schema (§6), ARTS spec
> (§7), ground-truth policy, and model choice remain valid and are referenced from there.

**Goal:** A smolagents-based multi-agent system that generates highly technical, research-grounded
Perceptual Learning Modules (PLMs) for three University of Arizona courses:
**CHEM 241A** (Organic Chemistry I), **MATH 223** (Vector Calculus), and the **ARH survey**
(note: ARH 292 does not exist in the UA catalog — the surveys are ARH 201/202; this proposal
targets **ARH 202**, trivially swappable).

**Agent budget (per spec):** 1 retrieval agent + 3 CodeAgents.

---

## 1. The science we must implement (cognitive psychology grounding)

PLMs (Kellman, Massey & Son 2010; Kellman & Massey 2013; UCLA Human Perception Lab) train
*pattern recognition, structure extraction, and fluency* — the "seeing" component of expertise
that lectures and problem sets don't train. The design principles below are **non-negotiable
requirements** for whatever our agents produce:

1. **Classification trials, not problem-solving.** Each trial asks the student to classify,
   discriminate, or map structure across representations (e.g., "which of these 3 graphs matches
   this equation?"), answered in seconds. Students never compute a full answer.
2. **Many short trials with immediate feedback.** Empirical modules ran ~60 trials/session,
   2+ sessions, 15–40 min each. Feedback is instant correct/incorrect, plus a brief
   structure-highlighting explanation on errors.
3. **High exemplar variability within categories.** Each category needs *dozens* of surface-varied
   exemplars so students learn the invariant structure, not the specific items.
4. **Interleaving, never blocking.** Categories are mixed trial-to-trial. Research (Kornell &
   Bjork; Kellman) shows interleaving — not mere spacing — drives inductive category learning.
5. **Adaptive Response-Time-based Sequencing (ARTS).** Item scheduling is driven by each
   learner's accuracy AND response time: a priority score per category governs spacing/sequencing;
   categories **retire** at mastery (canonical criterion: N consecutive correct with RT under
   threshold, e.g., 4–5 in a row under 5–15 s depending on trial type).
6. **Fluency = accuracy + speed.** RT is a first-class measurement, logged per trial.
7. **Transfer by design.** Assessment uses novel exemplars never seen in training; item banks
   flag a held-out transfer set.
8. **Evidence of effect:** Algebraic Transformations PLM cut equation-solving time from ~28 s to
   ~12 s with gains durable at 2-week follow-up; Linear Measurement PLM gains held at 4 months;
   analogous results in ECG reading, radiology, chemistry, and aviation.

---

## 2. Course-specific PLM specifications

### CHEM 241A — Organic Chemistry I (PLM fit: 9.5/10)
Category taxonomies (each becomes one module of interleaved classification trials):
- **FG-ID:** functional group identification in multi-group molecules.
- **Stereo:** chirality detection and R/S assignment at a glance.
- **RepMap:** same molecule across line-angle ↔ condensed formula ↔ Newman ↔ chair (pick the match).
- **MechClass:** substrate + reagent + conditions → SN1 / SN2 / E1 / E2 / addition / none.
- **ResoCheck:** valid vs. invalid resonance structures (electron-pushing legality).

Stimulus generation: programmatic 2D structure drawing. **Recommend adding RDKit**
(`pip install rdkit`) — it gives SMILES→2D depiction + correctness oracles (CIP R/S assignment,
canonical comparison) for free. Fallback without RDKit: matplotlib/pillow skeletal drawing
primitives written by the chem CodeAgent (workable but much more effort and error-prone).
Ground truth must come from rules/cheminformatics, never from LLM recall alone.

### MATH 223 — Vector Calculus (PLM fit: 7.5/10, targets the real bottlenecks)
- **Eq↔Graph:** equation ↔ 3D surface / contour plot matching (quadric surfaces, level sets).
- **FieldSense:** vector field plot → sign of divergence / curl at a point; field ↔ formula match.
- **CoordChoice:** region/integrand shown → best coordinate system (Cartesian/polar/cylindrical/spherical).
- **TheoremSelect:** problem setup → which theorem applies (FTLI / Green / Stokes / Divergence / direct).
- **CritPoint:** contour plot or Hessian info → local max / min / saddle.

Stimulus generation is fully in-stack: **sympy** computes symbolic ground truth (divergence, curl,
gradients, Hessians — guaranteed-correct answer keys), **matplotlib** renders surfaces, contours,
and quiver plots, **numpy** parameterizes infinite exemplar variation. This course gets *parametric
generative templates*: each template emits unlimited validated items.

### ARH 202 — Survey of Western Art, Renaissance→Modern (PLM fit: 9/10)
- **StyleClass:** unseen work → movement/period (Renaissance, Baroque, Rococo, Neoclassicism,
  Romanticism, Realism, Impressionism, Post-Impressionism, Cubism, Surrealism, AbEx…).
- **ArtistAttrib:** unseen work by a covered artist → which artist (within-period discrimination).
- **FeatureSpot:** image with region highlighted → which diagnostic feature (tenebrism, sfumato,
  impasto, broken color, linear vs. painterly…).
- **ChronoOrder:** two works → which came first (period ordering intuition).

Stimuli are **real artworks** — must be retrieved, never generated. Sources with open APIs and
public-domain images: The Met Open Access API, Art Institute of Chicago API, Rijksmuseum API,
Wikimedia Commons. **Hard rule: every image's metadata (artist, date, movement) comes from the
museum API record, cross-checked against a second source — the LLM never asserts attribution
from memory.** Optional quality gate: local CLIP (via `transformers`) embeds images and flags
outliers whose embedding sits far from their labeled movement's centroid.

---

## 3. System architecture

```
            deterministic Python driver (run_pipeline.py — NOT an agent)
                 │  phases, retries, file I/O, schema checks
                 │
   ┌─────────────┼──────────────────┬──────────────────┐
   │             │                  │                  │
retrieval_agent  chem_plm_agent   math_plm_agent    arh_plm_agent
(ToolCallingAgent) (CodeAgent)     (CodeAgent)       (CodeAgent)
   │
   └── shared: each CodeAgent gets retrieval_agent as a managed_agent
                 │
                 ▼
        item banks (JSON + PNG stimuli)
                 ▼
        plm_core/  (handwritten shared engine: ARTS scheduler, session runner,
                    mastery/retirement logic, RT logging, Gradio student UI)
```

Design rationale (straight from smolagents' *Building good agents* guide):
- **"Reduce LLM calls; prefer deterministic logic over agentic decisions"** → the orchestrator is
  a plain Python script, not a manager agent. Phases, looping over categories, schema validation,
  and file management are deterministic. (A manager CodeAgent variant is possible but adds cost
  and failure modes for zero benefit here — the workflow is a known pipeline, not open-ended.)
- **The ARTS engine is written once by us** in `plm_core/` — it is identical across courses and
  is exactly the kind of correctness-critical code that shouldn't be regenerated by three agents.
- **Per-course CodeAgents** (not functional split) because toolsets and domain prompts differ
  radically: sympy math vs. structure drawing vs. image curation. This is smolagents'
  "specialize units on sub-tasks, keep memories separate" principle.
- **Retrieval agent is a ToolCallingAgent** because the docs recommend it for atomic
  fetch/dispatch tasks (reliability > expressivity); it's shared as a `managed_agent` by all
  three CodeAgents, which call it like a function with a detailed task string.

## 4. Agent-by-agent specification

### 4.1 `retrieval_agent` (ToolCallingAgent)
- **Tools:**
  - `WebSearchTool()` (built-in)
  - `visit_webpage(url)` (markdownify pattern from the docs)
  - `RetrieverTool` — BM25 over the ingested project corpus (RAG-docs pattern:
    RecursiveCharacterTextSplitter, chunk_size 500/overlap 50, BM25Retriever k=10)
  - `fetch_museum_object(api, object_id)` and `search_museum(api, query, filters)` — thin wrappers
    over Met/AIC/Rijksmuseum REST APIs; download image → `assets/arh/`, return metadata JSON
  - `download_image(url, dest)` with pillow validation (opens, min resolution check)
- **Config:** `max_steps=10`, `name="retrieval_agent"`,
  `description="Retrieves course materials, verified artwork images+metadata from museum APIs, and reference facts. Give it a precise task: what to fetch, from where, and the exact output format."`
- Every tool logs verbosely (print statements) and raises explicit, instructive errors — the
  docs' information-flow rule.

### 4.2 `math_plm_agent` (CodeAgent)
- **Job:** for each MATH 223 category, write parametric item-generator functions, execute them to
  emit N validated items + rendered PNG stimuli, and self-check every answer key symbolically.
- **Tools:** `save_item(item_json)` (validates against schema, writes to bank),
  `render_figure(spec)` optional helper; plus `retrieval_agent` for syllabus/topic grounding.
- **Config:** `additional_authorized_imports=["numpy.*","sympy.*","matplotlib.*","json","random","itertools","math"]`,
  `max_steps=20`, `planning_interval=4`,
  `final_answer_checks=[bank_completeness_check]` (all categories ≥ target count, schema-valid,
  transfer set held out).
- **Key prompt content:** the PLM design rules of §1 (classification-only trials, distractor
  design = near-miss structures, exemplar variability requirements), category specs of §2.

### 4.3 `chem_plm_agent` (CodeAgent)
- Same shape as math agent. Imports add `rdkit.*` (if adopted) or `PIL.*` drawing fallback.
- **Ground-truth policy:** answer keys computed by RDKit/rule functions the agent writes
  (e.g., CIP assignment via RDKit, mechanism classification via an explicit decision table
  over substrate class + nucleophile/base strength + solvent), never bare LLM assertion.
  The decision table itself is reviewed content — sourced via `retrieval_agent` from standard
  orgo references, then frozen as code.

### 4.4 `arh_plm_agent` (CodeAgent)
- **Job:** build the ARH taxonomy (movements, target artists, diagnostic features) from the
  course scope via `retrieval_agent`; curate 40–80 images per movement through the museum-API
  tools; produce classification items whose metadata is museum-verified; flag CLIP outliers
  for exclusion; hold out transfer images.
- **Config:** `additional_authorized_imports=["PIL.*","json","random","numpy.*","datasets"]` +
  optionally `transformers.*` for CLIP; `max_steps=25` (curation is iterative);
  `planning_interval=5`; final-answer check = per-movement counts, dual-source metadata
  verification flags all true.

## 5. Pipeline (deterministic driver phases)

1. **Ingest** — retrieval_agent gathers: UA course descriptions/syllabi topics, orgo mechanism
   references, ARH 202 scope (movements/artists typically covered), math topic list. Chunk into
   BM25 knowledge base.
2. **Taxonomy** — each CodeAgent proposes its category taxonomy + per-category spec
   (structured JSON, human-reviewable checkpoint — *you approve before item generation*).
3. **Generate** — CodeAgents produce item banks: ≥60 items/category training + ≥15 transfer
   (math/chem come from parametric templates so counts are unlimited; ARH bounded by curated
   images).
4. **Validate** — automatic: schema, symbolic/rule verification, image checks, CLIP outliers;
   cross-agent spot audit: each agent re-verifies a random 10% sample of another course's items
   *where domain-agnostic checks apply* (schema, stimulus renders, single-correct-answer).
5. **Assemble** — banks compiled into `plm_core` format; ARTS config per category (RT thresholds
   per trial type, retirement N, session length).
6. **Deliver** — `plm_core` Gradio app: student picks course → interleaved adaptive session →
   dashboard of accuracy/RT/retired categories; all trials logged to JSONL for efficacy analysis.

## 6. Item schema (the contract everything validates against)

```json
{
  "id": "math223.fieldsense.000123",
  "course": "MATH223",
  "category": "FieldSense",
  "subcategory": "divergence_sign",
  "stimulus": {"type": "image", "path": "assets/math/fs_000123.png"},
  "prompt": "At the marked point, the divergence of this field is:",
  "choices": ["positive", "negative", "zero"],
  "correct": 0,
  "feedback": "Arrows spread apart near the point — net outflow, div > 0.",
  "ground_truth_method": "sympy.divergence, evaluated at point",
  "difficulty": 2,
  "transfer": false,
  "provenance": {"generator": "template_fieldsense_v1", "seed": 8841}
}
```
ARH items add `provenance.museum_api`, `provenance.object_id`, `provenance.verified_by` (2 sources).

## 7. ARTS engine spec (`plm_core/`, handwritten)

- Per-category priority score: `P = f(trials_since_last_seen, error_rate, mean_RT_vs_threshold)`
  — errors and slow-but-correct responses raise priority; recent presentation suppresses it
  (enforces spacing); random jitter enforces interleaving.
- **Retirement:** category retires after 4 consecutive correct with RT < category threshold
  (defaults: 8 s visual classification, 15 s multi-representation mapping); retired categories
  reappear sparsely as maintenance checks.
- Session: 15–25 min or all-retired; immediate feedback (+ structure highlight on error);
  every trial logs `{item_id, correct, rt_ms, timestamp}`.
- Mastery report per student per category; transfer assessment uses only held-out items.

## 8. Model & runtime choices

- **Model:** `LiteLLMModel(model_id="anthropic/claude-sonnet-5")` for all four agents (CodeAgents
  need strong code generation; the docs' first debugging rule is "use a stronger LLM").
  Swappable to `InferenceClientModel` (HF) if preferred.
- **Executor:** local Python executor is acceptable (all tools are ours; imports allowlisted);
  Docker/E2B executor is the upgrade path if item-generation code grows risky.
- **Stack check:** smolagents ✅ transformers ✅ (CLIP QA) datasets ✅ numpy ✅ matplotlib ✅
  pillow ✅ sympy ✅. `edgartools` is no longer needed (it was for the dropped FIN 421).
  **Gaps to add:** `rank_bm25`, `langchain-community` (RetrieverTool pattern), `markdownify`,
  `gradio`; strongly recommended: `rdkit`.

## 9. Build plan

| Milestone | Deliverable |
|---|---|
| M1 | `plm_core/` engine + Gradio runner working on a 20-item dummy bank |
| M2 | retrieval_agent + tools; ingest corpus; MATH agent end-to-end (easiest ground truth) |
| M3 | CHEM agent (RDKit decision) → validated chem bank |
| M4 | ARH agent + museum pipelines → curated, dual-verified image bank |
| M5 | Cross-validation phase, transfer sets, student pilot + RT/accuracy analytics |

## 10. Risks & mitigations

- **LLM-hallucinated answer keys** → every key must come from code oracles (sympy/RDKit/rule
  tables) or museum metadata; `final_answer_checks` reject unverified items.
- **ARH image licensing** → restrict to open-access/public-domain API endpoints; store object IDs.
- **Chem drawing without RDKit** → adopt RDKit or descope RepMap/Stereo to template-drawn subset.
- **Agent cost/looping** → deterministic driver, `max_steps` caps, planning_interval, parametric
  templates so agents write generators once instead of emitting items one-by-one.
- **ARH 292 ambiguity** → confirmed absent from UA catalog; ARH 202 assumed. Confirm with user.

## 11. Key sources

- Kellman, Massey & Son (2010), *Perceptual Learning Modules in Mathematics*, Topics in
  Cognitive Science — https://pmc.ncbi.nlm.nih.gov/articles/PMC6124488/
- Kellman & Massey (2013), *Perceptual Learning, Cognition, and Expertise* —
  https://kellmanlab.psych.ucla.edu/files/kellman_2013.pdf
- UCLA Kellman Lab, Perceptual & Adaptive Learning in STEM —
  https://kellmanlab.psych.ucla.edu/research-perceptual-and-adaptive-learning-in-stem.php
- Krasne et al. (2020), ECG PALM — mastery/RT criteria in medical perceptual learning
- smolagents docs: guided tour, *Building good agents*, multi-agent orchestration, Agentic RAG —
  https://huggingface.co/docs/smolagents/
- UA catalog: CHEM 241A (0105261), MATH 223 (0208981), ARH 202 (0080521)
