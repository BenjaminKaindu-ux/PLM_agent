---
title: PLM Factory — Perceptual Learning Drills
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: Adaptive perceptual learning drills (MATH 223 demo)
---

# 🧠 PLM Factory

A multi-agent system (built on [smolagents](https://huggingface.co/docs/smolagents)) that
generates **Perceptual Learning Modules** — rapid, timed, high-repetition classification
drills that build fast visual/pattern intuition — for three University of Arizona courses:

| Course | Perceptual skill trained | Item engine (ground-truth oracle) |
|---|---|---|
| MATH 223 — Vector Calculus | Read a vector field plot, instantly judge div/curl sign | `sympy` (symbolic — exact keys) |
| CHEM 241A — Organic Chemistry | Recognize functional groups, stereo, mechanisms at a glance | `rdkit` (SMARTS / CIP) |
| ARH 201/202 — Western Art surveys | Classify style/period/artist from a fragment | WikiArt labels + museum APIs |

**This Space hosts two tabs:**
1. **MATH223 Student Demo** — the *FieldSense* drill: timed 3-way classification of
   divergence/curl sign with ARTS adaptive sequencing (accuracy **and** response time;
   categories retire at 4 consecutive fast-and-correct answers), per Kellman, Massey & Son
   (2010), *Topics in Cognitive Science*.
2. **Agent Debug (dev)** — pick a course + concept and watch a real `smolagents` `CodeAgent`
   (Qwen2.5-Coder-32B-Instruct) generate a live item end to end, with its code trace shown so
   you can verify it calls the course's sanctioned generator tool rather than asserting an
   answer itself. Deliberately built as fixed dropdowns rather than smolagents' built-in
   free-text `GradioUI` chat, since an open-ended chat into a code-executing agent is a real
   prompt-injection surface on a public Space.

## Architecture (v2)

Three agents: **Retrieval Agent** (Chroma ×3 collections, `BAAI/bge-m3` embeddings via the HF
Inference API) + **Research Agent** (DuckDuckGo web search → cached, per-topic concept briefs;
`Qwen2.5-Coder-32B-Instruct`) + a single **PLM Agent** (CodeAgent, course as a parameter;
`Qwen2.5-Coder-32B-Instruct`) whose chain-of-thought only picks *category / difficulty /
instance parameters* — all item content and answer keys come from deterministic code oracles
(sympy / RDKit / WikiArt's own curated labels), never LLM assertion, enforced live via a
`step_callbacks` guard. Full design: `ARCHITECTURE.md`.

## Repo layout

```
app.py                        # this Space: student demo tab + agent debug tab
src/config.py                 # model assignments, Chroma paths, ARTS constants
src/plm_core/arts.py          # ARTS adaptive sequencing tracker (deterministic)
src/plm_core/plm_agent.py     # PLM Agent (CodeAgent, course as a parameter)
src/plm_core/retrieval.py     # Retrieval Agent + retrieve_course_context tool
src/plm_core/research.py      # Research Agent (web -> cached per-topic briefs)
src/plm_core/embeddings.py    # bge-m3 embeddings via HF Inference API
src/plm_core/ingest.py        # one-time textbook PDF -> Chroma ingestion (local-only)
src/plm_core/wikiart_cache.py # one-time WikiArt exemplar caching (local-only)
src/generators/math223.py     # sympy-oracle FieldSense item generator
src/generators/chem241a.py    # RDKit-oracle GroupSense/StereoSense item generator
src/generators/arh.py         # WikiArt-label-oracle StyleSense item generator
data/chroma/                  # pre-built Chroma DB (math223, chem241, arh collections)
data/wikiart_cache/           # cached WikiArt exemplar images + manifest
ARCHITECTURE.md               # confirmed v2 system design
PROPOSAL.md                   # superseded v1 (kept for item schema / ARTS detail)
requirements.txt              # what's actually installed on this Space
requirements-full.txt         # full project stack incl. local-only ingestion/observability
```
