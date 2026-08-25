"""Central configuration for the PLM Factory agents.

Model assignments (updated 2026-07-31):
- Retrieval agent embeddings: BAAI/bge-m3, called via HF Inference API
  (huggingface_hub.InferenceClient.feature_extraction, provider="hf-inference") — not
  loaded locally. Switched from stella_en_1.5B_v5 (local sentence-transformers) after
  a transformers-version incompatibility broke stella's custom remote code, and because
  the Space needs an API-backed embedding model rather than a local 1.5B+ model.
  Qwen3-Embedding-4B was considered but has no live HF Inference Provider mapping.
- Research agent + PLM agent: Qwen2.5-Coder-32B-Instruct via HF Inference API.

Auth: HF token comes from the local `hf auth login` credential store or the HF_TOKEN
environment variable. NEVER hard-code tokens in this repo.
"""

# Embedding model for the Retrieval Agent's Chroma collections — called via HF
# Inference API (see src/plm_core/embeddings.py), no local model download/load.
RETRIEVAL_EMBEDDING_MODEL = "BAAI/bge-m3"

# LLM powering the Research Agent (web -> cached concept briefs)
RESEARCH_AGENT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# LLM powering the PLM Agent (CodeAgent; course passed as a parameter)
PLM_AGENT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Where the Retrieval Agent's persistent Chroma DB lives (built once by ingest.py,
# just read by retrieval.py — kept here, not in ingest.py, so querying doesn't pull in
# ingestion-only deps like pypdf/langchain_text_splitters).
CHROMA_PATH = "data/chroma"

# Chroma collections managed by the Retrieval Agent
CHROMA_COLLECTIONS = {
    "MATH223": "math223",
    "CHEM241A": "chem241",
    "ARH": "arh",
}

# ARTS defaults (Kellman-style adaptive response-time sequencing)
RETIRE_STREAK = 4                 # consecutive correct-and-fast answers to retire a category
RT_THRESHOLD_SIMPLE_S = 8.0       # simple visual classification
RT_THRESHOLD_MAPPING_S = 15.0     # multi-representation mapping
SESSION_MINUTES = (10, 15)
TRIALS_PER_SESSION_TARGET = 60
