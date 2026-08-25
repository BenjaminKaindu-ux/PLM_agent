"""One-time (per-textbook) ingestion: PDF -> chunks -> persistent Chroma collection.

Run directly: `python -m src.plm_core.ingest`. Idempotent — rebuilds a collection
only if it doesn't already exist, since embedding ~1.5B-param stella vectors for
two full textbooks is the expensive step.
"""

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import CHROMA_COLLECTIONS, CHROMA_PATH
from src.plm_core.embeddings import BgeM3EmbeddingFunction

SOURCES = {
    "MATH223": {
        "path": "/Users/macbook/Downloads/calculus-volume-3_-_WEB.pdf",
        "source_name": "OpenStax Calculus Volume 3",
    },
    "CHEM241A": {
        "path": "/Users/macbook/Downloads/organic-chemistry_-_WEB.pdf",
        "source_name": "OpenStax Organic Chemistry (McMurry 10e)",
    },
}


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]], course: str, source_name: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs, metadatas, ids = [], [], []
    for page_num, text in pages:
        for j, chunk in enumerate(splitter.split_text(text)):
            docs.append(chunk)
            metadatas.append({"course": course, "source": source_name, "page": page_num})
            ids.append(f"{course}.p{page_num}.c{j}")
    return docs, metadatas, ids


def build_collection(client, course: str, cfg: dict):
    name = CHROMA_COLLECTIONS[course]
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        coll = client.get_collection(name)
        if coll.count() > 0:
            print(f"[{course}] collection '{name}' already has {coll.count()} chunks — skipping")
            return
    coll = client.get_or_create_collection(name, embedding_function=BgeM3EmbeddingFunction())

    print(f"[{course}] extracting text from {cfg['path']}", flush=True)
    pages = extract_pages(cfg["path"])
    print(f"[{course}] {len(pages)} pages with text", flush=True)

    docs, metadatas, ids = chunk_pages(pages, course, cfg["source_name"])
    print(f"[{course}] {len(docs)} chunks — embedding via HF Inference API + writing to Chroma", flush=True)

    batch = 16  # small batches: embeddings go over the HF Inference API, not local
    for i in range(0, len(docs), batch):
        coll.add(documents=docs[i : i + batch], metadatas=metadatas[i : i + batch], ids=ids[i : i + batch])
        if (i // batch) % 20 == 0:
            print(f"[{course}] {min(i + batch, len(docs))}/{len(docs)} chunks embedded", flush=True)

    print(f"[{course}] done — {coll.count()} chunks in '{name}'")


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    for course, cfg in SOURCES.items():
        build_collection(client, course, cfg)

    arh_name = CHROMA_COLLECTIONS["ARH"]
    if arh_name not in [c.name for c in client.list_collections()]:
        client.get_or_create_collection(arh_name, embedding_function=BgeM3EmbeddingFunction())
        print(f"[ARH] created empty collection '{arh_name}' — no local corpus yet "
              "(ASCCC OERI + Smarthistory are web sources, owned by the Research Agent, not built yet)")


if __name__ == "__main__":
    main()
