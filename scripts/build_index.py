"""Build a precomputed vector index for the Vercel deployment.

Run locally once (or after editing knowledge-base files):
    python scripts/build_index.py

Writes api/data/index.json containing chunk texts, source metadata, and
optional MiniLM embeddings produced by fastembed. The Vercel runtime uses the
chunk text for lightweight lexical retrieval and does not install fastembed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_index")

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"))
if not KNOWLEDGE_BASE_DIR.is_absolute():
    KNOWLEDGE_BASE_DIR = BASE_DIR / KNOWLEDGE_BASE_DIR
OUTPUT_PATH = BASE_DIR / "api" / "data" / "index.json"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def load_documents() -> list[dict]:
    """Load/tidy documents from the knowledge base (mirrors support_rag._load_documents)."""
    from langchain_community.document_loaders import (
        Docx2txtLoader,
        PyPDFLoader,
        TextLoader,
    )

    paths = sorted(
        path
        for path in KNOWLEDGE_BASE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".txt", ".pdf", ".docx"}
        and not path.name.startswith("~$")
    )
    documents = []
    for path in paths:
        source = path.relative_to(KNOWLEDGE_BASE_DIR).as_posix()
        extension = path.suffix.lower()
        try:
            if extension == ".txt":
                loader = TextLoader(str(path), encoding="utf-8-sig", autodetect_encoding=True)
            elif extension == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = Docx2txtLoader(str(path))
            for document in loader.load():
                document.page_content = re.sub(r"[ \t]+", " ", document.page_content).strip()
                if not document.page_content:
                    continue
                documents.append(
                    {
                        "text": document.page_content,
                        "source": source,
                        "file_type": extension,
                        "page": document.metadata.get("page"),
                    }
                )
            logger.info("loaded source=%s pages_or_docs=%d", source, len(documents))
        except Exception as exc:
            logger.warning("skipped source=%s error_type=%s", source, type(exc).__name__)
    return documents


def main() -> None:
    from fastembed import TextEmbedding
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents = load_documents()
    if not documents:
        raise SystemExit("No readable knowledge-base files found.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc in documents:
        pieces = splitter.create_documents([doc["text"]])
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "text": piece.page_content,
                    "source": doc["source"],
                    "file_type": doc["file_type"],
                    "page": doc["page"],
                    "chunk_index": i,
                }
            )
    logger.info("documents=%d chunks=%d", len(documents), len(chunks))

    logger.info("embedding_chunks=%d model=%s", len(chunks), EMBEDDING_MODEL)
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    vectors = [emb.tolist() for emb in model.embed(texts, batch_size=64)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": EMBEDDING_MODEL, "chunks": chunks, "vectors": vectors}
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    logger.info("written path=%s chunks=%d size_mb=%.2f", OUTPUT_PATH, len(chunks), size_mb)


if __name__ == "__main__":
    main()
