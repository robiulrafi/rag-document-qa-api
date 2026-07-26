"""Ingestion: load the PDF, chunk it, embed, and write to ChromaDB.

This module WRITES the vector store. Querying (reading) lives in rag_query.py
and runs separately — mixing the two causes duplicate chunks, because
Chroma.from_documents() appends on every call.

Chunking is structure-aware: we split on section headers (e.g. "3. Fees and
Payment") so each numbered clause becomes its own chunk. Character-based
splitting cut blindly every 250 chars and glued the tail of one section to the
head of the next — which buried the "$25,000 monthly fee" inside a chunk about
termination, so no retriever could rank it for a cost question.
"""

import re

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "sample.pdf"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

# Split BEFORE each numbered section header: newline + "3. Fees and Payment".
# Lookahead (?=...) keeps the header attached to its section.
SECTION_RE = r"(?=\n\d+\.\s+[A-Z])"

# Any single section longer than this is further split, so no chunk is too
# large to embed cleanly. Preserves the section's metadata.
MAX_SECTION_CHARS = 1000
_fallback = RecursiveCharacterTextSplitter(
    chunk_size=MAX_SECTION_CHARS,
    chunk_overlap=100,
)


def structure_aware_split(pages) -> list[Document]:
    """Split each page on section boundaries; keep each numbered clause whole."""
    chunks: list[Document] = []
    for page in pages:
        sections = re.split(SECTION_RE, page.page_content)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= MAX_SECTION_CHARS:
                chunks.append(
                    Document(page_content=section, metadata=page.metadata)
                )
            else:
                for sub in _fallback.split_text(section):
                    chunks.append(
                        Document(page_content=sub, metadata=page.metadata)
                    )
    return chunks


def main() -> None:
    # 1. LOAD
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {PDF_PATH}")

    # 2. CHUNK (structure-aware)
    chunks = structure_aware_split(pages)
    print(f"Split into {len(chunks)} structure-aware chunks")
    print("\n--- first 3 chunks (note each starts at a section boundary) ---")
    for i, c in enumerate(chunks[:3], 1):
        preview = c.page_content.replace("\n", " ")[:80]
        print(f"  [{i}] (page {c.metadata.get('page')}) {preview}...")

    # 3. EMBED + WRITE
    #    NOTE: from_documents() APPENDS. Delete ./chroma_db before re-ingesting,
    #    or you accumulate duplicate chunks.
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"\nEmbedded and stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}")

    # 4. SANITY CHECK — does the fee query now surface the fee chunk?
    query = "How much does the service cost?"
    results = store.similarity_search(query, k=3)
    print(f"\nSanity check — query: {query!r}")
    for i, chunk in enumerate(results, 1):
        page = chunk.metadata.get("page", "?")
        has_fee = "  <== fee chunk" if "25,000" in chunk.page_content else ""
        preview = chunk.page_content.replace("\n", " ")[:90]
        print(f"  {i}. [page {page}] {preview}...{has_fee}")


if __name__ == "__main__":
    main()