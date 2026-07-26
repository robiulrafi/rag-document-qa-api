"""Does HYBRID (vector + BM25) find the $25,000 fee chunk that vector-only missed?

This rebuilds BM25 in-memory from the chunks already in Chroma (no re-parsing
the PDF), unions it with vector results, and shows whether the fee chunk surfaces.
"""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# --- pull all chunks back out of Chroma to build BM25 (text is stored alongside vectors) ---
raw = store.get()
docs = [Document(page_content=t, metadata=m or {})
        for t, m in zip(raw["documents"], raw["metadatas"])]
print(f"rebuilt {len(docs)} docs from Chroma for BM25\n")

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 3
vector = store.as_retriever(search_kwargs={"k": 3})

def hybrid(query):
    v = vector.invoke(query)
    b = bm25.invoke(query)
    seen, merged = set(), []
    for d in v + b:                       # union
        if d.page_content not in seen:    # dedupe
            seen.add(d.page_content)
            merged.append(d)
    return merged

q = "How much does the service cost?"
print(f"QUERY: {q!r}\n")

print("--- VECTOR ONLY (what you have now) ---")
for i, d in enumerate(vector.invoke(q), 1):
    hit = "  <== $25,000!" if "25,000" in d.page_content else ""
    print(f"  [{i}] {d.page_content[:75].strip()}{hit}")

print("\n--- BM25 ONLY ---")
for i, d in enumerate(bm25.invoke(q), 1):
    hit = "  <== $25,000!" if "25,000" in d.page_content else ""
    print(f"  [{i}] {d.page_content[:75].strip()}{hit}")

print("\n--- HYBRID (vector + BM25) ---")
for i, d in enumerate(hybrid(q), 1):
    hit = "  <== $25,000!" if "25,000" in d.page_content else ""
    print(f"  [{i}] {d.page_content[:75].strip()}{hit}")