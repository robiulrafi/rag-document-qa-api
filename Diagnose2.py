"""Is the fee chunk findable at all? Try queries closer to the document's wording.
This isolates: is it a QUERY-PHRASING problem (rewrite fixes it) or a
RETRIEVAL-METHOD problem (needs hybrid/BM25)?
"""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})

queries = [
    "How much does the service cost?",          # original — failed
    "What is the monthly fee?",                 # reworded (what your rewrite node might produce)
    "monthly fee payment amount dollars",       # keyword-style
    "fee",                                      # bare keyword
]

for q in queries:
    docs = retriever.invoke(q)
    print(f"\nQUERY: {q!r}")
    for i, d in enumerate(docs, 1):
        t = d.page_content.replace("\n"," ")
        hit = "  <-- $/fee" if ("$" in t or "fee" in t.lower()) else ""
        print(f"  [{i}] {t[:80]}{hit}")

# also: does the fee amount even EXIST in the store?
print("\n\n=== does the raw text contain a dollar fee anywhere? ===")
all_docs = store.get()
texts = all_docs.get("documents", [])
import re
for t in texts:
    m = re.search(r'\$[\d,]+|\bfee\b.{0,40}', t, re.I)
    if "$" in t:
        print(f"  FOUND $ in a chunk: ...{t[max(0,t.find('$')-40):t.find('$')+40]}...")