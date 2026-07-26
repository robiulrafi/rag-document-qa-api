"""Confirm the chunking hypothesis: is $25,000 trapped in a chunk about breach?
Print the FULL text of any chunk containing 25,000.
"""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
raw = store.get()

for i, t in enumerate(raw["documents"]):
    if "25,000" in t:
        print(f"=== CHUNK {i} containing $25,000 (full text) ===")
        print(repr(t))
        print(f"\nlength: {len(t)} chars")
        print(f"\nreadable:\n{t}")
        print("\n" + "="*60)