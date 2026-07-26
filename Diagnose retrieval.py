"""Diagnostic: is the actual fee amount being retrieved at all?
Pull MORE chunks and see where (if anywhere) the price appears.
"""
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# retrieve TOP 10 instead of 3
retriever = store.as_retriever(search_kwargs={"k": 10})
docs = retriever.invoke("How much does the service cost?")

print(f"Top 10 chunks for 'How much does the service cost?':\n")
for i, d in enumerate(docs, 1):
    text = d.page_content.replace("\n", " ")
    # flag chunks that look like they contain a price
    has_dollar = "$" in text or "fee" in text.lower() or "per month" in text.lower()
    flag = "  <-- mentions fee/$" if has_dollar else ""
    print(f"[{i}] {text[:95]}{flag}")