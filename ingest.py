from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")        # ← your file name
pages = loader.load()

print(f"Loaded {len(pages)} pages")
print(f"\n--- First page preview ---")
print(pages[0].page_content[:400])
print(f"\n--- Metadata ---")
print(pages[0].metadata)                  # source + page number— matters for citations


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,        # target characters per chunk
    chunk_overlap=50,      # overlap so context isn't cut mid-thought
)
chunks = splitter.split_documents(pages)

print(f"\nSplit {len(pages)} pages into {len(chunks)} chunks")
print(f"\n--- First chunk ---")
print(chunks[0].page_content)
print(f"\n--- Chunk metadata ---")
print(chunks[0].metadata)


from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",       # same store as yesterday
)

print(f"\nEmbedded and stored {len(chunks)} chunks in ChromaDB")

query = "How much does the service cost?"     # ← ask something your PDF actually covers
results = store.similarity_search(query, k=3)

print(f"\nQuery: {query}\n")
for i, chunk in enumerate(results, 1):
    src = chunk.metadata.get("source", "?")
    page = chunk.metadata.get("page", "?")
    print(f"{i}. [page {page}] {chunk.page_content[:200]}...\n")




from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- keyword retriever (needs your chunks in memory) ---
bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 3

# --- semantic retriever (your existing Chroma store) ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
vector_retriever = store.as_retriever(search_kwargs={"k": 3})

# --- hybrid: query both, merge, dedupe ---
def hybrid_search(query):
    hits = bm25.invoke(query) + vector_retriever.invoke(query)
    seen, merged = set(), []
    for doc in hits:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged.append(doc)
    return merged

# --- test: an EXACT term, where BM25 earns its keep ---
print("=== Query: 'Northwind Trading' (exact term) ===")
for i, doc in enumerate(hybrid_search("Northwind Trading"), 1):
    print(f"{i}. {doc.page_content[:120]}...\n")