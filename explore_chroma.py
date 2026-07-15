from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

docs = [
    "Python is a programming language.",
    "The stock market closed higher today.",
    "Machine learning models need training data.",
    "Investors watched interest rates closely.",
]

# Build a local, persistent vector store
store = Chroma.from_texts(
    texts=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# Query with something new — no keyword overlap on purpose
query = "How do neural networks learn?"
results = store.similarity_search(query, k=2)

print(f"Query: {query}\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")