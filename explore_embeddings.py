from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",   # similar meaning to #1
    "Quarterly revenue grew 12%.",   # unrelated
]

vectors = embeddings.embed_documents(sentences)

print(f"Number of vectors: {len(vectors)}")
print(f"Vector length (dimensions): {len(vectors[0])}")
print(f"First 5 numbers of vector 0: {vectors[0][:5]}")


import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\ncat vs feline  (similar):   {cosine(vectors[0], vectors[1]):.3f}")
print(f"cat vs revenue (unrelated): {cosine(vectors[0], vectors[2]):.3f}")