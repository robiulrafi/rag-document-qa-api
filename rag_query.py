from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant answering questions about a document.\n\n"
     "Answer ONLY using the context below. Do not use outside knowledge.\n"
     "If the answer is not in the context, say exactly: "
     "\"I don't know — that isn't covered in the document.\"\n"
     "Cite the page number for each fact you state.\n\n"
     "CONTEXT:\n{context}"),
    ("human", "{question}"),
])


def format_context(docs):
    """Turn retrieved chunks into a single context string with page markers."""
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="nomic-embed-text")
store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="llama3.2", temperature=0)

def answer_question(question: str):
    docs = retriever.invoke(question)              # 1. RETRIEVE
    context = format_context(docs)                  # 2. AUGMENT
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({                         # 3. GENERATE
        "context": context,
        "question": question,
    })
    return answer, docs


if __name__ == "__main__":
    q = "What is the employee vacation policy?"
    answer, sources = answer_question(q)
    print(f"Q: {q}\n")
    print(f"A: {answer}\n")
    print("Sources:")
    for d in sources:
        print(f"  - page {d.metadata.get('page')}: {d.page_content[:80]}...")