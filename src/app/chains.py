"""LangChain chains built on a local Ollama model.

Everything here runs locally — no API keys, no per-token cost.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import settings


# --------------------------------------------------------------------------
# Structured output schema
# --------------------------------------------------------------------------
class Answer(BaseModel):
    """Schema the LLM must populate when structured output is requested."""

    summary: str = Field(description="A detailed 2-sentence explanation of the concept")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0
    )


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
def get_llm() -> ChatOllama:
    """Factory for the local Ollama chat model.

    Using a factory (rather than a module-level singleton) keeps the model
    swappable in tests and makes the dependency explicit.
    """
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.TEMPERATURE,
    )


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, concise assistant."),
        ("human", "{question}"),
    ]
)

STRUCTURED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Return only valid JSON matching the "
            "requested schema. The 'confidence' field must be a float between "
            "0.0 and 1.0.",
        ),
        ("human", "{question}"),
    ]
)


# --------------------------------------------------------------------------
# Chains (LCEL:  prompt | llm | parser)
# --------------------------------------------------------------------------
def build_qa_chain():
    """Plain text chain. Supports .invoke, .ainvoke, .stream, .astream."""
    return QA_PROMPT | get_llm() | StrOutputParser()


def build_structured_chain():
    """Chain that returns a validated `Answer` object.

    `with_structured_output` handles schema enforcement natively, so no
    OutputFixingParser is needed.
    """
    structured_llm = get_llm().with_structured_output(Answer)
    return STRUCTURED_PROMPT | structured_llm
