"""Steps 6-11 of RAG: retrieve relevant chunks, build the prompt, ask Gemini."""
import os
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from rag.vector_store import get_vectorstore, count_chunks

DEFAULT_MODEL = "gemini-2.5-flash"

NOT_FOUND_MESSAGE = (
    "I couldn't find enough information about this in the knowledge base, so I'd rather "
    "not guess.\n\n**You can:**\n- Rephrase your question with Python keywords\n"
    "- Upload a PDF, TXT or DOCX that covers this topic (sidebar -> Upload documents)\n"
    "- Lower the *Minimum relevance* slider in Settings"
)

GREETING_MESSAGE = (
    "Hi! I'm your Python tutor. Ask me about variables, loops, functions, OOP, files, "
    "NumPy, Pandas and more. I answer from the documents in the knowledge base."
)

SYSTEM_PROMPT = """You are "Python AI Tutor", a patient teacher for beginners.

RULES
1. Answer using ONLY the CONTEXT below. Context items are numbered [1], [2], ...
2. Cite the items you used with their numbers, e.g. "Lists are mutable [1]".
3. If the CONTEXT does not contain enough information, say clearly that the knowledge base
   does not contain sufficient information. Never invent facts or citations.
4. If the student pastes their own code or an error message, you may analyse that code
   directly (explain it line by line, find the bug), but say when a point goes beyond the context.
5. Explain in simple language. Put every code example in a fenced block that opens with
   three backticks and the word python, and closes with three backticks on its own line.
   Never use single backticks around multi-line code.
6. When asked, you can summarise a topic, create practice questions, or write small exercises
   based on the CONTEXT.
7. Keep answers focused and well structured. Student level: {level}.

CONTEXT
{context}
"""


def _to_text(content) -> str:
    """Gemini may return a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return str(content)


def retrieve(question: str, history: List[Dict], k: int, min_score: float) -> List[Dict]:
    """Semantic search. Returns chunks whose similarity >= min_score."""
    query = question
    # Very short follow-ups ("and tuples?") are easier to find with the previous question.
    if len(question.split()) < 4:
        previous = [m["content"] for m in history if m["role"] == "user"]
        if previous:
            query = f"{previous[-1]} {question}"

    vs = get_vectorstore()
    if count_chunks(vs) == 0:
        return []
    results = vs.similarity_search_with_score(query, k=k)  # (Document, cosine distance)
    chunks = []
    for doc, distance in results:
        similarity = 1.0 - float(distance)
        if similarity >= min_score:
            chunks.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "text": doc.page_content,
                "score": max(0.0, similarity),
            })
    return chunks


def _build_context(chunks: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        page = f", page {c['page']}" if c.get("page") else ""
        blocks.append(f"[{i}] (source: {c['source']}{page})\n{c['text']}")
    return "\n\n".join(blocks)


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("MISSING_API_KEY")
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        google_api_key=api_key,
        temperature=0.3,
    )


def ask(question: str, history: List[Dict], k: int = 4,
        min_score: float = 0.25, level: str = "Beginner") -> Dict:
    """Full RAG round trip. Returns {"answer": str, "sources": list}."""
    chunks = retrieve(question, history, k, min_score)
    if not chunks:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    llm = _get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(
        level=level, context=_build_context(chunks)))]
    for m in history[-6:]:  # short memory for follow-up questions
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))
    messages.append(HumanMessage(content=question))

    answer = _to_text(llm.invoke(messages).content).strip()
    sources = [{**c, "excerpt": c["text"][:400].strip() + ("..." if len(c["text"]) > 400 else "")}
               for c in chunks]
    return {"answer": answer or NOT_FOUND_MESSAGE, "sources": sources}
