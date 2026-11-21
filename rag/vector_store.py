"""Step 5 of RAG: store and search vectors in ChromaDB."""
from typing import Callable, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.document_loader import load_all_documents
from rag.embeddings import get_embeddings
from rag.text_splitter import split_documents
from utils.helpers import VECTOR_DIR, ensure_dirs

COLLECTION_NAME = "python_tutor"


def get_vectorstore() -> Chroma:
    ensure_dirs()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(VECTOR_DIR),
        collection_metadata={"hnsw:space": "cosine"},  # distance = 1 - similarity
    )


def count_chunks(vs: Optional[Chroma] = None) -> int:
    vs = vs or get_vectorstore()
    return vs._collection.count()


def add_chunks(vs: Chroma, chunks: List[Document],
               progress: Optional[Callable[[float], None]] = None, batch: int = 64) -> None:
    for start in range(0, len(chunks), batch):
        vs.add_documents(chunks[start:start + batch])
        if progress:
            progress(min(1.0, (start + batch) / max(1, len(chunks))))


def delete_source(vs: Chroma, source_name: str) -> None:
    """Remove every chunk that came from one file."""
    vs._collection.delete(where={"source": source_name})


def rebuild_vectorstore(progress: Optional[Callable[[float], None]] = None):
    """Wipe the collection and index every document again."""
    try:
        get_vectorstore().delete_collection()
    except Exception:
        pass
    vs = get_vectorstore()
    documents = load_all_documents()
    chunks = split_documents(documents)
    if chunks:
        add_chunks(vs, chunks, progress)
    return len(documents), len(chunks)
