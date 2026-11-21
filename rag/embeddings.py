"""Step 4 of RAG: turn text into vectors with Sentence Transformers."""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Loaded once and reused. The first call downloads the model (~90 MB)."""
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
