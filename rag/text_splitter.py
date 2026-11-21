"""Step 3 of RAG: split long text into small overlapping chunks."""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 900      # characters per chunk
CHUNK_OVERLAP = 150   # shared characters between neighbouring chunks


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
