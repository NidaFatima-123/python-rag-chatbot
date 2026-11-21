"""Step 1-2 of RAG: load documents and extract their text."""
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from utils.helpers import ALLOWED_EXTENSIONS, DATA_DIR, UPLOAD_DIR


def _load_pdf(path: Path) -> List[Document]:
    from pypdf import PdfReader
    docs = []
    reader = PdfReader(str(path))
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text,
                                 metadata={"source": path.name, "page": number}))
    return docs


def _load_txt(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [Document(page_content=text, metadata={"source": path.name})] if text else []


def _load_docx(path: Path) -> List[Document]:
    from docx import Document as DocxDocument
    parts = [p.text for p in DocxDocument(str(path)).paragraphs if p.text.strip()]
    text = "\n".join(parts).strip()
    return [Document(page_content=text, metadata={"source": path.name})] if text else []


def load_file(path: Path) -> List[Document]:
    """Load one PDF / TXT / DOCX file. Raises ValueError for unsupported types."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".txt":
        return _load_txt(path)
    if suffix == ".docx":
        return _load_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_all_documents() -> List[Document]:
    """Load the starter data and all uploaded files. Bad files are skipped."""
    documents: List[Document] = []
    for folder in (DATA_DIR, UPLOAD_DIR):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                try:
                    documents.extend(load_file(path))
                except Exception as error:  # keep going if one file is broken
                    print(f"[warning] Could not read {path.name}: {error}")
    return documents
