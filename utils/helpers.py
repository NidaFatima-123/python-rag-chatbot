"""Small helper functions: folders, validation, saving/listing/deleting files."""
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DIR = BASE_DIR / "vectorstore"

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_MB = 10
MAX_QUESTION_CHARS = 3000


def ensure_dirs() -> None:
    for folder in (DATA_DIR, UPLOAD_DIR, VECTOR_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def validate_question(text: str):
    """Return (is_valid, error_message)."""
    if not text or not text.strip():
        return False, "Please type a question first."
    if len(text) > MAX_QUESTION_CHARS:
        return False, f"Your message is too long. Keep it under {MAX_QUESTION_CHARS} characters."
    return True, ""


def is_greeting(text: str) -> bool:
    cleaned = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return cleaned in {"hi", "hello", "hey", "hii", "salam", "assalamualaikum",
                       "good morning", "good evening", "hi there", "hello there"}


def safe_filename(name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_") or "document"
    return f"{stem}{suffix}"


def save_upload(uploaded_file):
    """Validate and save a Streamlit UploadedFile. Returns (ok, message, path)."""
    ensure_dirs()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"{uploaded_file.name}: only PDF, TXT and DOCX files are allowed.", None
    data = uploaded_file.getvalue()
    if len(data) == 0:
        return False, f"{uploaded_file.name}: the file is empty.", None
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return False, f"{uploaded_file.name}: larger than {MAX_FILE_MB} MB.", None
    path = UPLOAD_DIR / safe_filename(uploaded_file.name)
    path.write_bytes(data)
    return True, "saved", path


def list_documents():
    """Every document that belongs to the knowledge base."""
    ensure_dirs()
    docs = []
    for folder, origin in ((DATA_DIR, "Starter"), (UPLOAD_DIR, "Uploaded")):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
                docs.append({"name": p.name, "path": p,
                             "size_kb": max(1, p.stat().st_size // 1024), "origin": origin})
    return docs


def delete_document(path: Path) -> None:
    Path(path).unlink(missing_ok=True)
