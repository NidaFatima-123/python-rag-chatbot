# Python Programming AI Tutor – RAG-Based Chatbot
# URL of Python AI Tutor https://python-rag-chatbot-eirutgyjmhwhp26wnuf6nn.streamlit.app/

A Streamlit chatbot that answers Python questions **from your documents** (RAG), with source citations.

**Stack:** Streamlit · LangChain · sentence-transformers/all-MiniLM-L6-v2 · ChromaDB · Google Gemini · PyPDF

> **Honest status:** The Python files pass a syntax check, but this project has **not been run end to end**
> (no internet or API key was available when it was written). If you hit an error, see Troubleshooting.

## 1. Install (Python 3.10 – 3.12 recommended)

```bash
cd python-rag-chatbot
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## 2. Get and set your Gemini API key
1. Open https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Copy `.env.example` to a new file named `.env` and paste the key:
   ```
   GOOGLE_API_KEY=AIza...your_key...
   ```
The key stays on the server in `.env`; it is never sent to the browser. Never share or upload `.env`.

## 3. Run
```bash
streamlit run app.py
```
On the **first start** the app downloads the embedding model (~90 MB) and indexes the starter files in `data/`.
This takes a minute or two; later starts are fast.

## 4. Optional: add the official Python tutorial
```bash
python scripts/download_docs.py
```
It downloads real pages from docs.python.org/3/tutorial/ into `data/official_docs/`. Then click
**Rebuild knowledge base** in the sidebar.

## How it works (RAG pipeline)
1. `rag/document_loader.py` reads PDF / TXT / DOCX files from `data/` and `uploads/`.
2. `rag/text_splitter.py` cuts text into ~900-character overlapping chunks.
3. `rag/embeddings.py` turns each chunk into a vector (MiniLM).
4. `rag/vector_store.py` saves vectors in ChromaDB (`vectorstore/` folder) and searches them.
5. `rag/rag_chain.py` embeds your question, retrieves the closest chunks, and sends
   *chunks + question* to Gemini. If nothing relevant is found, it says so and does **not** call Gemini.
6. `app.py` is the interface: chat, history, uploads, document list, settings, sources.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (run this) |
| `rag/*.py` | The RAG pipeline, one step per file |
| `utils/helpers.py` | Paths, validation, file save/list/delete |
| `data/*.txt` | Starter Python knowledge base |
| `scripts/download_docs.py` | Optional downloader for the official tutorial |
| `.env` | Your secret API key (you create it) |
| `.streamlit/config.toml` | Theme and upload limit |

## Using the app
- **New chat / Clear chat / Chat history** – top of the sidebar (history lasts while the browser tab session lives).
- **Upload documents** – add PDF/TXT/DOCX; they are indexed immediately. Scanned (image-only) PDFs have no text and can't be read.
- **View documents** – delete any file; its chunks are removed from ChromaDB too.
- **Rebuild knowledge base** – re-index everything from scratch.
- **Settings** – explanation level, how many chunks to retrieve, minimum relevance.
- Code blocks are syntax highlighted and have a copy button (built into Streamlit).

## Troubleshooting
| Problem | Fix |
|---|---|
| `No Gemini API key found` | Create `.env` from `.env.example`, paste key, restart Streamlit. |
| `model not found` (404) | Set `GEMINI_MODEL=` in `.env` to a model your key can use (see AI Studio). |
| `429` / quota | Free-tier limit reached. Wait and retry. |
| `pip install` fails on a package | Use Python 3.10–3.12; run `pip install --upgrade pip` first. |
| Import errors from `langchain_*` | Reinstall inside a fresh venv with `pip install -r requirements.txt`. |
| Always "couldn't find enough information" | Lower *Minimum relevance* in Settings, or rebuild the knowledge base. |
| Weird results after deleting `vectorstore/` | Click **Rebuild knowledge base**. |
| Dark mode looks off in some parts | It is a CSS overlay; the light theme is the best tested design. |
