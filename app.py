"""Python Programming AI Tutor - RAG-based chatbot (Streamlit front end)."""
import os
import uuid

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Python AI Tutor", page_icon="🐍", layout="wide",
                   initial_sidebar_state="auto")

from rag.document_loader import load_file
from rag.rag_chain import ask, GREETING_MESSAGE
from rag.text_splitter import split_documents
from rag.vector_store import (add_chunks, count_chunks, delete_source,
                              get_vectorstore, rebuild_vectorstore)
from utils.helpers import (delete_document, ensure_dirs, is_greeting, list_documents,
                           save_upload, validate_question)

SUGGESTED = [
    "What is the difference between a list and a tuple?",
    "Explain for loops with an example",
    "How do I define a function with default arguments?",
    "Explain inheritance in Python with code",
    "How do I read a file safely in Python?",
    "Create 3 practice questions on dictionaries",
    "What is a DataFrame in Pandas?",
    "Explain try, except and finally line by line",
]

PALETTES = {
    "light": dict(bg="#F5F7FB", panel="#FFFFFF", side="#EAF0F8", text="#16202E", muted="#5B6778",
                  border="#D9E0EC", accent="#3776AB", accent2="#FFD43B", bubble="#FFFFFF",
                  user="#E3EEF9", code="#F0F3F9", codetext="#16202E"),
    "dark": dict(bg="#0E1621", panel="#152233", side="#0A111B", text="#E8EEF7", muted="#9AA8BC",
                 border="#25364B", accent="#5AA2DB", accent2="#FFD43B", bubble="#152233",
                 user="#1C3350", code="#0A111B", codetext="#E8EEF7"),
}

STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
code, pre, [data-testid="stCode"] * { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { max-width: 880px; padding-top: 1.6rem; padding-bottom: 7rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"] { color: var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp small { color: var(--muted); }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }
.brand { display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .6rem; }
.brand .logo { width:42px; height:42px; border-radius:12px; display:grid; place-items:center;
  font-size:1.35rem; background: linear-gradient(135deg, var(--accent) 55%, var(--accent2) 55%); }
.brand b { font-size:1.05rem; display:block; line-height:1.15; }
.brand span { font-size:.78rem; color: var(--muted); }
.kb-card { background: var(--panel); border:1px solid var(--border); border-radius:12px;
  padding:.7rem .9rem; font-size:.88rem; }
.kb-card .ok { color:#2BA36B; font-weight:700; } .kb-card .warn { color:#D98A00; font-weight:700; }

/* Buttons */
.stButton > button, .stDownloadButton > button { border-radius: 10px; border:1px solid var(--border);
  background: var(--panel); color: var(--text); transition: border-color .15s, background .15s; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button:focus-visible { outline: 3px solid var(--accent2); outline-offset: 1px; }
.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color:#fff; }
.stButton > button[kind="primary"]:hover { color:#fff; filter: brightness(1.08); }

/* Hero: the one memorable element */
.hero { background: var(--side); border:1px solid var(--border); border-radius: 16px; overflow:hidden;
  margin-bottom:1.2rem; }
.hero .bar { display:flex; gap:6px; padding:.55rem .8rem; border-bottom:1px solid var(--border); }
.hero .bar i { width:10px; height:10px; border-radius:50%; background: var(--border); display:block; }
.hero .body { padding:1.2rem 1.4rem 1.4rem; font-family:'JetBrains Mono', monospace; }
.hero .prompt { color: var(--accent); font-weight:600; font-size:1.55rem; line-height:1.3; }
.hero .prompt em { font-style:normal; color: var(--text); }
.hero .cursor { display:inline-block; width:.6ch; height:1.1em; background: var(--accent2);
  vertical-align:-0.15em; margin-left:2px; animation: blink 1.1s steps(1) infinite; }
.hero .sub { font-family:'DM Sans',sans-serif; color: var(--muted); margin-top:.6rem; font-size:1rem; }
@keyframes blink { 50% { opacity:0; } }
@media (prefers-reduced-motion: reduce) { .hero .cursor { animation: none; } }

/* Chat */
[data-testid="stChatMessage"] { background: var(--bubble); border:1px solid var(--border);
  border-radius: 14px; padding: .8rem 1rem; margin-bottom:.7rem; }
[data-testid="stChatMessage"]:has([data-testid*="User"]) { background: var(--user); }
[data-testid="stChatInput"] { border-radius: 14px; }
[data-testid="stChatInput"] textarea { color: var(--text); }
[data-testid="stCode"], pre { border-radius: 10px; }
.stApp pre, .stApp [data-testid="stCode"] pre { background: var(--code) !important; }
.stApp pre code, .stApp [data-testid="stCode"] code { color: var(--codetext); }
.stApp :not(pre) > code { background: var(--code); color: var(--accent); padding:.1em .35em; border-radius:6px; }
[data-testid="stExpander"] { background: var(--panel); border:1px solid var(--border); border-radius:12px; }

/* Phones and small tablets */
@media (max-width: 768px) {
  .block-container { padding: 1rem .8rem 7rem; }
  .hero .prompt { font-size: 1.15rem; }
  [data-testid="stChatMessage"] { padding: .65rem .7rem; }
}
"""


def inject_css() -> None:
    p = PALETTES["dark" if st.session_state.dark else "light"]
    variables = ":root{" + ";".join(f"--{k}:{v}" for k, v in p.items()) + "}"
    st.markdown(f"<style>{variables}{STATIC_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- state
def new_chat() -> None:
    ss = st.session_state
    current = ss.chats.get(ss.current)
    if current is not None and not current["messages"]:
        return  # already on an empty chat
    cid = uuid.uuid4().hex[:8]
    ss.chats[cid] = {"title": "New chat", "messages": []}
    ss.current = cid


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("chats", {})
    ss.setdefault("current", None)
    ss.setdefault("dark", False)
    ss.setdefault("pending", None)
    ss.setdefault("top_k", 4)
    ss.setdefault("min_score", 0.25)
    ss.setdefault("level", "Beginner")
    ss.setdefault("uploader_key", 0)
    ss.setdefault("kb_ready", False)
    if ss.current is None or ss.current not in ss.chats:
        new_chat()


def select_chat(cid: str) -> None:
    st.session_state.current = cid


def clear_chat() -> None:
    chat = st.session_state.chats[st.session_state.current]
    chat["messages"] = []
    chat["title"] = "New chat"


def set_pending(question: str) -> None:
    st.session_state.pending = question


# ---------------------------------------------------------------- knowledge base
def ensure_knowledge_base() -> None:
    if st.session_state.kb_ready:
        return
    try:
        if count_chunks(get_vectorstore()) == 0:
            with st.spinner("First start: building the knowledge base. "
                            "This downloads a ~90 MB embedding model once..."):
                rebuild_vectorstore()
        st.session_state.kb_ready = True
    except Exception as error:
        st.error(f"Could not prepare the knowledge base: {error}")


def friendly_error(error: Exception) -> str:
    text = str(error)
    low = text.lower()
    if "MISSING_API_KEY" in text:
        return ("No Gemini API key found. Copy `.env.example` to `.env`, paste your key after "
                "`GOOGLE_API_KEY=`, then restart the app.")
    if "api key" in low and ("invalid" in low or "not valid" in low):
        return "Gemini rejected the API key. Check the key in your `.env` file."
    if "429" in text or "quota" in low or "rate" in low:
        return "Gemini rate limit or quota reached. Wait a minute and try again."
    if "404" in text or "not found" in low:
        return "Gemini model not found. Set another `GEMINI_MODEL` in `.env`."
    return f"Something went wrong: {text[:300]}"


# ---------------------------------------------------------------- sidebar
def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">🐍</div><div><b>Python AI Tutor</b>'
                    '<span>Answers from your documents</span></div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button("New chat", on_click=new_chat, use_container_width=True, type="primary")
        c2.button("Clear chat", on_click=clear_chat, use_container_width=True)
        st.toggle("Dark mode", key="dark")

        st.markdown("**Chat history**")
        for cid, chat in reversed(list(ss.chats.items())):
            label = chat["title"][:30] + ("..." if len(chat["title"]) > 30 else "")
            st.button(label, key=f"chat_{cid}", on_click=select_chat, args=(cid,),
                      use_container_width=True,
                      type="primary" if cid == ss.current else "secondary")

        st.divider()
        render_upload_section()
        st.divider()
        render_kb_status()
        st.divider()
        with st.expander("Settings"):
            ss.level = st.selectbox("Explanation level", ["Beginner", "Intermediate", "Advanced"],
                                    index=["Beginner", "Intermediate", "Advanced"].index(ss.level))
            ss.top_k = st.slider("Chunks to retrieve", 1, 8, ss.top_k)
            ss.min_score = st.slider("Minimum relevance", 0.0, 0.8, float(ss.min_score), 0.05,
                                     help="Lower = more results, higher = stricter matches.")


def render_upload_section() -> None:
    ss = st.session_state
    st.markdown("**Upload documents**")
    files = st.file_uploader("PDF, TXT or DOCX (max 10 MB each)", type=["pdf", "txt", "docx"],
                             accept_multiple_files=True, key=f"uploader_{ss.uploader_key}")
    if st.button("Add to knowledge base", disabled=not files, use_container_width=True):
        with st.status("Processing documents...", expanded=True) as status:
            vs = get_vectorstore()
            for f in files:
                ok, message, path = save_upload(f)
                if not ok:
                    st.write(f"❌ {message}")
                    continue
                try:
                    st.write(f"📄 Reading {path.name}")
                    delete_source(vs, path.name)  # replace old version if it exists
                    docs = load_file(path)
                    if not docs:
                        st.write(f"❌ {path.name}: no readable text (scanned PDF?)")
                        delete_document(path)
                        continue
                    chunks = split_documents(docs)
                    add_chunks(vs, chunks)
                    st.write(f"✅ {path.name}: {len(chunks)} chunks indexed")
                except Exception as error:
                    st.write(f"❌ {path.name}: {error}")
            status.update(label="Done", state="complete", expanded=False)
        ss.uploader_key += 1
        st.rerun()


def render_kb_status() -> None:
    docs = list_documents()
    try:
        chunks = count_chunks(get_vectorstore())
    except Exception:
        chunks = 0
    if chunks:
        state = '<span class="ok">Ready</span>'
    else:
        state = '<span class="warn">Empty - rebuild needed</span>'
    st.markdown("**Knowledge base**")
    st.markdown(f'<div class="kb-card">Status: {state}<br>Documents: <b>{len(docs)}</b><br>'
                f'Chunks: <b>{chunks}</b></div>', unsafe_allow_html=True)

    with st.expander(f"View documents ({len(docs)})"):
        if not docs:
            st.caption("No documents yet. Upload a file above.")
        for d in docs:
            a, b = st.columns([5, 1])
            a.markdown(f"{d['name']}  \n<small>{d['origin']} · {d['size_kb']} KB</small>",
                       unsafe_allow_html=True)
            if b.button("🗑", key=f"del_{d['origin']}_{d['name']}", help=f"Delete {d['name']}"):
                try:
                    delete_source(get_vectorstore(), d["name"])
                    delete_document(d["path"])
                    st.toast(f"Deleted {d['name']}")
                except Exception as error:
                    st.toast(f"Could not delete: {error}")
                st.rerun()

    if st.button("Rebuild knowledge base", use_container_width=True):
        bar = st.progress(0.0, text="Rebuilding...")
        try:
            n_docs, n_chunks = rebuild_vectorstore(lambda x: bar.progress(x, text="Indexing..."))
            bar.empty()
            st.toast(f"Rebuilt: {n_docs} documents, {n_chunks} chunks")
        except Exception as error:
            bar.empty()
            st.error(f"Rebuild failed: {error}")
        st.rerun()


# ---------------------------------------------------------------- main area
def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            page = f" · page {s['page']}" if s.get("page") else ""
            st.markdown(f"**[{i}] {s['source']}**{page} · relevance {s['score']:.0%}")
            st.caption(s["excerpt"])


def render_hero() -> None:
    st.markdown(
        '<div class="hero"><div class="bar"><i></i><i></i><i></i></div><div class="body">'
        '<div class="prompt">&gt;&gt;&gt; <em>Ask me anything about Python</em><span class="cursor"></span></div>'
        '<div class="sub">I search the knowledge base first, then explain what I find with '
        'examples and sources. Try one of these:</div></div></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED):
        cols[i % 2].button(q, key=f"sug_{i}", on_click=set_pending, args=(q,),
                           use_container_width=True)


def render_chat() -> None:
    ss = st.session_state
    chat = ss.chats[ss.current]
    api_ok = os.getenv("GOOGLE_API_KEY", "").strip() not in ("", "your_api_key_here")
    if not api_ok:
        st.warning("Add your Gemini API key to `.env` (see README) and restart the app.")

    if not chat["messages"]:
        render_hero()

    for msg in chat["messages"]:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🐍"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources"))

    prompt = st.chat_input("Ask a Python question, paste code, or paste an error...")
    if ss.pending:
        prompt, ss.pending = ss.pending, None
    if not prompt:
        return

    valid, error_text = validate_question(prompt)
    if not valid:
        st.error(error_text)
        return

    history = [{"role": m["role"], "content": m["content"]} for m in chat["messages"]]
    chat["messages"].append({"role": "user", "content": prompt})
    if chat["title"] == "New chat":
        chat["title"] = prompt.strip().replace("\n", " ")

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🐍"):
        if is_greeting(prompt):
            result = {"answer": GREETING_MESSAGE, "sources": []}
        else:
            try:
                with st.spinner("Searching the knowledge base and thinking..."):
                    result = ask(prompt, history, k=ss.top_k, min_score=ss.min_score, level=ss.level)
            except Exception as error:
                chat["messages"].pop()  # let the student retry the same question
                st.error(friendly_error(error))
                return
        st.markdown(result["answer"])
        render_sources(result["sources"])
    chat["messages"].append({"role": "assistant", "content": result["answer"],
                             "sources": result["sources"]})
    st.rerun()


def main() -> None:
    ensure_dirs()
    init_state()
    inject_css()
    ensure_knowledge_base()
    render_sidebar()
    render_chat()


main()
