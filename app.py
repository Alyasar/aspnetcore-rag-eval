"""Evaluated RAG for ASP.NET Core: the Streamlit web app.

Ask questions about ASP.NET Core 10 (Blazor, Fundamentals, MVC). Answers come only from
Microsoft's official docs, with numbered sources. The retrieval + generation code lives in rag.py.
"""

import os                                               # os: environment variables (settings and the API key)
from pathlib import Path                                # Path: file paths

import pandas as pd                                     # pandas: reading the evaluation result files
import streamlit as st                                  # streamlit: turns this script into a web app

st.set_page_config(page_title="ASP.NET Core Docs Q&A", page_icon="🔎", layout="wide")   # Must be the first Streamlit command

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Loading overlay ----------------
# A full-screen layer with a centred spinner. It sits on top of everything (z-index), so nothing
# underneath can be clicked, and it hides Streamlit's "Running… Stop" indicator in the corner.
LOADER_HTML = """
<style>
#app-loader {
    position: fixed; inset: 0; z-index: 999999;          /* cover the whole window, above everything */
    background: rgba(255, 255, 255, 0.96);                /* almost opaque white */
    display: flex; flex-direction: column;                /* stack spinner and text vertically... */
    align-items: center; justify-content: center;         /* ...centred on the screen */
    gap: 18px;
}
#app-loader .spinner {
    width: 56px; height: 56px; border-radius: 50%;
    border: 6px solid #e5e7eb; border-top-color: #e5484d; /* grey ring with one red segment */
    animation: spin 0.9s linear infinite;                 /* rotate forever */
}
#app-loader p { font-size: 2rem; color: #444; margin: 0; }
@keyframes spin { to { transform: rotate(360deg); } }
[data-testid="stStatusWidget"] { visibility: hidden; }    /* hide Streamlit's corner "Running… Stop" */
</style>
<div id="app-loader">
    <div class="spinner"></div>
    <p>Loading the search models… (The first visit can take up to a minute)</p>
</div>
"""

loader = st.empty()                                     # A placeholder we can fill now and clear later
if not st.session_state.get("ready"):                   # Only for a visitor's first page load, not on every click
    loader.markdown(LOADER_HTML, unsafe_allow_html=True)   # Show the overlay immediately

# On Streamlit Cloud, settings live in "secrets". Copy them into environment variables, where rag.py reads them.
# Locally there is usually no secrets file, so this does nothing and your setx variables are used instead.
try:
    for key in ("GROQ_API_KEY", "RAG_LLM_PROVIDER", "RAG_LLM"):
        if key in st.secrets:
            os.environ.setdefault(key, str(st.secrets[key]))
except Exception:                                       # No secrets file: that's fine
    pass

from rag import RagPipeline, NOT_FOUND                  # The slow import (PyTorch etc.), now happening behind the overlay

REPORTS = Path(__file__).resolve().parent / "reports"   # Where the evaluation results are saved
MAX_QUESTIONS_PER_SESSION = 20                          # Protects the free Groq quota from one visitor using it all

EXAMPLES = [                                            # Example questions shown as buttons
    "What does the @key directive attribute do in Blazor?",
    "How do I apply a filter to all controllers globally?",
    "How do I read changed appsettings.json values while the app is running?",
]


class GenerationError(Exception):                       # Raised when the LLM call fails, so failed answers are never cached
    pass


@st.cache_resource(show_spinner=False)                  # No extra spinner: the overlay already shows loading
def load_pipeline():                                    # Loads the models ONCE and shares them across all visitors
    return RagPipeline(max_retries=2)                   # Only 2 retries: visitors shouldn't wait minutes on a rate limit


@st.cache_data(show_spinner=False, max_entries=500, ttl=24 * 3600)
def cached_ask(question):                               # Answers repeated questions from a cache (saves quota and time)
    result = load_pipeline().ask(question)
    if result["answer"].startswith("[Generation failed"):
        raise GenerationError(result["answer"])         # Exceptions aren't cached, so the next try calls the LLM again
    return result


load_pipeline()                                         # Load the models NOW (behind the overlay), not on the first question
loader.empty()                                          # Everything is ready: remove the overlay
st.session_state.ready = True                           # Remember it, so the overlay isn't shown again in this session


def set_question(q):                                    # Runs when an example button is clicked
    st.session_state.question = q                       # Put the example into the text box...
    st.session_state.submit = True                      # ...and ask it straight away


def percent_table(df, cols):                            # Format fractions (0.78) as percentages ("78%") for display
    out = df.copy()
    for c in cols:
        out[c] = (out[c] * 100).round().astype(int).astype(str) + "%"
    return out


# ---------------- Sidebar: how it works and the evaluation ----------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Hybrid search** (vector + keyword) finds 30 candidate passages in 6,601 chunks of the docs\n"
        "2. A **reranker** picks the best 5\n"
        "3. An **LLM** answers using only those 5 passages, citing them as [1]…[5]\n\n"
        "Docs: ASP.NET Core 10, Blazor, Fundamentals and MVC "
        "([CC BY 4.0, Microsoft](https://github.com/dotnet/AspNetCore.Docs))."
    )
    st.header("Evaluation")
    st.caption("50 questions with known answers, hand-graded.")
    retrieval_file = REPORTS / "retrieval_comparison.csv"
    if retrieval_file.exists():
        ret = pd.read_csv(retrieval_file, index_col=0)[["Hit@5", "MRR"]]
        ret["MRR"] = ret["MRR"].round(3)
        st.markdown("**Retrieval**: right passage in top 5")
        st.dataframe(percent_table(ret, ["Hit@5"]), use_container_width=True)
    gen_files = sorted(REPORTS.glob("generation_summary__*.csv"))
    if gen_files:
        gen = pd.concat(pd.read_csv(p, index_col=0) for p in gen_files)[["Score", "Hallucinated"]]
        gen.index = gen.index.str.replace("hybrid_rerank__", "", regex=False)   # Shorter row names
        st.markdown("**Answers**: graded against expected answers")
        st.dataframe(percent_table(gen, ["Score", "Hallucinated"]), use_container_width=True)
    st.caption("Every wrong answer came from a retrieval miss: the search, not the LLM, is the bottleneck.")

# ---------------- Main area ----------------
st.title("🔎 Evaluated RAG for ASP.NET Core")
st.write("Ask about **ASP.NET Core 10**: Blazor, Fundamentals (DI, middleware, configuration, routing…) and MVC. "
         "Answers come only from Microsoft's official documentation, with sources.")

with st.form("ask_form"):                               # A form: pressing Enter or the button submits the question
    st.text_input("Your question", key="question",
                  placeholder="e.g. How do I register a keyed service?")
    submitted = st.form_submit_button("Ask", type="primary")

st.caption("Or try an example:")
for col, q in zip(st.columns(len(EXAMPLES)), EXAMPLES):     # One button per example, side by side
    col.button(q, on_click=set_question, args=(q,), use_container_width=True)

submit = submitted or st.session_state.pop("submit", False)  # Asked via the form OR an example button
question = st.session_state.get("question", "").strip()      # The question text, without extra spaces
st.session_state.setdefault("count", 0)                      # How many questions this visitor has asked

if submit and question:
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.warning("You've reached the question limit for this session (this demo runs on a free API quota). Thanks for trying it!")
        st.stop()
    with st.spinner("Searching the docs and writing an answer…"):
        try:
            result = cached_ask(question)
        except GenerationError:
            st.error("The language model is busy right now (free-tier rate limit). Please try again in a minute.")
            st.stop()
    st.session_state.count += 1

    st.subheader("Answer")
    if NOT_FOUND.lower() in result["answer"].replace("’", "'").lower():
        st.info(result["answer"] + "  \nThis demo only covers ASP.NET Core 10 Blazor, Fundamentals and MVC docs.")
    else:
        st.markdown(result["answer"])
    st.caption(f"{result['model']} · {result['total_seconds']} s")

    st.subheader("Sources")
    for i, h in enumerate(result["hits"], start=1):     # The 5 passages the answer was based on
        with st.expander(f"[{i}] {h['where']}"):
            st.markdown(f"[Open on learn.microsoft.com]({h['url']}) · reranker score {h['rerank_score']}")
            st.markdown(h["text"])