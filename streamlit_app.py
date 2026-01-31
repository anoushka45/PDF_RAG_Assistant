import streamlit as st
from pathlib import Path

from config import *
from src.loaders.pdf_loader import load_pdfs
from src.splitters.text_splitter import split_docs
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.chroma_store import ChromaStore
from src.retriever.rag_retriever import RAGRetriever
from src.llm.groq_llm import GroqLLM
from src.pipelines.rag_pipeline import run_rag


# -------------------- Page Config --------------------
st.set_page_config(
    page_title="DocQuery - PDF RAG",
    layout="wide",
    page_icon="📄"
)

# -------------------- Header --------------------
st.title("📄 DocQuery")
st.caption("Ask intelligent questions from your PDFs using Retrieval-Augmented Generation")

# -------------------- Sidebar --------------------
st.sidebar.header("📂 Available PDFs")

pdf_files = sorted([p.name for p in Path(PDF_DIR).glob("*.pdf")])

if not pdf_files:
    st.sidebar.warning("No PDFs found in data/raw_pdfs")
    st.stop()

selected_pdfs = st.sidebar.multiselect(
    "Select PDFs to query from:",
    pdf_files,
    default=pdf_files
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 Tip: Select one or multiple PDFs to limit the search scope."
)

# -------------------- RAG Setup --------------------
@st.cache_resource
def setup_rag():
    docs = load_pdfs(PDF_DIR)
    chunks = split_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)

    embedder = EmbeddingManager(EMBEDDING_MODEL)
    embeddings = embedder.embed([c.page_content for c in chunks])

    store = ChromaStore(VECTOR_DB_DIR)
    store.add(chunks, embeddings)

    retriever = RAGRetriever(store, embedder)
    llm = GroqLLM(LLM_MODEL)

    return retriever, llm


with st.spinner("🔧 Initializing RAG pipeline..."):
    retriever, llm = setup_rag()

# -------------------- Query Input --------------------
st.markdown("### 💬 Ask a Question")
query = st.text_input(
    "Type your question below:",
    placeholder="e.g. What is attention mechanism?"
)

# -------------------- Query Handling --------------------
if query and selected_pdfs:
    with st.spinner("🤔 Thinking..."):
        answer, sources = run_rag(query, retriever, llm, TOP_K)

        # filter sources based on selected PDFs
        filtered_sources = [
            s for s in sources if s["metadata"]["source"] in selected_pdfs
        ]

    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 📚 Sources Used")

    if filtered_sources:
        for s in filtered_sources:
            st.markdown(
                f"""
                🔹 **{s['metadata']['source']}**  
                Similarity score: `{s['score']:.2f}`
                """
            )
    else:
        st.warning("No matching sources found in selected PDFs.")

elif query and not selected_pdfs:
    st.warning("Please select at least one PDF from the sidebar.")
