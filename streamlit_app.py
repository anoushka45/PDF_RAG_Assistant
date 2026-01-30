import streamlit as st

from config import *
from src.loaders.pdf_loader import load_pdfs
from src.splitters.text_splitter import split_docs
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.chroma_store import ChromaStore
from src.retriever.rag_retriever import RAGRetriever
from src.llm.groq_llm import GroqLLM
from src.pipelines.rag_pipeline import run_rag

st.set_page_config(page_title="DocQuery - PDF RAG", layout="wide")

st.title("📄 DocQuery – PDF RAG Assistant")
st.caption("Ask questions from your PDFs using Retrieval-Augmented Generation")

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

with st.spinner("Setting up RAG pipeline..."):
    retriever, llm = setup_rag()

query = st.text_input("Ask a question from your documents:")

if query:
    with st.spinner("Thinking..."):
        answer, sources = run_rag(query, retriever, llm, TOP_K)

    st.subheader("✅ Answer")
    st.write(answer)

    st.subheader("📚 Sources")
    for s in sources:
        st.markdown(f"- **{s['metadata']['source']}** (score: {s['score']:.2f})")