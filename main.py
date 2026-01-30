from config import *
from src.loaders.pdf_loader import load_pdfs
from src.splitters.text_splitter import split_docs
from src.embeddings.embedding_manager import EmbeddingManager
from src.vectorstore.chroma_store import ChromaStore
from src.retriever.rag_retriever import RAGRetriever
from src.llm.groq_llm import GroqLLM
from src.pipelines.rag_pipeline import run_rag

print("Loading PDFs...")
docs = load_pdfs(PDF_DIR)

print("Splitting documents...")
chunks = split_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)

print("Generating embeddings...")
embedder = EmbeddingManager(EMBEDDING_MODEL)
embeddings = embedder.embed([c.page_content for c in chunks])

print("Saving to vector store...")
store = ChromaStore(VECTOR_DB_DIR)
store.add(chunks, embeddings)

retriever = RAGRetriever(store, embedder)
llm = GroqLLM(LLM_MODEL)

while True:
    query = input("\nAsk a question (or type exit): ")
    if query.lower() == "exit":
        break

    answer, sources = run_rag(query, retriever, llm, TOP_K)
    print("\nAnswer:\n", answer)

    print("\nSources:")
    for s in sources:
        print("-", s["metadata"]["source"])
