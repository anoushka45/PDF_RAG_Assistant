# 📄 DocQuery – PDF RAG Assistant

DocQuery is a modular **Retrieval-Augmented Generation (RAG)** system that allows users
to ask natural language questions over a collection of PDF documents.

Instead of relying solely on an LLM (which may hallucinate), this system retrieves
relevant document chunks using vector similarity search and generates **grounded,
context-aware answers**.

---

## 🚀 Features

- Load and process multiple PDF documents
- Chunk documents with configurable overlap
- Generate semantic embeddings using SentenceTransformers
- Store and search embeddings using ChromaDB
- Retrieve top-k relevant chunks for a query
- Generate accurate answers using Groq-hosted LLMs
- Simple Streamlit-based UI for interaction

---

## 🧠 Architecture Overview

PDFs
↓
Text Chunking
↓
Embeddings (SentenceTransformer)
↓
Vector Store (ChromaDB)
↓
Retriever (Top-K Similarity)
↓
LLM (Groq)
↓
Answer + Sources



## 🛠 Tech Stack

- **Python**
- **SentenceTransformers**
- **ChromaDB**
- **LangChain**
- **Groq LLM**
- **Streamlit**