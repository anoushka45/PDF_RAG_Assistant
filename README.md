# 📄 DocQuery – Modular PDF RAG Assistant

DocQuery is a **modular Retrieval-Augmented Generation (RAG) system** that enables users to ask natural-language questions over a collection of PDF documents.
The system retrieves semantically relevant document chunks using embeddings and a vector database, then generates **grounded, source-aware answers** using an LLM.

This project was built to **understand and implement real-world RAG architecture**, focusing on correctness, modularity, safety, and evaluation rather than just producing answers.

---

## ✨ Key Features

* 📚 **Multi-PDF ingestion** with automatic document parsing
* ✂️ **Configurable text chunking** with overlap for semantic continuity
* 🧠 **SentenceTransformer embeddings** for semantic search
* 🗂️ **ChromaDB vector store** with persistence and metadata
* 🔍 **Top-K similarity-based retrieval**
* 🤖 **LLM-powered answer generation (Groq)**
* 📌 **Source attribution with similarity scores**
* 🧪 **Graceful handling of out-of-scope questions**
* 🛑 **Safety-aware responses** (medical/unsupported queries are refused)
* 🖥️ **Interactive Streamlit UI** with PDF selection

---

📌 **Flow overview**


![alt text](<screenshots/flow diagram.png>)


## 🖥️ User Interface

The Streamlit UI allows users to:

* View available PDFs
* Select which document(s) to query
* Ask natural-language questions
* See generated answers
* Inspect **source documents with similarity scores**

📸 **UI Screenshots**

![alt text](<screenshots/demo query 3.png>) 
![alt text](<screenshots/demo query 2.png>)
 ![alt text](<screenshots/demo query.png>) 
 ![alt text](<screenshots/testing safety .png>)

## 🧪 Evaluation & Observations

### ✔️ In-Scope Questions

* Factual and design-related questions retrieve relevant chunks
* Similarity scores typically range from **0.5–0.8**
* Higher scores are observed for focused, single-intent queries

### ❌ Out-of-Scope Questions

Examples:

* General knowledge (sports, cooking, math)
* Made-up concepts
* Content not present in PDFs

**Behavior:**

* System responds with *“No relevant information found in the provided documents”*
* Prevents hallucinated answers

### 🛑 Safety Handling

When asked **medical or unsupported advisory questions**, the LLM:

* Refuses to answer
* Avoids generating unsafe or misleading content

This aligns with **responsible AI behavior**.

---

## 📊 Similarity Score Interpretation

Similarity scores reflect **retrieval confidence**, not answer quality.

| Score Range | Interpretation            |
| ----------- | ------------------------- |
| 0.75 – 1.0  | Strong semantic match     |
| 0.6 – 0.75  | Good contextual relevance |
| 0.45 – 0.6  | Partial relevance         |
| < 0.45      | Likely unrelated          |

Multi-part questions often yield lower similarity scores due to **embedding dilution**, even when the final answer is correct.

---

## 🛠️ Tech Stack

* **Python**
* **SentenceTransformers**
* **ChromaDB**
* **Groq LLM**
* **Streamlit**
* **LangChain-style RAG concepts (custom implementation)**

---

## 🚀 Getting Started

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Add PDFs

Place your PDF files in:

```
data/pdfs/
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📌 Future Improvements

* Query decomposition for multi-intent questions
* Hybrid search (BM25 + embeddings)
* Inline citations inside answers
* Conversation memory
* REST API backend (FastAPI)




