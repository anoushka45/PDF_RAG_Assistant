def run_rag(query, retriever, llm, top_k):
    docs = retriever.retrieve(query, top_k)

    context = "\n\n".join(d["content"] for d in docs)

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    return llm.generate(prompt), docs
