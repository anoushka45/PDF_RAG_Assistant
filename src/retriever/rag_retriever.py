class RAGRetriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query, top_k):
        query_emb = self.embedder.embed([query])[0]

        results = self.store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]
            })

        return docs
