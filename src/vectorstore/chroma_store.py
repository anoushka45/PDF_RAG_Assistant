import chromadb
import os
import uuid

class ChromaStore:
    def __init__(self, persist_dir):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("pdf_docs")

    def add(self, documents, embeddings):
        ids, texts, metas, embs = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(str(uuid.uuid4()))
            texts.append(doc.page_content)
            metas.append(doc.metadata)
            embs.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embs
        )
