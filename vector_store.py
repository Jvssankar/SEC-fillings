import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL


class SECVectorStore:
    def __init__(self):
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DB_DIR
        )

    def add_documents(self, documents):
        if not documents:
            return
        self.db.add_documents(documents)
        # ❌ DO NOT CALL persist() — handled automatically

    def retrieve(self, query, k=5, filters=None):
        chroma_filter = None

        if filters:
            chroma_filter = {
                "$and": [{"$eq": {k: v}} for k, v in filters.items()]
            }

        return self.db.similarity_search(
            query=query,
            k=k,
            filter=chroma_filter
        )

    def count(self):
        return self.db._collection.count()

    def debug_metadata(self, limit=5):
        results = self.db.get(limit=limit)
        for i, meta in enumerate(results["metadatas"]):
            print(f"\nMetadata {i+1}:")
            print(meta)
