from vector_store import SECVectorStore
from config import CSV_PATH
import pandas as pd

# Initialize vector store
vs = SECVectorStore()

# Load SEC filings CSV (make sure it has 'id' and 'text' columns)
df = pd.read_csv(CSV_PATH)

documents = [{"id": str(i), "text": row["text"]} for i, row in df.iterrows()]

# Add to Chroma DB
vs.add_documents(documents)

print(f"Added {vs.count()} documents to the vector store.")
