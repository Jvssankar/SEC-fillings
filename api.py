from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import SECRAGChain
from vector_store import SECVectorStore

app = FastAPI(title="SEC Filing RAG System")

rag = SECRAGChain()
store = SECVectorStore()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/")
def root():
    return {"message": "SEC Filing Q&A System running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/info")
def info():
    return store.get_info()


@app.post("/ask")
def ask_question(req: QuestionRequest):
    return rag.ask(req.question, req.top_k)
