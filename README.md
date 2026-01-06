# SEC Filing Summarizer & Verified Q&A System (RAG-Based)

## Project Description
This project is an AI-powered SEC filing analysis system that allows users to ask questions on company filings such as 10-K and 10-Q reports and receive factually accurate answers, verified source citations, and concise summaries.  
It uses Retrieval-Augmented Generation (RAG) to ensure all responses are strictly grounded in official SEC documents, reducing hallucinations and improving trustworthiness.

## Project Objectives
- Enable easy understanding of complex SEC filings  
- Provide document-backed answers with citations  
- Generate clear summaries alongside verified answers  
- Improve reliability of AI-generated financial insights  

## Technology Stack

### Programming & Data Processing
- Python  
- Pandas  

### Document Processing
- SEC filing text extraction  
- Document chunking and metadata handling  

### AI & NLP
- Large Language Model (LLM) for question answering and summarization  
- Text embeddings for semantic search (e.g., `text-embedding-3-small`)  

### Retrieval System
- Retrieval-Augmented Generation (RAG)  

### Frameworks & Tools
- LangChain (pipelines, retrieval, prompts)  
- Pydantic (structured and validated outputs)  

## Key Features
- Ask natural language questions on SEC filings  
- Answers verified with source citations  
- Summarized explanations for better readability  
- Works with real SEC filing datasets  
- Reduced hallucinations through document grounding  

## Output Format
Each query returns:
- **Verified Answer** – generated strictly from filing content  
- **Summary** – simplified explanation of the retrieved sections  
- **Source Reference** – document chunk or section used  
- **Claim Percentage** – indicates whether the query is satisfied, partially satisfied, or not satisfied  

## Dataset
**SEC Filings Dataset (Kaggle):**  
https://www.kaggle.com/datasets/kharanshuvalangar/sec-filings
