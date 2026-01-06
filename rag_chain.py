from transformers import pipeline

# Initialize LLM
llm = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",   # use "small" if slow
    max_new_tokens=300
)

MAX_CONTEXT_CHARS = 1500   # 🔥 CRITICAL FIX


def generate_answer(context: str, question: str) -> str:
    # 🔹 Truncate context to avoid token overflow
    context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are a financial analyst assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm(prompt)
    return response[0]["generated_text"]
