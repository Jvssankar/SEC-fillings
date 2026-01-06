# query_engine.py

from vector_store import SECVectorStore
from transformers import pipeline
from langchain_core.prompts import PromptTemplate


vector_store = SECVectorStore()

qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

summary_model = pipeline(
    "summarization",
    model="google/flan-t5-base",
    max_length=120,
    min_length=40
)

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer STRICTLY using the context below.
If the answer is not present, say "Not found in the filing."

Context:
{context}

Question:
{question}

Answer:
"""
)


def calculate_claim_percentage(answer: str):
    if "not found" in answer.lower():
        return "0% (Not satisfied)"
    if len(answer.split()) < 25:
        return "50% (Partially satisfied)"
    return "90% (Satisfied)"


def ask_question(company, form_type, year, question):
    # 🔑 Chroma filters (ONLY supported keys)
    filters = {
        "company": company,
        "form_type": form_type
    }

    docs = vector_store.retrieve(question, k=5, filters=filters)

    # 🔑 YEAR filtering done safely in Python
    docs = [
        d for d in docs
        if d.metadata.get("filed_at", "").startswith(year)
    ]

    if not docs:
        print("❌ No matching filing found.")
        return

    context = ""
    sources = set()

    for i, doc in enumerate(docs, 1):
        context += f"\n[Source {i}]\n{doc.page_content}\n"
        if "url" in doc.metadata:
            sources.add(doc.metadata["url"])

    prompt = PROMPT.format(
        context=context[:1500],
        question=question
    )

    answer = qa_model(prompt)[0]["generated_text"]
    summary = summary_model(context[:1000])[0]["summary_text"]
    claim = calculate_claim_percentage(answer)

    print("\n✅ VERIFIED ANSWER:\n", answer)
    print("\n📝 SUMMARY:\n", summary)

    print("\n📌 SOURCE URL(S):")
    for src in sources:
        print("-", src)

    print("\n📊 CLAIM PERCENTAGE:", claim)
    print("-" * 60)


if __name__ == "__main__":
    print("SEC Filing RAG System Ready\n")

    while True:
        company = input("Company (or exit): ")
        if company.lower() == "exit":
            break

        form_type = input("Form Type : ")
        year = input("Year (YYYY): ")
        question = input("Question: ")

        ask_question(company, form_type, year, question)
