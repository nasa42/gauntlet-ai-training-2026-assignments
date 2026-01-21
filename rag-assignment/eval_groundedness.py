"""
Groundedness Evaluation
=======================
Evaluates whether LLM answers are grounded in the retrieved context.
Loads from results.json (saved by generation scripts).
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")

RESULTS_FILE = Path(__file__).parent / "result.json"

GROUNDEDNESS_PROMPT = """You are a groundedness evaluator. Determine if the answer is supported by the context.

GROUNDED = Every claim in the answer can be traced to the context
NOT_GROUNDED = Answer contains information not in the context (hallucination)

Context:
{context}

Question: {question}

Answer:
{answer}

Respond in this exact format:
REASONING: <brief explanation>
VERDICT: <GROUNDED or NOT_GROUNDED>"""


def get_judge_llm():
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_CHAT_DEPLOYMENT,
        api_version="2024-02-01",
        temperature=0.0,
    )


def evaluate_groundedness(question: str, context: str, answer: str) -> dict:
    prompt = ChatPromptTemplate.from_template(GROUNDEDNESS_PROMPT)
    llm = get_judge_llm()
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": question,
        "answer": answer,
    })

    reasoning = ""
    verdict = "UNKNOWN"

    for line in response.strip().split("\n"):
        if line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
        elif line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()

    return {
        "is_grounded": verdict == "GROUNDED",
        "verdict": verdict,
        "reasoning": reasoning,
    }


def main():
    print("\n" + "=" * 60)
    print("🔬 GROUNDEDNESS EVALUATION")
    print("=" * 60)

    if not RESULTS_FILE.exists():
        print(f"\n❌ No result.json found")
        print("   Run a generation script first.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    print(f"\n📁 Loaded: {RESULTS_FILE}")
    print(f"   Pattern: {data.get('rag_pattern', 'unknown')}")
    print(f"   Question: {data.get('question', 'N/A')[:50]}...")

    question = data.get("question", "")
    context = data.get("context", "")
    answer = data.get("answer", "")

    if not context or not answer:
        print("\n❌ Missing context or answer in result.json")
        return

    print("\n🔍 Evaluating groundedness...")
    result = evaluate_groundedness(question, context, answer)

    print("\n" + "=" * 60)
    print("📊 RESULT")
    print("=" * 60)

    if result["is_grounded"]:
        print("   ✅ GROUNDED")
    else:
        print("   ❌ NOT GROUNDED (hallucination detected)")

    print(f"\n   Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    main()
