import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

RESULTS_FILE = Path(__file__).parent / "result.json"
DATABASE_FILE = Path(__file__).parent / "database.json"

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
    print(f"📊 LangSmith tracing enabled → {os.getenv('LANGCHAIN_PROJECT', 'default')}")

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
import certifi

MONGO_URI = os.getenv("MONGO_URI")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")

DB_NAME = "syncro_rag"
COLLECTION_NAME = "confluence_docs"
INDEX_NAME = "vector_index"

RAG_PROMPT = """You are a helpful assistant that answers questions about Syncro and RepairShopr products based on the provided documentation.

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version="2024-02-01"
    )


def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_CHAT_DEPLOYMENT,
        api_version="2024-02-01",
        temperature=0
    )


def get_vector_store():
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=get_embeddings(),
        index_name=INDEX_NAME
    ), client


def retrieve(query: str, k: int = 5, alpha: float = 0.5) -> list:
    vector_store, client = get_vector_store()
    try:
        vector_results = vector_store.similarity_search_with_score(query, k=k*2)

        query_terms = set(query.lower().split())

        scored_results = []
        for doc, vector_score in vector_results:
            content_lower = doc.page_content.lower()
            keyword_matches = sum(1 for term in query_terms if term in content_lower)
            keyword_score = keyword_matches / len(query_terms) if query_terms else 0

            combined_score = (alpha * (1 - vector_score)) + ((1 - alpha) * keyword_score)
            scored_results.append({
                "doc": doc,
                "combined_score": combined_score,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
            })

        scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_results[:k]
    finally:
        client.close()


def format_context(results: list) -> str:
    context_parts = []
    for i, r in enumerate(results, 1):
        doc = r["doc"]
        title = doc.metadata.get("title", "Unknown")
        space = doc.metadata.get("space_name", "Unknown")
        context_parts.append(
            f"[Document {i}: {title} ({space})]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def select_alpha():
    print("\n   ⚖️  Select weighting (alpha):")
    print("      [1] 0.3 (30% vector, 70% keyword)")
    print("      [2] 0.5 (50% vector, 50% keyword)")
    print("      [3] 0.7 (70% vector, 30% keyword)")

    choice = input("   Enter choice (1-3): ").strip()
    if choice == "1":
        return 0.3
    elif choice == "2":
        return 0.5
    elif choice == "3":
        return 0.7
    return 0.5


def generate_answer(question: str, alpha: float = 0.5, k: int = 5) -> dict:
    print(f"🔍 Retrieving documents (Hybrid: {int(alpha*100)}% vector, {int((1-alpha)*100)}% keyword)...")

    results = retrieve(question, k=k, alpha=alpha)

    if not results:
        return {"answer": "I couldn't find any relevant information.", "sources": [], "scores": [], "context": ""}

    print(f"   Found {len(results)} documents")

    context = format_context(results)

    print("🤖 Generating answer...")
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "alpha": alpha,
        "sources": [
            {"title": r["doc"].metadata.get("title", "Unknown"), "space": r["doc"].metadata.get("space_name", "Unknown")}
            for r in results
        ],
        "scores": [
            {"combined": r["combined_score"], "vector": r["vector_score"], "keyword": r["keyword_score"]}
            for r in results
        ]
    }


def save_result(result: dict):
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "rag_pattern": "hybrid",
            "model": AZURE_CHAT_DEPLOYMENT,
            "timestamp": datetime.now().isoformat(),
            **result
        }, f, indent=2)
    print(f"💾 Saved to {RESULTS_FILE}")


def load_demo_queries():
    with open(DATABASE_FILE, "r") as f:
        return json.load(f)["demo_queries"]


def main():
    print("\n" + "=" * 60)
    print("🤖 HYBRID RAG GENERATION")
    print("   Vector + Keyword scoring → LLM")
    print("=" * 60)

    demo_queries = load_demo_queries()
    print("\nSelect a query:\n")

    for i, q in enumerate(demo_queries, 1):
        print(f"  [{i}] {q}")

    print(f"\n  [0] Enter custom query")
    print(f"  [q] Quit\n")

    while True:
        choice = input("Enter choice: ").strip().lower()

        if choice == "q":
            print("\n👋 Goodbye!\n")
            break

        if choice == "0":
            question = input("\nEnter your question: ").strip()
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(demo_queries):
                    question = demo_queries[idx - 1]
                else:
                    print("❌ Invalid choice")
                    continue
            except ValueError:
                print("❌ Invalid input")
                continue

        alpha = select_alpha()

        print(f"\n📝 Question: \"{question}\"")
        print(f"⚖️  Alpha: {alpha}")
        print("-" * 60)

        try:
            result = generate_answer(question, alpha=alpha, k=5)

            print("\n" + "=" * 60)
            print("💬 ANSWER:")
            print("=" * 60)
            print(result["answer"])

            print("\n📚 Sources (with scores):")
            for src, score in zip(result["sources"], result["scores"]):
                print(f"   • {src['title']} ({src['space']})")
                print(f"     Combined: {score['combined']:.3f} | Vector: {score['vector']:.3f} | Keyword: {score['keyword']:.3f}")

            save_result(result)

        except Exception as e:
            print(f"\n❌ Error: {e}")

        input("\nPress Enter to continue...")
        print()


if __name__ == "__main__":
    main()
