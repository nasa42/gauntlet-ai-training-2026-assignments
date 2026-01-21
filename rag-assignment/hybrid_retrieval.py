import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DATABASE_FILE = Path(__file__).parent / "database.json"

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if os.getenv("LANGSMITH_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
    print(f"📊 LangSmith tracing enabled → {os.getenv('LANGCHAIN_PROJECT', 'default')}")

from langchain_openai import AzureOpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import certifi

MONGO_URI = os.getenv("MONGO_URI")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

DB_NAME = "syncro_rag"
COLLECTION_NAME = "confluence_docs"
INDEX_NAME = "vector_index"


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version="2024-02-01"
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
                "keyword_matches": keyword_matches,
            })

        scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored_results[:k]
    finally:
        client.close()


def format_results(results, query_terms):
    print("\n" + "=" * 60)
    for i, r in enumerate(results, 1):
        doc = r["doc"]
        print(f"\n📄 Result {i}")
        print(f"   Combined: {r['combined_score']:.4f} | Vector: {r['vector_score']:.4f} | Keyword: {r['keyword_score']:.4f}")
        print(f"   Keywords matched: {r['keyword_matches']}/{len(query_terms)}")
        print(f"   Title: {doc.metadata.get('title', 'N/A')}")
        print(f"   Space: {doc.metadata.get('space_name', 'N/A')}")
        print(f"   Content: {doc.page_content[:200]}...")
    print("\n" + "=" * 60)


def load_demo_queries():
    with open(DATABASE_FILE, "r") as f:
        return json.load(f)["demo_queries"]


def select_alpha():
    print("\n   ⚖️  Select weighting (alpha):")
    print("      [1] 0.3 (30% vector, 70% keyword)")
    print("      [2] 0.5 (50% vector, 50% keyword) - balanced")
    print("      [3] 0.7 (70% vector, 30% keyword)")
    print("      [4] Custom value")

    choice = input("   Enter choice (1-4): ").strip()
    if choice == "1":
        return 0.3
    elif choice == "2":
        return 0.5
    elif choice == "3":
        return 0.7
    elif choice == "4":
        try:
            val = float(input("   Enter alpha (0.0-1.0): ").strip())
            return max(0.0, min(1.0, val))
        except ValueError:
            return 0.5
    return 0.5


def main():
    print("\n" + "=" * 60)
    print("🔍 HYBRID RAG - Vector + Keyword Search")
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
            query = input("\nEnter your query: ").strip()
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(demo_queries):
                    query = demo_queries[idx - 1]
                else:
                    print("❌ Invalid choice")
                    continue
            except ValueError:
                print("❌ Invalid input")
                continue

        alpha = select_alpha()
        query_terms = set(query.lower().split())

        print(f"\n📝 Query: \"{query}\"")
        print(f"⚖️  Alpha: {alpha} ({int(alpha*100)}% vector, {int((1-alpha)*100)}% keyword)")
        print(f"🔑 Keywords: {', '.join(query_terms)}")
        print("-" * 40)

        results = retrieve(query, k=5, alpha=alpha)
        format_results(results, query_terms)

        input("\nPress Enter to continue...")
        print("\n" + "=" * 60)
        print("Select another query or [q] to quit:\n")


if __name__ == "__main__":
    main()
