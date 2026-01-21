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

AVAILABLE_SPACES = {
    "1": ("SKB", "Syncro Knowledge Base"),
    "2": ("RSKB", "RepairShopr Knowledge Base"),
}


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


def retrieve(query: str, space_name: str = None, k: int = 5) -> list:
    vector_store, client = get_vector_store()
    try:
        if space_name:
            results = vector_store.similarity_search_with_score(
                query,
                k=k,
                pre_filter={"space_name": {"$eq": space_name}}
            )
        else:
            results = vector_store.similarity_search_with_score(query, k=k)
        return results
    finally:
        client.close()


def format_results(results):
    print("\n" + "=" * 60)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n📄 Result {i} (score: {score:.4f})")
        print(f"   Title: {doc.metadata.get('title', 'N/A')}")
        print(f"   Space: {doc.metadata.get('space_name', 'N/A')}")
        print(f"   Content: {doc.page_content[:200]}...")
    print("\n" + "=" * 60)


def load_demo_queries():
    with open(DATABASE_FILE, "r") as f:
        return json.load(f)["demo_queries"]


def select_space():
    print("\n   📁 Select space to filter:")
    for key, (code, name) in AVAILABLE_SPACES.items():
        print(f"      [{key}] {code} ({name})")
    print(f"      [3] No filter (search all)")

    choice = input("   Enter choice (1/2/3): ").strip()
    if choice in AVAILABLE_SPACES:
        return AVAILABLE_SPACES[choice][0]
    return None


def main():
    print("\n" + "=" * 60)
    print("🔍 METADATA-FILTERED RAG - Filter by Space")
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

        space_name = select_space()

        print(f"\n📝 Query: \"{query}\"")
        if space_name:
            print(f"📁 Filter: {space_name}")
        else:
            print("📁 Filter: None (all spaces)")
        print("-" * 40)

        results = retrieve(query, space_name=space_name, k=5)
        format_results(results)

        input("\nPress Enter to continue...")
        print("\n" + "=" * 60)
        print("Select another query or [q] to quit:\n")


if __name__ == "__main__":
    main()
