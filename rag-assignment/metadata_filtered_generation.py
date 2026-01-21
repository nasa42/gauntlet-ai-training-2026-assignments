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

AVAILABLE_SPACES = {
    "1": ("SKB", "Syncro Knowledge Base"),
    "2": ("RSKB", "RepairShopr Knowledge Base"),
}

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


def retrieve(query: str, space_name: str = None, k: int = 5) -> list:
    vector_store, client = get_vector_store()
    try:
        if space_name:
            results = vector_store.similarity_search(
                query,
                k=k,
                pre_filter={"space_name": {"$eq": space_name}}
            )
        else:
            results = vector_store.similarity_search(query, k=k)
        return results
    finally:
        client.close()


def format_context(documents: list) -> str:
    context_parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Unknown")
        space = doc.metadata.get("space_name", "Unknown")
        context_parts.append(
            f"[Document {i}: {title} ({space})]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def select_space():
    print("\n   📁 Select space to filter:")
    for key, (code, name) in AVAILABLE_SPACES.items():
        print(f"      [{key}] {code} ({name})")
    print(f"      [3] No filter (search all)")

    choice = input("   Enter choice (1/2/3): ").strip()
    if choice in AVAILABLE_SPACES:
        return AVAILABLE_SPACES[choice][0]
    return None


def generate_answer(question: str, space_name: str = None, k: int = 5) -> dict:
    if space_name:
        print(f"🔍 Retrieving documents (Metadata-filtered: {space_name})...")
    else:
        print("🔍 Retrieving documents (No filter)...")

    documents = retrieve(question, space_name=space_name, k=k)

    if not documents:
        return {"answer": "I couldn't find any relevant information.", "sources": [], "context": ""}

    print(f"   Found {len(documents)} documents")

    context = format_context(documents)

    print("🤖 Generating answer...")
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "space_filter": space_name,
        "sources": [
            {"title": doc.metadata.get("title", "Unknown"), "space": doc.metadata.get("space_name", "Unknown")}
            for doc in documents
        ]
    }


def save_result(result: dict):
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "rag_pattern": "metadata_filtered",
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
    print("🤖 METADATA-FILTERED RAG GENERATION")
    print("   Filter by space → Vector search → LLM")
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

        space_name = select_space()

        print(f"\n📝 Question: \"{question}\"")
        if space_name:
            print(f"📁 Filter: {space_name}")
        print("-" * 60)

        try:
            result = generate_answer(question, space_name=space_name, k=5)

            print("\n" + "=" * 60)
            print("💬 ANSWER:")
            print("=" * 60)
            print(result["answer"])

            print("\n📚 Sources:")
            for src in result["sources"]:
                print(f"   • {src['title']} ({src['space']})")

            save_result(result)

        except Exception as e:
            print(f"\n❌ Error: {e}")

        input("\nPress Enter to continue...")
        print()


if __name__ == "__main__":
    main()
