import os
import re
import certifi
from pathlib import Path
from html import unescape

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from pymongo import MongoClient

load_dotenv(Path(__file__).parent / ".env")

if os.getenv("LANGCHAIN_API_KEY"):
    print(f"📊 LangSmith tracing enabled → {os.getenv('LANGCHAIN_PROJECT', 'default')}")

MONGO_URI = os.getenv("MONGO_URI")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

DB_NAME = "syncro_rag"
COLLECTION_NAME = "confluence_docs"
INDEX_NAME = "vector_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

DOCS_DIR = Path(__file__).parent / "fetched-confluence-docs"


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown file."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    metadata = {}
    for line in frontmatter_text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            value = value.strip().strip('"').strip("'")
            metadata[key.strip()] = value

    return metadata, body


def clean_html(html_content: str) -> str:
    """Convert HTML to clean text."""
    text = re.sub(r'<ac:image[^>]*>.*?</ac:image>', '[IMAGE]', html_content, flags=re.DOTALL)
    text = re.sub(r'<ri:attachment[^>]*/?>', '', text)
    text = re.sub(r'<ac:[^>]*/?>', '', text)
    text = re.sub(r'</ac:[^>]*>', '', text)

    text = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n\1\n\n', text, flags=re.DOTALL)
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)

    text = unescape(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def load_markdown_files() -> list[Document]:
    """Load all markdown files and convert to LangChain Documents."""
    documents = []

    for md_file in DOCS_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(content)

        body = re.sub(r'^#\s+.*\n', '', body)
        clean_text = clean_html(body)

        if len(clean_text) < 50:
            continue

        space_name = md_file.parent.name
        metadata["space_name"] = space_name
        metadata["source_file"] = md_file.name

        doc = Document(page_content=clean_text, metadata=metadata)
        documents.append(doc)

    print(f"Loaded {len(documents)} documents with content")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(doc_chunks)
        chunks.extend(doc_chunks)

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def setup_mongodb():
    """Connect to MongoDB and prepare collection."""
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client[DB_NAME]

    if COLLECTION_NAME in db.list_collection_names():
        db[COLLECTION_NAME].delete_many({})
        print(f"Cleared existing documents in {COLLECTION_NAME}")
    else:
        db.create_collection(COLLECTION_NAME)
        print(f"Created collection {COLLECTION_NAME}")

    return client, db[COLLECTION_NAME]


def store_embeddings(collection, chunks: list[Document]):
    """Create embeddings and store in MongoDB."""
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version="2024-02-01"
    )

    print(f"Creating embeddings for {len(chunks)} chunks...")

    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )

    print("Embeddings stored successfully")
    return vector_store


def print_index_instructions():
    """Print MongoDB Atlas Vector Search index setup instructions."""
    print("\n" + "=" * 60)
    print("CREATE VECTOR SEARCH INDEX IN MONGODB ATLAS")
    print("=" * 60)
    print(f"""
1. Go to MongoDB Atlas → Your Cluster → Atlas Search
2. Click "Create Search Index" → "JSON Editor"
3. Select database: {DB_NAME}
4. Select collection: {COLLECTION_NAME}
5. Index name: {INDEX_NAME}
6. Paste this JSON:

{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }},
    {{
      "type": "filter",
      "path": "space_id"
    }},
    {{
      "type": "filter",
      "path": "space_name"
    }}
  ]
}}

Wait for index status to become "Active" before querying.
""")
    print("=" * 60)


def main():
    print("=" * 50)
    print("RAG Pipeline - Document Ingestion")
    print("=" * 50)

    if not MONGO_URI:
        raise ValueError("Set MONGO_URI in .env")
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")

    documents = load_markdown_files()
    if not documents:
        raise ValueError("No documents found")

    chunks = chunk_documents(documents)

    client, collection = setup_mongodb()

    try:
        store_embeddings(collection, chunks)
        print(f"\n✅ Ingestion complete!")
        print(f"   Database: {DB_NAME}")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Chunks stored: {len(chunks)}")
        print_index_instructions()
    finally:
        client.close()


if __name__ == "__main__":
    main()
