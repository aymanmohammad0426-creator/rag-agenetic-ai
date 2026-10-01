from hashlib import sha256
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

from .config import (
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    PDF_PATH,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


def run_ingestion():
    """Load and chunk the PDF, embed it, and upsert vectors using Pinecone's SDK."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in .env file")

    print("📄 Loading PDF document...")
    docs = PyPDFLoader(PDF_PATH).load()
    print(f"✅ Loaded {len(docs)} pages from PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1024,
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    print("📍 Embedding and indexing chunks in Pinecone...")
    batch_size = 40
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.page_content for chunk in batch]
        vectors = embeddings.embed_documents(texts)
        records = []
        for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
            metadata = dict(chunk.metadata)
            metadata["text"] = chunk.page_content
            records.append(
                {
                    "id": sha256(
                        f"{metadata.get('source', '')}:{metadata.get('page', '')}:"
                        f"{chunk.page_content}".encode("utf-8")
                    ).hexdigest(),
                    "values": vector,
                    "metadata": metadata,
                }
            )
        index.upsert(vectors=records)
        if start + batch_size < len(chunks):
            print("Waiting 65 seconds for the embedding quota to reset...")
            time.sleep(65)

    print(f"✅ Indexed {len(chunks)} chunks in Pinecone index: {PINECONE_INDEX_NAME}")
    return index


if __name__ == "__main__":
    run_ingestion()
