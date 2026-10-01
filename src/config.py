import os
from dotenv import load_dotenv

load_dotenv()

# Google AI API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# Document paths
PDF_PATH = "data/Ebook-Agentic-AI.pdf"

# Model configurations
EMBEDDING_MODEL = "models/gemini-embedding-001"  # Google's embedding model
LLM_MODEL = "gemini-3.8-flash"  # Or gemini-pro for faster responses