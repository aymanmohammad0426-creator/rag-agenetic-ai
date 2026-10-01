# Agentic AI RAG Chatbot

A production-ready RAG system that answers questions strictly grounded in the **Agentic AI eBook**.

## Features

- **Document Grounding**: Retrieves context from the eBook; refuses out-of-domain queries
- **LangGraph Workflow**: Stateful retrieval → generation pipeline
- **Pinecone Vector Index**: Semantic search over 1536-dim embeddings
- **FastAPI Backend**: RESTful `/chat` endpoint with confidence scores

## Setup

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure .env with OpenAI + Pinecone keys
cp .env.example .env
# Edit .env with your API keys

# Ingest document (run once)
python -c "from src.ingestion import run_ingestion; run_ingestion()"

# Start server
python app.py
```

## Testing

```bash
python test_queries.py
```

## API Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI?"}'
```

## Architecture

1. **Ingestion**: PDF → Chunks → Embeddings → Pinecone
2. **Retrieval**: Query embedding → Top-5 semantic matches
3. **Generation**: Retrieved context → LLM grounded answer
4. **API**: FastAPI wraps LangGraph graph