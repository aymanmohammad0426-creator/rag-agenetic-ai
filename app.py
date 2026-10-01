from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.graph import graph
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse


app = FastAPI(
    title="Agentic AI RAG Chatbot (Google Gemini)",
    description="Answers questions strictly from the Agentic AI eBook using Google AI API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[str]
    confidence_score: float

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Query the RAG chatbot powered by Google Gemini."""
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise ValueError("Query cannot be empty")
        
        print(f"\n📨 Incoming query: {request.query}")
        
        # Initialize state
        initial_state = {
            "question": request.query,
            "context": [],
            "answer": "",
            "score": 0.0
        }
        
        # Invoke RAG graph
        result = graph.invoke(initial_state)
        
        return QueryResponse(
            question=request.query,
            answer=result["answer"],
            retrieved_chunks=result["context"],
            confidence_score=round(result["score"], 2)
        )
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Agentic AI RAG (Google Gemini)"}

@app.get("/")
async def root():
    """Root endpoint with documentation."""
    return RedirectResponse(url="/docs")
    

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Agentic AI RAG Server (Google Gemini)... on http://127.0.0.1:8000/docs#")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)