from typing import List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone

from .config import (
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    LLM_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def build_rag_graph():
    """Build the RAG workflow with direct Pinecone SDK search."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in .env file")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1024,
    )
    index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        convert_system_message_to_human=True,
    )

    def retrieve_node(state: AgentState):
        print(f"🔍 Retrieving context for: '{state['question']}'")
        query_vector = embeddings.embed_query(state["question"])
        result = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
        )
        context_texts = [
            match.metadata.get("text", "")
            for match in result.matches
            if match.metadata and match.metadata.get("text")
        ]
        print(f"✅ Retrieved {len(context_texts)} context chunks")
        return {"context": context_texts}

    def generate_node(state: AgentState):
        context_str = "\n\n---\n\n".join(state["context"])
        system_prompt = """Answer using only the retrieved excerpts from the eBook.

- Combine relevant details across excerpts to answer the question.
- If the excerpts support only a partial answer, give that answer and say what they do not establish.
- Do not treat a table of contents heading as a definition.
- Do not add facts from outside the excerpts.
- Cite the supporting excerpt or section in your answer.
- If no excerpt contains relevant information, say: "I cannot answer this question based on the retrieved passages."""
        user_prompt = f"""Context from the Agentic AI eBook:
{context_str}

Question: {state['question']}

Answer based ONLY on the context above:"""
        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            content = response.content
            if isinstance(content, str):
                answer_text = content
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
                answer_text = "\n".join(text_parts).strip()
                if not answer_text:
                    answer_text = str(content)
            else:
                answer_text = str(content)
        except Exception as exc:
            print(f"❌ Error calling Gemini: {exc}")
            answer_text = f"Error generating response: {exc}"

        confidence = min(0.95, 0.7 + len(state["context"]) * 0.05)
        print(f"✅ Answer generated | Confidence: {confidence}")
        return {"answer": answer_text, "score": confidence}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


print("🔧 Initializing RAG graph with Google Gemini and Pinecone...")
graph = build_rag_graph()
print("✅ Graph initialized successfully")
