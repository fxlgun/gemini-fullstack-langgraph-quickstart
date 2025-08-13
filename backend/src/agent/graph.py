import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from typing import List, Any
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict


class OverallState(TypedDict):
    messages: Annotated[list, add_messages] | None  # Conversation history
    rag_query: str | None                           # Query string for RAG
    rag_docs: List[Any] | None   

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

CHROMA_DB_FOLDER = os.getenv("CHROMA_DB_FOLDER")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")

gemini_rag_prompt = """
You are an AI assistant with access to retrieved knowledge.

Context:
{context}

Question:
{query}

Answer clearly and concisely using only the given context when possible.
If the context does not have the answer, say you don't know.
"""


# 1️⃣ Retrieve docs
def retrieve_rag_docs(state: OverallState, config: RunnableConfig) -> dict:
    query = state["messages"][-1]["content"]

    client = chromadb.PersistentClient(path=CHROMA_DB_FOLDER)
    collection = client.get_collection(COLLECTION_NAME)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    q_emb = embed_model.encode([query]).tolist()[0]

    results = collection.query(query_embeddings=[q_emb], n_results=10)
    docs = results["documents"][0]

    return { 
        "messages": state["messages"],
        "rag_query": query,
        "rag_docs": docs,
    }


# 2️⃣ Generate answer with Gemini
def answer_with_rag(state: OverallState, config: RunnableConfig) -> dict:
    query = state["rag_query"]
    docs = state["rag_docs"]
    context = "\n\n---\n\n".join(docs)

    formatted_prompt = gemini_rag_prompt.format(context=context, query=query)


    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    result = llm.invoke(formatted_prompt)

    return {
        "messages": state["messages"] + [AIMessage(content=result.content)],
        "rag_docs": docs
    }

# Build graph
builder = StateGraph(OverallState)
builder.add_node("retrieve_rag_docs", retrieve_rag_docs)
builder.add_node("answer_with_rag", answer_with_rag)

builder.add_edge(START, "retrieve_rag_docs")
builder.add_edge("retrieve_rag_docs", "answer_with_rag")
builder.add_edge("answer_with_rag", END)

graph = builder.compile(name="land-rag-agent")

