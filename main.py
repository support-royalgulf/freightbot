# -*- coding: utf-8 -*-

import os
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

print("🚀 Starting FreightBot server...")

# --------------------------------
# ENV CHECK
# --------------------------------

REQUIRED_VARS = ["GOOGLE_API_KEY", "CHROMA_API_KEY", "CHROMA_TENANT"]

for v in REQUIRED_VARS:
    if v not in os.environ:
        raise ValueError(f"Missing environment variable: {v}")

print("✅ Environment variables loaded")

# --------------------------------
# FASTAPI INIT
# --------------------------------

app = FastAPI(title="FreightBot RAG Chatbot")

# Enable browser access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Replace with your domain later
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# CONNECT TO CHROMA CLOUD
# --------------------------------

print("🔌 Connecting to Chroma Cloud...")

client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database="freightbot"
)

print("✅ Chroma Cloud connected")

# --------------------------------
# EMBEDDINGS
# --------------------------------

print("🧠 Loading Gemini embeddings...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# --------------------------------
# VECTOR DATABASE
# --------------------------------

print("📦 Loading vector database collection...")

db = Chroma(
    client=client,
    collection_name="freightbot",
    embedding_function=embeddings
)

print("✅ Vector database ready")

# --------------------------------
# GEMINI CHAT MODEL
# --------------------------------

print("🤖 Loading Gemini chat model...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

print("✅ Gemini model loaded")

# --------------------------------
# CHAT MEMORY
# --------------------------------

chat_history = []

# --------------------------------
# QUERY CLEANER (SPELL FIX)
# --------------------------------

def improve_query(user_question: str) -> str:
    prompt = f"""
Fix spelling mistakes and rewrite this question clearly:

{user_question}
"""
    result = llm.invoke(prompt)
    return result.content

# --------------------------------
# MAIN CHAT FUNCTION
# --------------------------------

def ask_bot(question: str) -> str:

    print("🔍 User Question:", question)

    better_question = improve_query(question)

    print("✨ Improved Query:", better_question)

    docs_with_score = db.similarity_search_with_score(
        better_question,
        k=8
    )

    filtered_docs = [doc for doc, score in docs_with_score if score < 0.8]

    print("📄 Retrieved chunks:", len(filtered_docs))

    if len(filtered_docs) == 0:
        return "I don't have that information in my knowledge base."

    context = "\n\n".join([d.page_content for d in filtered_docs])

    prompt = f"""
You are a professional shipping company assistant.

Rules:
- Use ONLY provided context
- If answer not found say: I don't have that information
- Be short and clear

Conversation History:
{chat_history}

Context:
{context}

User Question:
{question}
"""

    response = llm.invoke(prompt)

    chat_history.append(f"User: {question}")
    chat_history.append(f"Bot: {response.content}")

    return response.content


# --------------------------------
# API MODELS
# --------------------------------

class Query(BaseModel):
    question: str


# --------------------------------
# API ROUTES
# --------------------------------

@app.get("/")
def root():
    return {
        "status": "FreightBot is running",
        "vector_db": "connected",
        "model": "Gemini"
    }


@app.post("/chat")
def chat_endpoint(data: Query):

    print("📩 Incoming chat request")

    answer = ask_bot(data.question)

    print("✅ Response sent")

    return {"answer": answer}


# --------------------------------
# LOCAL DEV ONLY
# --------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
