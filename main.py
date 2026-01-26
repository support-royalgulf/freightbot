# -*- coding: utf-8 -*-
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------------------
# CONFIG
# ----------------------------
# Gemini API key from environment variable
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("Set GOOGLE_API_KEY in environment variables!")

GEMINI_KEY = os.environ["GOOGLE_API_KEY"]

PDF_FOLDER = "./docs"          # Place your PDFs here
VECTOR_DB_DIR = "./drive_rag" # Persisted vector database

# ----------------------------
# LOAD PDF DOCUMENTS
# ----------------------------
docs = []
for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
        docs.extend(loader.load())

print(f"Loaded {len(docs)} PDF pages.")

# ----------------------------
# SPLIT DOCUMENTS INTO CHUNKS
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

# ----------------------------
# EMBEDDINGS & VECTOR DB
# ----------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DB_DIR
)

print("Vector DB created")

# ----------------------------
# GEMINI CHAT MODEL
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def improve_query(user_question: str) -> str:
    """Fix typos / rewrite query for better retrieval"""
    prompt = f"""
Fix spelling mistakes and rewrite this question clearly:

{user_question}
"""
    result = llm.invoke(prompt)
    return result.content

chat_history = []

def ask_bot(question: str) -> str:
    global chat_history

    better_question = improve_query(question)

    docs_with_score = db.similarity_search_with_score(better_question, k=8)
    filtered_docs = [doc for doc, score in docs_with_score if score < 0.7]

    if len(filtered_docs) == 0:
        return "I don't have that information in my knowledge base."

    context = "\n\n".join([d.page_content for d in filtered_docs])

    prompt = f"""
You are a professional assistant.

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

# ----------------------------
# FASTAPI SERVER
# ----------------------------
app = FastAPI(title="FreightBot RAG Chatbot")

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat_endpoint(data: Query):
    answer = ask_bot(data.question)
    return {"answer": answer}

# ----------------------------
# RUN SERVER (for local dev)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
