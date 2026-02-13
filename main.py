import os
import re
import uuid
import asyncio
from datetime import datetime

import chromadb
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Rate Limiting Imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from google.oauth2 import service_account
from googleapiclient.discovery import build

print(" Starting Royal Gulf Shipping FreightBot server...")

# ---------------------------
# FASTAPI & RATE LIMITER INIT
# ---------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# GOOGLE SHEETS
# ---------------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
CREDS = service_account.Credentials.from_service_account_file(
    "service_account.json",
    scopes=SCOPES
)
sheets_service = build("sheets", "v4", credentials=CREDS)

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "1-Y-YPLdrrz54gqJUpLX7WSlYW_en4nlLPG7AM3DfOZo")
SPREADSHEET_ID_1 = os.getenv("TRACKING_SHEET_ID", "1UT5CAcmOSzvSSJ4viqETIloqVvAgnTGClNfK2xVHFfE")

# ---------------------------
# CHROMA & EMBEDDINGS (Stable 2026)
# ---------------------------
client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database="freightbot"
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

db = Chroma(
    client=client,
    collection_name="freightbot",
    embedding_function=embeddings
)

# ---------------------------
# GEMINI (Stable 2026)
# ---------------------------
llm_config = {"model": "gemini-2.0-flash", "temperature": 0.1}
classifier_llm = ChatGoogleGenerativeAI(**llm_config)
rag_llm = ChatGoogleGenerativeAI(**llm_config)

# ---------------------------
# SESSION MEMORY
# ---------------------------
sessions = {}

# ---------------------------
# HELPERS
# ---------------------------

def mask_pii(text: str) -> str:
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'
    text = re.sub(email_pattern, "[EMAIL_REDACTED]", text)
    text = re.sub(phone_pattern, "[PHONE_REDACTED]", text)
    return text

async def save_chat_to_sheet(session_id, role, message):
    try:
        loop = asyncio.get_event_loop()
        body = {"values": [[datetime.now().strftime("%Y-%m-%d %H:%M"), session_id, role, message]]}
        await loop.run_in_executor(None, lambda: sheets_service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID, range="Sheet1!A:D",
            valueInputOption="RAW", body=body
        ).execute())
    except Exception as e:
        print(f"Logging error: {e}")

async def ask_bot(question, history=[]):
    scrubbed_question = mask_pii(question)
    
    # Similarity Search
    docs = await asyncio.get_event_loop().run_in_executor(
        None, lambda: db.similarity_search_with_score(scrubbed_question, k=4)
    )
    # Loosen filter slightly to ensure cold storage etc are caught
    filtered = [doc for doc, score in docs if score < 0.80]

    if not filtered:
        return "NO_CONTEXT"

    context = "\n".join([d.page_content for d in filtered])
    
    system_instr = (
        "You are the Royal Gulf Shipping Assistant. "
        "Answer ONLY using the provided context. Answer in 1-2 very short, precise sentences. "
        "If the answer is not in the context, say 'I don't have that information'."
    )
    
    secure_human_msg = f"CONTEXT:\n{context}\n\nUSER QUERY: <user_query>{scrubbed_question}</user_query>"
    
    messages = [SystemMessage(content=system_instr)]
    messages.extend(history[-2:]) 
    messages.append(HumanMessage(content=secure_human_msg))

    response = await rag_llm.ainvoke(messages)
    return response.content

async def get_eta_from_sheet(ref_id):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: sheets_service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID_1, range="Orders!A:Q"
    ).execute())
    
    rows = result.get("values", [])
    if not rows: return None

    # Normalizing headers to find index
    headers = [h.strip().lower() for h in rows[0]]
    try:
        ref_idx = headers.index("ref id")
        eta_idx = headers.index("eta")
        
        for row in rows[1:]:
            # EXACT MATCH as per client requirement
            if len(row) > ref_idx and str(row[ref_idx]).strip() == str(ref_id).strip():
                return row[eta_idx] if len(row) > eta_idx else "Pending"
    except ValueError:
        return "System Error: Required columns not found."
    return None

# ---------------------------
# API ROUTES
# ---------------------------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

@app.post("/reset")
async def reset_chat(req: ChatRequest):
    sessions.pop(req.session_id, None)
    return {"status": "success", "message": "Session cleared"}

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(req: ChatRequest, request: Request):
    session_id = req.session_id or str(uuid.uuid4())
    user_msg = req.message.strip()
    
    # This stops malicious or accidental long inputs before they hit the AI
    if len(user_msg) > 200:
        return {
            "session_id": session_id,
            "answer": "⚠️ Your message is too long. Please keep it under 200 characters.",
            "buttons": ["General Queries", "Order Tracking"]
        }

    # If the user clicks the menu button, just return the buttons 
    # WITHOUT calling the ask_bot (Gemini) function.
    if user_msg.lower() == "menu":
        return {
            "session_id": session_id,
            "answer": "Main Menu:", 
            "buttons": ["General Queries", "Order Tracking"]
        }
    # ---------------------------
    # 1. NEW SESSION / GREETING
    if session_id not in sessions:
        sessions[session_id] = {"stage": "ask_name", "history": [], "user_name": None, "flow": None}
        greeting = "Hello 👋 Welcome to Royal Gulf Shipping. May I know your name?"
        return {
            "session_id": session_id, 
            "answer": greeting,
            "buttons": ["📞 Talk to an Agent"]
        }

    session = sessions[session_id]
    await save_chat_to_sheet(session_id, "User", user_msg)

    # 2. HANDLE EXIT
    if any(k in user_msg.lower() for k in ["exit", "end chat", "bye", "✕"]):
        sessions.pop(session_id, None)
        return {"answer": "Thank you for choosing Royal Gulf Shipping! 🚢", "end_chat": True}

    # 3. FLOW: NAME REGISTRATION
    if session["stage"] == "ask_name":
        session["user_name"] = user_msg
        session["stage"] = "choose_flow"
        return {
            "answer": f"Nice to meet you {user_msg}! How can I help you today?",
            "buttons": ["General Queries", "Order Tracking"]
        }

    # 4. HANDLE BUTTON CLICKS DIRECTLY
    if user_msg == "General Queries":
        session["flow"] = "general"
        session["stage"] = "active"
        return {"answer": "Sure! What would you like to know about our services?", "buttons": ["Order Tracking", "📞 Talk to an Agent"]}
    
    if user_msg == "Order Tracking":
        session["flow"] = "tracking"
        session["stage"] = "ask_ref"
        return {"answer": "Please provide your exact Reference ID."}

    # 5. EXECUTION: TRACKING
    if session["flow"] == "tracking" and session["stage"] == "ask_ref":
        
        # --- NEW: Allow user to escape if they click another button instead of typing an ID ---
        if user_msg == "General Queries":
            session["flow"] = "general"
            session["stage"] = "active"
            return {"answer": "Switched to General Queries. How can I help?", "buttons": ["Order Tracking", "📞 Talk to an Agent"]}

        eta = await get_eta_from_sheet(user_msg)
        
        if eta:
            reply = f"📦 Status for {user_msg}: {eta}"
            session["stage"] = "choose_flow" 
            return {"answer": reply, "buttons": ["General Queries", "Order Tracking"]}
        else:
            # --- NEW: Add buttons here so they aren't trapped in the text box ---
            return {
                "answer": "❌ Ref ID not found exactly as entered. Please try again or choose another option:",
                "buttons": ["General Queries", "Order Tracking", "📞 Talk to an Agent"]
            }

    # 6. EXECUTION: RAG (General Queries)
    answer = await ask_bot(user_msg, session["history"])
    
    if answer == "NO_CONTEXT":
        answer = "I couldn't find specific details in our manual. Would you like to speak to a human agent?"
    
    # Manual short-cut for extra safety
    if len(answer.split('.')) > 2:
        answer = ". ".join(answer.split('.')[:2]) + "."

    session["history"].append(HumanMessage(content=user_msg))
    await save_chat_to_sheet(session_id, "Bot", answer)
    
    return {
        "session_id": session_id, 
        "answer": answer, 
        "buttons": ["General Queries", "Order Tracking", "📞 Talk to an Agent"]
    }
