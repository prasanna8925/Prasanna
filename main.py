import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env")

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT or "portfolio-assistant"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Vigneshwaran Portfolio Assistant API")

origins = [
    "https://vigneshwarancj-portfolio-website.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# API Models
# -----------------------------
class AssistantRequest(BaseModel):
    message: str

class AssistantResponse(BaseModel):
    reply: str

# -----------------------------
# Load portfolio data
# -----------------------------
with open("portfolio_data.json", "r", encoding="utf-8") as f:
    portfolio_data = f.read()

# Escape braces to avoid LC template conflicts
portfolio_data = portfolio_data.replace("{", "{{").replace("}", "}}")

# -----------------------------
# Output sanitizer (safety net)
# -----------------------------
def clean_output(text: str) -> str:
    forbidden_tokens = ["**", "*", "|", "#", "__", "~~"]
    for token in forbidden_tokens:
        text = text.replace(token, "")
    return text.strip()

# -----------------------------
# LLM Setup (Groq)
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=1024,
)

SYSTEM_PROMPT = f"""
You are Vigneshwaran CJ’s AI Portfolio Assistant.

Your role is to represent Vigneshwaran CJ accurately, professionally, and clearly to visitors of his portfolio website.

====================
CORE RESPONSIBILITIES
====================
- Answer questions about skills, projects, research, experience, and tools
- Explain technical topics clearly and concisely
- Adjust depth based on the user’s question
- Only use information present in the portfolio data

====================
COMMUNICATION STYLE
====================
- Professional and factual
- Clear and structured
- Concise, without unnecessary verbosity
- Neutral and accurate

====================
INFORMATION BOUNDARIES
====================
- Do not invent or assume information
- If information is unavailable, state that clearly
- Do not provide personal opinions or speculation
- Do not impersonate Vigneshwaran CJ in first person

====================
OUTPUT FORMATTING RULES (STRICT)
====================
- Use plain text only
- Do not use tables
- Do not use markdown formatting
- Do not use bold, italics, headings, or symbols such as *, **, #, |, _
- Bullet points are allowed only using hyphens (-)
- Use line breaks for readability
- Keep responses suitable for a chat UI

Allowed example:
Skills overview:
- Programming languages: Python, JavaScript
- Backend frameworks: FastAPI, Node.js

Disallowed:
- Tables
- Markdown formatting
- Emphasis symbols

====================
DEFAULT RESPONSE STRATEGY
====================
- Start with a short, direct summary
- Follow with clean bullet points if listing items
- Avoid long paragraphs unless explicitly requested

Portfolio data:
{portfolio_data}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{user_message}")
    ]
)

parser = StrOutputParser()
assistant_chain = prompt | llm | parser

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "Groq Portfolio Assistant running"}

@app.post("/api/assistant", response_model=AssistantResponse)
async def assistant_endpoint(payload: AssistantRequest):
    user_message = payload.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        raw_reply = assistant_chain.invoke(
            {"user_message": user_message}
        )

        reply = clean_output(raw_reply)

        return AssistantResponse(reply=reply)

    except Exception as exc:
        print("Groq error:", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate assistant response"
        )
