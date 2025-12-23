import os
import re
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
    raise RuntimeError("GROQ_API_KEY not set in environment")

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT or "prasanna-portfolio-assistant"

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Prasanna Portfolio Assistant API",
    version="1.0.0"
)

origins = [
    "https://prasanna-chatbot.vercel.app",
    "http://localhost:3000"
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
    status: str

# -----------------------------
# Load Portfolio Data
# -----------------------------
with open("portfolio_data.json", "r", encoding="utf-8") as f:
    portfolio_data = f.read()

# Escape braces for LangChain template safety
portfolio_data = portfolio_data.replace("{", "{{").replace("}", "}}")

# -----------------------------
# Output Sanitizer
# -----------------------------
def clean_output(text: str) -> str:
    if not text:
        return ""

    # Remove markdown & symbols
    text = re.sub(r"[*#_|`~]", "", text)

    # Normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# -----------------------------
# Out-of-Scope Guard
# -----------------------------
OUT_OF_SCOPE_KEYWORDS = [
    "programming",
    "framework",
    "tech stack",
    "machine learning",
    "ai model",
    "research",
    "deep learning"
]

def is_out_of_scope(message: str) -> bool:
    message = message.lower()
    return any(keyword in message for keyword in OUT_OF_SCOPE_KEYWORDS)

# -----------------------------
# LLM Setup (Groq)
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=512,
)

SYSTEM_PROMPT = f"""
You are Prasanna’s AI Portfolio Assistant.

You represent Prasanna accurately and professionally to visitors of his portfolio website.

CORE RESPONSIBILITIES
- Answer questions about Prasanna’s professional background
- Explain his digital marketing skills and experience
- Summarize work experience, tools, and achievements
- Use only the provided portfolio data

COMMUNICATION STYLE
- Clear and professional
- Short paragraphs suitable for chat UI
- Bullet points using hyphens when listing
- No unnecessary verbosity

INFORMATION BOUNDARIES
- Do not invent information
- If something is unavailable, say so clearly
- Do not speak in first person
- Do not provide personal opinions

OUTPUT RULES (STRICT)
- Plain text only
- No markdown or tables
- No symbols like *, #, _, |
- Bullet points only with hyphens
- Line breaks allowed

DEFAULT RESPONSE STRUCTURE
- One-line summary
- Bullet points if applicable
- Avoid long paragraphs

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
    return {"status": "Prasanna Portfolio Assistant running"}

@app.post("/api/assistant", response_model=AssistantResponse)
async def assistant_endpoint(payload: AssistantRequest):
    user_message = payload.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Handle out-of-scope questions
    if is_out_of_scope(user_message):
        return AssistantResponse(
            reply=(
                "Prasanna’s portfolio focuses on digital marketing, campaign execution, "
                "and performance optimization. Technical development or research details "
                "are not part of the available information."
            ),
            status="success"
        )

    try:
        raw_reply = assistant_chain.invoke(
            {"user_message": user_message}
        )

        reply = clean_output(raw_reply)

        if not reply:
            reply = "I do not have enough information to answer that question."

        return AssistantResponse(
            reply=reply,
            status="success"
        )

    except Exception as exc:
        print("Groq error:", exc)
        return AssistantResponse(
            reply="I am unable to respond at the moment. Please try again shortly.",
            status="error"
        )
