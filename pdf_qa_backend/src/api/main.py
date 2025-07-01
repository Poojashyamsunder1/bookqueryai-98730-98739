"""
FastAPI backend for PDF QA: allows users to upload PDF files, ask questions about them using an LLM (OpenAI), and retrieve previous Q&A history.
Features:
- POST /upload_pdf: Upload PDF, extract text using pdfplumber, and store in database.
- POST /ask_question: Ask a question regarding the latest uploaded PDF, get answer from OpenAI API, save both question and answer in database.
- GET /history: Retrieve the history of question/answer pairs.
Dependencies:
- pdf_qa_database: Database storage/retrieval for PDF content and Q&A history.
- pdfplumber: PDF text extraction.
- OpenAI API: Retrieves answers about the PDF.
Configuration:
- Set environment variables in .env as required: OPENAI_API_KEY, DATABASE_URL, etc.
- Do NOT hardcode credentials.
"""

import os
from typing import List, Optional
from fastapi import (
    FastAPI, 
    UploadFile, 
    File,
    HTTPException, 
    status, 
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pdfplumber
import io
import uuid

from dotenv import load_dotenv

import httpx

# Load env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
DB_URL = os.getenv("DATABASE_URL")  # Provided by 'pdf_qa_database' dependency

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in environment.")

if not DB_URL:
    raise RuntimeError("DATABASE_URL must be set in environment for pdf_qa_database.")

# ------------- Models -------------

class UploadPdfResponse(BaseModel):
    message: str = Field(..., description="Outcome message of upload action")
    pdf_id: str = Field(..., description="Identifier of the uploaded PDF")

class AskQuestionRequest(BaseModel):
    pdf_id: str = Field(..., description="ID of the uploaded PDF to query against")
    question: str = Field(..., description="The user's question about the PDF")

class AskQuestionResponse(BaseModel):
    pdf_id: str
    question: str
    answer: str

class HistoryItem(BaseModel):
    pdf_id: str
    question: str
    answer: str

class HistoryResponse(BaseModel):
    history: List[HistoryItem]

# ------------- Database Interface Placeholder ----------------------

class Database:
    """
    Minimal stub for database interaction.
    Replace with your preferred DB client (e.g., SQLAlchemy) and schema.
    The following schema is assumed:
    - pdfs: pdf_id (str, PK), content (text)
    - history: id (int, PK), pdf_id (str, FK), question (str), answer (str)
    """
    def __init__(self, db_url: str):
        import sqlite3  # For demo purposes; swap for production DB.
        self.conn = sqlite3.connect(db_url.replace("sqlite:///", ""), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS pdfs (
            pdf_id TEXT PRIMARY KEY,
            content TEXT NOT NULL
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            FOREIGN KEY(pdf_id) REFERENCES pdfs(pdf_id)
        )""")
        self.conn.commit()

    # PUBLIC_INTERFACE
    def save_pdf(self, pdf_id: str, content: str):
        """Save PDF text content to DB."""
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO pdfs (pdf_id, content) VALUES (?, ?)", (pdf_id, content))
        self.conn.commit()

    # PUBLIC_INTERFACE
    def get_pdf_content(self, pdf_id: str) -> Optional[str]:
        """Retrieve PDF text by id."""
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM pdfs WHERE pdf_id=?", (pdf_id,))
        row = cur.fetchone()
        return row[0] if row else None

    # PUBLIC_INTERFACE
    def save_qa(self, pdf_id: str, question: str, answer: str):
        """Save a Q&A pair for a PDF."""
        cur = self.conn.cursor()
        cur.execute("INSERT INTO history (pdf_id, question, answer) VALUES (?, ?, ?)", (pdf_id, question, answer))
        self.conn.commit()

    # PUBLIC_INTERFACE
    def get_history(self, pdf_id: Optional[str] = None) -> List[HistoryItem]:
        """Return previous Q&A pairs, optionally filtered by PDF id."""
        cur = self.conn.cursor()
        if pdf_id:
            cur.execute("SELECT pdf_id, question, answer FROM history WHERE pdf_id=? ORDER BY id DESC", (pdf_id,))
        else:
            cur.execute("SELECT pdf_id, question, answer FROM history ORDER BY id DESC")
        rows = cur.fetchall()
        return [HistoryItem(pdf_id=row[0], question=row[1], answer=row[2]) for row in rows]

db = Database(DB_URL)

# ------------- OpenAI Integration ----------------------

# PUBLIC_INTERFACE
async def openai_answer(question: str, context: str) -> str:
    """
    Query OpenAI API to answer 'question' given 'context' (PDF text).
    Returns the model's answer.
    """
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": "You are an assistant who answers questions based only on the provided context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(OPENAI_API_URL, json=payload, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"OpenAI API error: {response.text}"
            )
        completions = response.json()
        return completions['choices'][0]['message']['content'].strip()

# ------------- FastAPI App and Routes ---------------

app = FastAPI(
    title="PDF Question Answering API",
    version="1.0.0",
    description="Upload a PDF, ask questions about its contents, and retrieve Q&A history.",
    openapi_tags=[
        {"name": "PDF", "description": "Endpoints for PDF upload and management"},
        {"name": "QuestionAnswer", "description": "Endpoints for question answering"},
        {"name": "History", "description": "Endpoints for retrieving Q&A history"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint for server monitoring and liveness probes."""
    return {"status": "ok", "message": "Healthy"}

# PUBLIC_INTERFACE
@app.post("/upload_pdf", response_model=UploadPdfResponse, tags=["PDF"], summary="Upload a PDF file and extract its text.", description="Upload a PDF file to the backend. Text extracted and stored by PDF ID.")
async def upload_pdf(file: UploadFile = File(...)) -> UploadPdfResponse:
    """
    Accepts a PDF file upload, extracts text using pdfplumber, stores in database, and returns PDF ID.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="The uploaded file must be a PDF.")

    pdf_bytes = await file.read()
    pdf_id = str(uuid.uuid4())  # Generate unique PDF ID

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Simple extraction: concatenate text from all pages
            all_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            if not all_text.strip():
                raise ValueError("No extractable text found in PDF.")
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {ex}")

    db.save_pdf(pdf_id, all_text)
    return UploadPdfResponse(message="PDF uploaded and text extracted successfully", pdf_id=pdf_id)

# PUBLIC_INTERFACE
@app.post("/ask_question", response_model=AskQuestionResponse, tags=["QuestionAnswer"], summary="Ask a question about an uploaded PDF.", description="Ask a question using a PDF ID. The backend returns an answer using an LLM.")
async def ask_question(request: AskQuestionRequest) -> AskQuestionResponse:
    """
    Given a pdf_id and question, extract text context from DB, call OpenAI to answer, store Q/A in history, and return answer.
    """
    pdf_text = db.get_pdf_content(request.pdf_id)
    if not pdf_text:
        raise HTTPException(status_code=404, detail="PDF not found.")

    try:
        # Use a chunk of context if extremely large
        max_context_len = 8000
        context = pdf_text[:max_context_len]
        answer = await openai_answer(request.question, context)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error during LLM processing: {str(ex)}")

    db.save_qa(request.pdf_id, request.question, answer)
    return AskQuestionResponse(pdf_id=request.pdf_id, question=request.question, answer=answer)

# PUBLIC_INTERFACE
@app.get("/history", response_model=HistoryResponse, tags=["History"], summary="Get Q&A history.", description="Retrieve previous questions and answers. Optionally filter by PDF ID (use query param 'pdf_id').")
def history(pdf_id: Optional[str] = None) -> HistoryResponse:
    """
    Returns history of Q&A pairs, optionally filtered by PDF.
    """
    history_items = db.get_history(pdf_id=pdf_id)
    return HistoryResponse(history=history_items)
