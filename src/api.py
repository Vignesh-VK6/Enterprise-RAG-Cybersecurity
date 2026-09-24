from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
import shutil
import pdfplumber

from src.rag_service import ask_rag


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# CHAT HISTORY
# ==================================================

chat_history = []


# ==================================================
# FASTAPI APP
# ==================================================

app = FastAPI(
    title="Enterprise RAG API",
    description="FastAPI backend for Cybersecurity & Privacy RAG system",
    version="1.0.0"
)


# ==================================================
# ROOT
# ==================================================

@app.get("/")
def root():
    return {
        "message": "Enterprise RAG API is running"
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ==================================================
# ASK RAG
# ==================================================

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = ask_rag(request.question)

    chat_history.append({
        "question": request.question,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", [])
    })

    return result


# ==================================================
# UPLOAD DOCUMENT
# ==================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Document uploaded successfully",
        "filename": file.filename,
        "path": str(file_path)
    }


# ==================================================
# PROCESS DOCUMENT
# ==================================================

@app.post("/process")
def process_document(filename: str):

    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    if file_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    extracted_text = ""

    with pdfplumber.open(file_path) as pdf:

        page_count = len(pdf.pages)

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                extracted_text += text + "\n"

    return {
        "message": "Document processed successfully",
        "filename": filename,
        "pages": page_count,
        "characters": len(extracted_text)
    }


# ==================================================
# SOURCES
# ==================================================

@app.get("/sources")
def get_sources():

    return {
        "sources": [
            {
                "document": "NIST - pdf 1.pdf",
                "organization": "NIST",
                "domain": "Cybersecurity",
                "sub_domain": "Cybersecurity & Privacy",
                "pages": 178,
                "chunks": 630
            }
        ]
    }


# ==================================================
# DELETE DOCUMENT
# ==================================================

@app.delete("/documents/{filename}")
def delete_document(filename: str):

    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    if file_path.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files can be deleted."
        )

    file_path.unlink()

    return {
        "message": "Document deleted successfully",
        "filename": filename
    }


# ==================================================
# CHAT HISTORY
# ==================================================

@app.get("/chat-history")
def get_chat_history():

    return {
        "count": len(chat_history),
        "history": chat_history
    }