# assistant.py
from fastapi import APIRouter, UploadFile, File, Form
from utils.pdf_parser import extract_text_from_pdf
from utils.qa_engine import get_document_chunks, get_most_relevant_chunk, answer_question_with_context, summarize_document
import tempfile

router = APIRouter()
  

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global assistant_chunks
    suffix = ".pdf" if file.filename.endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    if suffix == ".pdf":
        text = extract_text_from_pdf(temp_path)
    else:
        with open(temp_path, 'r', encoding='utf-8') as f:
            text = f.read()

    assistant_chunks = get_document_chunks(text)
    summary = summarize_document(text)
    return {"summary": summary}

@router.post("/ask")
async def ask_question(question: str = Form(...)):
    global assistant_chunks
    if not assistant_chunks:
        return {"answer": "Please upload a document before asking questions."}
    best_chunk = get_most_relevant_chunk(question)
    answer = answer_question_with_context(question, best_chunk)
    return {"answer": answer}
