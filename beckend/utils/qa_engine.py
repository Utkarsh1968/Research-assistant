from langchain_core.documents import Document
import google.generativeai as genai
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3",
    task="text-generation",
    max_new_tokens=512,
    huggingfacehub_api_token= os.getenv("HUGGINGFACEHUB_API_TOKEN")   
)
model = ChatHuggingFace(llm= llm)

vector_store = Chroma(
    embedding_function= HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2'),
    collection_name= "Features"
)

parser = StrOutputParser()

def get_document_chunks(text, chunk_size=500):
    doc = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = splitter.split_documents([doc])  
    vector_store.add_documents(chunks)  
    return chunks

def get_most_relevant_chunk(question):
    relevant_docs = vector_store.similarity_search(question, k=1)
    return relevant_docs[0].page_content if relevant_docs else None

def answer_question_with_context(question, context):
    prompt = f"""You are a smart and helpful AI assistant.

Given the context below, answer the question in a natural, free-form style. 
If the answer is not explicitly stated, use logical reasoning or inferred knowledge from the context.
Avoid saying "no information provided" unless it's truly impossible to infer.
Keep the answer short, clear, and directly related to the question.

Also explain briefly which part of the context helped you answer.

Context:
{context}

Question:
{question}

Answer (with explanation of supporting context):"""
    chain = model | parser
    response = chain.invoke(prompt)
    if isinstance(response, str):
        return response.strip()
    return response.text.strip()



def summarize_document(text):
    prompt = f'''You are a smart AI research assistant.

TASK:
Summarize the following research paper. Focus on preserving key insights, methods, experiments, and results. Make sure to explain technical parts in simpler language while maintaining correctness.

CONSTRAINT:
Keep the summary around 300 words. Ensure no critical information is omitted. Provide matrix (if any).
FORMAT:
- Title
- Objective
- Methodology
- Results
- Conclusion.\n\n{text[:2000]}'''
    chain = model | parser
    response = chain.invoke(prompt)
    if isinstance(response, str):   
        return response.strip()
    return response.text.strip()

def generate_questions(document_text, num_questions=3):
    prompt = f"""
Generate {num_questions} multiple-choice questions (MCQs) from the following document. 
make the MCQs logical and good reasoning from the document
For each question, provide:
- The question text.
- Four answer options (A, B, C, D).
- The index (0-based) of the correct answer.

Respond in this JSON format:
[
  {{
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "correctAnswer": 0
  }},
  ...
]

Document:
{document_text[:2000]}
"""
    response = gemini_model.generate_content(prompt)
    import json
    import re

    text = response.text.strip()
    # Remove code block markers if present
    if text.startswith("```") and text.endswith("```"):
        text = text.strip("`").strip()
    # Try to extract the JSON array if extra text is present
    match = re.search(r'(\[\s*{.*}\s*\])', text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        questions = json.loads(text)
        # Add id field for frontend
        for idx, q in enumerate(questions):
            q["id"] = idx + 1
        return questions
    except Exception as e:
        print("Error parsing MCQ JSON:", e, text)
        return []


def evaluate_answer(question, user_answer, context):
    prompt = f"""You are an AI evaluator. Given the original question, user answer, and document context, provide a score (0-10) and feedback.

Question:
{question}

User Answer:
{user_answer}

Context:
{context}

Now evaluate the answer. Respond in the format:
Score: X/10
Feedback: <your explanation>
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()
