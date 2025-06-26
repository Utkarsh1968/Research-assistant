# 🤖 RISA: AI Research Assistant

> **An intelligent assistant that helps summarize, analyze, and question research papers — built using FastAPI & Gemini Pro, with a sleek React frontend.**

---

## 🧠 Features

- 📄 Upload PDF research papers
- ✨ Get AI-generated **summaries** of the document
- ❓ Generate **challenge questions** and quiz-like insights
- 💬 Ask **free-form questions** based on document content
- ⚡️ Fast performance, even on large files
- 🖥️ Built to run on both **local** and **cloud (Render, Vercel)**

---

## 📸 Screenshots

> Add screenshots in this section after deployment or while running locally.

| Upload PDF | Summary View | Q&A Interface |
|------------|---------------|---------------|
| ![upload](assets/upload.png) | ![summary](assets/summary.png) | ![qa](assets/qa.png) |

---

## 🛠️ Tech Stack

| Technology    | Purpose                   |
|---------------|---------------------------|
| 🐍 Python + FastAPI | Backend REST API             |
| ⚡️ Uvicorn         | ASGI Server                   |
| 📚 PyMuPDF (`fitz`) | PDF Parsing                  |
| 🌐 React + Vite     | Frontend Framework           |
| 💬 Gemini Pro       | AI Summarization + QA        |
| 🧪 Pydantic         | Data Validation              |
| 🧑‍💻 TailwindCSS     | UI Styling                   |
| 🧾 Render & Vercel  | Deployment Platforms         |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Utkarsh1968/Research-assistant.git
cd Research-assistant
