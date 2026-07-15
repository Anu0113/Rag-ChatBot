# RAG PDF ChatBot

A PDF question-answering chatbot built using **Streamlit, Google Gemini, LangChain, FAISS, and pdfplumber**.

The application allows users to upload a PDF document and ask questions about its content. RagBot retrieves the most relevant sections from the document and uses Gemini to generate a clear answer based only on the uploaded PDF.

## Live Demo

[View the deployed website](https://rag-chatbot-pdf-assistant.streamlit.app/)

---

## Project Overview

RAG PDF ChatBot is based on the concept of **Retrieval-Augmented Generation**, commonly called RAG.

Instead of sending the complete PDF to the language model every time, the application:

1. Extracts text from the uploaded PDF.
2. Divides the text into smaller chunks.
3. Converts those chunks into numerical embeddings.
4. Stores the embeddings in a FAISS vector database.
5. Retrieves the most relevant chunks for the user’s question.
6. Sends the retrieved context to Gemini.
7. Displays the generated answer along with source page numbers.

This improves response relevance and ensures that answers are based on the uploaded document.

---

## Features

- Upload and process PDF documents
- Ask questions about the uploaded PDF
- Generate document summaries
- Extract key points and important figures
- Display answers in a chat-style interface
- Maintain conversation history during the session
- Show source page numbers for generated answers
- Suggested question buttons
- Clear chat option
- Remove uploaded document option
- Download chat history as a text file
- Secure Gemini API key storage
- Responsive Streamlit interface

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| Google Gemini API | Generates answers |
| LangChain | Builds the RAG workflow |
| FAISS | Stores and retrieves document embeddings |
| pdfplumber | Extracts text from PDF files |
| Gemini Embeddings | Converts document text into vectors |

---

## Project Structure

```text
ChatBot/
│
├── .streamlit/
│   └── secrets.toml
│
├── .gitignore
├── RagBot.py
├── requirements.txt
└── README.md
