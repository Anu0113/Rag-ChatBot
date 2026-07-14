from io import BytesIO
import hashlib

import pdfplumber
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="RAG ChatBot",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99,102,241,0.13), transparent 28%),
                radial-gradient(circle at top right, rgba(168,85,247,0.10), transparent 24%);
        }

        .block-container {
            max-width: 1050px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }

        .hero-card {
            padding: 1.8rem 2rem;
            border: 1px solid rgba(120,120,120,0.20);
            border-radius: 22px;
            background: rgba(255,255,255,0.04);
            box-shadow: 0 12px 35px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 750;
            margin-bottom: 0.35rem;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            opacity: 0.78;
            margin: 0;
        }

        .feature-card {
            min-height: 118px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 16px;
            background: rgba(255,255,255,0.035);
        }

        .feature-card h4 {
            margin: 0 0 0.45rem 0;
        }

        .feature-card p {
            opacity: 0.72;
            margin: 0;
            font-size: 0.92rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(120,120,120,0.18);
        }

        [data-testid="stChatMessage"] {
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid rgba(120,120,120,0.15);
            margin-bottom: 0.7rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(120,120,120,0.16);
            border-radius: 14px;
            padding: 0.75rem;
            background: rgba(255,255,255,0.025);
        }

        .source-text {
            font-size: 0.82rem;
            opacity: 0.62;
            margin-top: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Hero section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📄 RAG ChatBot</div>
        <p class="hero-subtitle">
            Upload a PDF and ask questions directly from your document.
            RagBot uses Gemini and semantic search to find relevant answers.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Read Gemini API key
# ---------------------------------------------------------
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error(
        "Gemini API key not found. Add GEMINI_API_KEY to "
        ".streamlit/secrets.toml."
    )
    st.stop()


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
default_state = {
    "vector_store": None,
    "processed_file_hash": None,
    "messages": [],
    "page_count": 0,
    "chunk_count": 0,
    "file_name": "",
    "file_size": 0,
    "uploader_key": 0,
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_document() -> None:
    """Remove the current document and its conversation."""
    st.session_state.vector_store = None
    st.session_state.processed_file_hash = None
    st.session_state.messages = []
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0
    st.session_state.file_name = ""
    st.session_state.file_size = 0
    st.session_state.uploader_key += 1


def clear_chat() -> None:
    """Clear only the chat while keeping the uploaded document."""
    st.session_state.messages = []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📁 Your Document")
    st.caption("Upload your PDF here:")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    if st.session_state.vector_store is not None:
        st.success("Document ready")

        st.markdown(f"**File:** {st.session_state.file_name}")

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Pages", st.session_state.page_count)
        metric_col2.metric("Chunks", st.session_state.chunk_count)

        size_mb = st.session_state.file_size / (1024 * 1024)
        st.caption(f"File size: {size_mb:.2f} MB")

        st.divider()

        if st.button("🧹 Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()

        if st.button("🗑️ Remove document", use_container_width=True):
            reset_document()
            st.rerun()

        if st.session_state.messages:
            chat_text = "\n\n".join(
                f"{'You' if item['role'] == 'user' else 'RagBot'}:\n"
                f"{item['content']}"
                for item in st.session_state.messages
            )

            st.download_button(
                "⬇️ Download chat",
                data=chat_text,
                file_name="ragbot_chat.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()
    st.caption(
        "How it works:\n\n"
        "1. Upload a PDF\n"
        "2. Wait for processing\n"
        "3. Ask a question"
    )


# ---------------------------------------------------------
# Document processing
# ---------------------------------------------------------
@st.cache_resource(show_spinner="RagBot is processing the PDF...")
def create_vector_store(pdf_bytes: bytes, api_key: str):
    """Extract PDF pages, split them and create a FAISS vector store."""
    page_documents = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()

            if page_text and page_text.strip():
                page_documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"page": page_number},
                    )
                )

    if not page_documents:
        raise ValueError(
            "No readable text was found in the PDF. "
            "The document may contain only scanned images."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(page_documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store, page_count, len(chunks)


if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    current_file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if current_file_hash != st.session_state.processed_file_hash:
        try:
            vector_store, page_count, chunk_count = create_vector_store(
                pdf_bytes,
                GEMINI_API_KEY,
            )

            st.session_state.vector_store = vector_store
            st.session_state.processed_file_hash = current_file_hash
            st.session_state.messages = []
            st.session_state.page_count = page_count
            st.session_state.chunk_count = chunk_count
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_size = len(pdf_bytes)

            st.toast("PDF processed successfully!", icon="✅")

        except Exception as error:
            reset_document()
            st.error(f"Unable to process the PDF: {error}")
            st.stop()


# ---------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------
if st.session_state.vector_store is None:
    st.markdown("### Welcome to RagBot")
    st.write(
        "Upload a PDF from the sidebar. You can then ask for summaries, "
        "important points, dates, figures, definitions and conclusions."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🔎 Find information</h4>
                <p>Ask direct questions and retrieve relevant details from the PDF.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📝 Summarize</h4>
                <p>Generate a clear overview of long reports, notes and documents.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <h4>📌 Check sources</h4>
                <p>See the page numbers used to create each response.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ---------------------------------------------------------
# Gemini model and prompt
# ---------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.2,
    max_output_tokens=600,
    google_api_key=GEMINI_API_KEY,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are RagBot, a helpful PDF question-answering assistant.

Answer using only the supplied PDF context.

Rules:
1. Give a complete and clearly explained answer.
2. Include relevant facts, figures and details from the context.
3. Use bullets when they improve readability.
4. Do not use outside knowledge.
5. Do not invent details.
6. If the answer is absent, say:
   "The uploaded PDF does not contain this information."
""",
        ),
        (
            "human",
            """
PDF context:
{context}

Recent conversation:
{history}

Question:
{question}
""",
        ),
    ]
)


def format_context(documents) -> str:
    """Format retrieved chunks and preserve their page numbers."""
    sections = []

    for document in documents:
        page = document.metadata.get("page", "Unknown")
        sections.append(
            f"[Page {page}]\n{document.page_content}"
        )

    return "\n\n".join(sections)


def recent_chat_history() -> str:
    """Return a small amount of recent history for follow-up questions."""
    recent_messages = st.session_state.messages[-6:]

    if not recent_messages:
        return "No previous conversation."

    return "\n".join(
        f"{'User' if item['role'] == 'user' else 'RagBot'}: "
        f"{item['content']}"
        for item in recent_messages
    )


def answer_question(question: str):
    """Retrieve relevant chunks and ask Gemini to answer."""
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    documents = retriever.invoke(question)
    context = format_context(documents)

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "history": recent_chat_history(),
            "question": question,
        }
    )

    response = llm.invoke(formatted_prompt)

    # Extract only the visible text from Gemini's response
    if isinstance(response.content, str):
        answer = response.content
    else:
        text_parts = []

        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        answer = "\n".join(text_parts).strip()

    pages = sorted(
        {
            document.metadata.get("page")
            for document in documents
            if document.metadata.get("page") is not None
        }
    )

    return answer, pages


# ---------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------
st.markdown("#### Try a suggested question")

suggestion_columns = st.columns(4)

suggestions = [
    ("📝 Summarize", "Summarize this PDF."),
    ("📌 Key points", "What are the key points in this PDF?"),
    ("✅ Conclusion", "What is the conclusion of this PDF?"),
    ("🔢 Important figures", "List the important numbers and figures in this PDF."),
]

selected_question = None

for column, (label, question) in zip(suggestion_columns, suggestions):
    with column:
        if st.button(label, use_container_width=True):
            selected_question = question


# ---------------------------------------------------------
# Display chat history
# ---------------------------------------------------------
st.divider()

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("pages"):
            page_text = ", ".join(
                f"Page {page}" for page in message["pages"]
            )
            st.markdown(
                f'<div class="source-text">Sources: {page_text}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------
# Chat input with send button
# ---------------------------------------------------------
typed_question = st.chat_input("Ask a question about your PDF...")
user_question = selected_question or typed_question


# ---------------------------------------------------------
# Generate a response
# ---------------------------------------------------------
if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_question)

    try:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("RagBot is reading the PDF..."):
                answer, source_pages = answer_question(user_question)

            st.markdown(answer)

            if source_pages:
                page_text = ", ".join(
                    f"Page {page}" for page in source_pages
                )
                st.markdown(
                    f'<div class="source-text">Sources: {page_text}</div>',
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "pages": source_pages,
            }
        )

    except Exception as error:
        st.error(f"RagBot could not generate an answer: {error}")