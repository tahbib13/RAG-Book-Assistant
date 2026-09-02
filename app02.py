import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHROMA_DIR = "chroma_db"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- Main Background ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(88, 101, 242, 0.15),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(139, 92, 246, 0.12),
                transparent 30%
            ),
            #0b1020;
        color: #f5f7ff;
    }


    /* ---------- Main container ---------- */

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- Header ---------- */

    .hero {
        padding: 35px;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 80, 0.95),
                rgba(25, 30, 60, 0.95)
            );
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 50px rgba(0,0,0,0.25);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #b9c2dd;
        line-height: 1.6;
    }


    /* ---------- Cards ---------- */

    .card {
        background: rgba(20, 27, 52, 0.90);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }


    /* ---------- Stat Cards ---------- */

    .stat-card {
        background: linear-gradient(
            135deg,
            rgba(34, 43, 78, 0.95),
            rgba(20, 27, 52, 0.95)
        );
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
    }

    .stat-number {
        font-size: 30px;
        font-weight: 800;
    }

    .stat-label {
        color: #aeb8d4;
        font-size: 14px;
    }


    /* ---------- Question Box ---------- */

    .question-box {
        background: linear-gradient(
            135deg,
            rgba(79, 70, 229, 0.15),
            rgba(124, 58, 237, 0.10)
        );
        border: 1px solid rgba(129, 140, 248, 0.25);
        border-radius: 20px;
        padding: 22px;
        margin-top: 20px;
    }


    /* ---------- Answer ---------- */

    .answer-header {
        font-size: 24px;
        font-weight: 750;
        margin-bottom: 10px;
    }


    /* ---------- Source ---------- */

    .source-box {
        background: rgba(12, 18, 38, 0.9);
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 14px;
        margin: 8px 0;
    }


    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #080d1c;
        border-right: 1px solid rgba(255,255,255,0.08);
    }


    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        min-height: 45px;
    }


    /* ---------- Input ---------- */

    .stTextInput > div > div > input {
        border-radius: 14px;
        padding: 14px;
    }


    /* ---------- Divider ---------- */

    hr {
        border-color: rgba(255,255,255,0.08);
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_info" not in st.session_state:
    st.session_state.document_info = None


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        📚 RAG Book Assistant
    </div>

    <div class="hero-subtitle">
        Your intelligent document companion.
        Upload a PDF, build a knowledge base, and ask questions
        using Retrieval-Augmented Generation.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ RAG Configuration")

    st.markdown("---")

    st.markdown("### 📖 Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF book",
        type=["pdf"],
        help="Upload a PDF document to create your knowledge base."
    )

    st.markdown("---")

    st.markdown("### 🧩 Chunk Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    st.markdown("---")

    st.markdown("### 🔎 Retrieval")

    top_k = st.slider(
        "Retrieved Documents",
        min_value=2,
        max_value=8,
        value=4
    )

    st.markdown("---")

    if os.path.exists(CHROMA_DIR):

        st.success("🟢 Vector database is ready")

        if st.button(
            "🗑️ Clear Vector Database",
            use_container_width=True
        ):

            try:
                shutil.rmtree(CHROMA_DIR)

                st.session_state.messages = []
                st.session_state.document_info = None

                st.success("Database cleared!")

                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

    else:

        st.info("⚪ No vector database found")

    st.markdown("---")

    st.caption(
        "Powered by LangChain + Mistral AI + Chroma"
    )


# =========================================================
# PDF PROCESSING
# =========================================================

if uploaded_file:

    # Save uploaded PDF
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name


    # Document information
    file_size = uploaded_file.size / (1024 * 1024)

    st.markdown("### 📄 Uploaded Document")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">📕</div>
                <div class="stat-label">
                    {uploaded_file.name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {file_size:.2f} MB
                </div>
                <div class="stat-label">
                    File Size
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-number">
                    🤖
                </div>
                <div class="stat-label">
                    Mistral RAG
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # CREATE VECTOR DATABASE
    # =====================================================

    if st.button(
        "🚀 Create Knowledge Base",
        use_container_width=True
    ):

        with st.status(
            "Building your document knowledge base...",
            expanded=True
        ) as status:

            st.write("📖 Loading PDF...")

            loader = PyPDFLoader(file_path)

            docs = loader.load()

            page_count = len(docs)

            st.write(
                f"✅ {page_count} pages loaded"
            )


            st.write("✂️ Splitting document into chunks...")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = splitter.split_documents(docs)

            st.write(
                f"✅ Created {len(chunks)} text chunks"
            )


            st.write("🧠 Creating embeddings...")

            embeddings = MistralAIEmbeddings(
                model="mistral-embed"
            )


            # Remove previous database
            if os.path.exists(CHROMA_DIR):

                shutil.rmtree(CHROMA_DIR)


            st.write("💾 Creating Chroma vector database...")

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )


            st.session_state.document_info = {
                "pages": page_count,
                "chunks": len(chunks),
                "name": uploaded_file.name
            }


            status.update(
                label="Knowledge base created successfully! 🎉",
                state="complete"
            )


        st.success(
            "Your PDF is now ready for questions!"
        )

        st.rerun()


# =========================================================
# VECTOR DATABASE + RAG
# =========================================================

if os.path.exists(CHROMA_DIR):

    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )


    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


    # =====================================================
    # LLM
    # =====================================================

    llm = ChatMistralAI(
        model="mistral-small-2603"
    )


    # =====================================================
    # IMPROVED PROMPT
    # =====================================================

    prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",
                """
You are an expert AI research assistant specializing
in answering questions from uploaded documents.

Your answer MUST be based ONLY on the provided context.

Do not use outside knowledge.

If the answer cannot be found in the context, respond:

"I could not find the answer in the document."

When the answer is available:

1. Start with a direct answer.
2. Explain the answer in detail.
3. Break complex information into clear sections.
4. Use bullet points or numbered lists when appropriate.
5. Include important examples mentioned in the document.
6. Explain technical terms when necessary.
7. Compare concepts when the question asks for comparison.
8. Mention important details instead of giving an overly short answer.
9. Do not invent facts.
10. Do not assume information that is not present in the context.

The response should be informative, natural, and easy to understand.

Context:
{context}

Question:
{question}
"""
            ),

            (
                "human",
                """
Using only the context above, provide the most
complete and descriptive answer possible.

Question:
{question}
"""
            )

        ]
    )


    # =====================================================
    # DOCUMENT STATUS
    # =====================================================

    st.markdown("## 📊 Knowledge Base")

    col1, col2, col3, col4 = st.columns(4)

    info = st.session_state.document_info

    if info:

        with col1:
            st.metric(
                "📄 Pages",
                info["pages"]
            )

        with col2:
            st.metric(
                "🧩 Chunks",
                info["chunks"]
            )

        with col3:
            st.metric(
                "🔎 Retrieval",
                top_k
            )

        with col4:
            st.metric(
                "🤖 Model",
                "Mistral"
            )

    else:

        with col1:
            st.metric("📄 Pages", "Ready")

        with col2:
            st.metric("🧩 Chunks", "Ready")

        with col3:
            st.metric("🔎 Retrieval", top_k)

        with col4:
            st.metric("🤖 Model", "Mistral")


    st.divider()


    # =====================================================
    # EXAMPLE QUESTIONS
    # =====================================================

    st.markdown("## 💡 Try asking")

    example_questions = [
        "What is the main topic of this book?",
        "Summarize the most important concepts.",
        "Explain the key findings in detail.",
        "What are the main advantages and disadvantages?",
    ]

    cols = st.columns(2)

    for i, question in enumerate(example_questions):

        with cols[i % 2]:

            if st.button(
                question,
                key=f"example_{i}",
                use_container_width=True
            ):

                st.session_state["selected_question"] = question


    # =====================================================
    # QUESTION INPUT
    # =====================================================

    st.markdown(
        """
        <div class="question-box">
            <h3>💬 Ask Your Book</h3>
            <p>
                Ask anything about the uploaded document.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    default_question = st.session_state.pop(
        "selected_question",
        ""
    )


    query = st.text_input(
        "Your question",
        value=default_question,
        placeholder="e.g. Explain the main contribution of this book...",
        label_visibility="collapsed"
    )


    # =====================================================
    # ANSWER
    # =====================================================

    if query:

        with st.spinner(
            "🔍 Searching the document and generating answer..."
        ):

            # Retrieve relevant chunks
            docs = retriever.invoke(query)


            # Create context
            context = "\n\n".join(
                [
                    f"[Page {doc.metadata.get('page', 0) + 1}]\n"
                    f"{doc.page_content}"
                    for doc in docs
                ]
            )


            # Generate prompt
            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )


            # LLM response
            response = llm.invoke(
                final_prompt
            )


        # =================================================
        # SAVE CHAT
        # =================================================

        st.session_state.messages.append(
            {
                "question": query,
                "answer": response.content,
                "sources": docs
            }
        )


    # =====================================================
    # DISPLAY CHAT HISTORY
    # =====================================================

    if st.session_state.messages:

        st.markdown("## 🧠 Conversation")


        for message in reversed(
            st.session_state.messages
        ):

            # Question
            with st.chat_message(
                "user",
                avatar="👤"
            ):

                st.markdown(
                    message["question"]
                )


            # Answer
            with st.chat_message(
                "assistant",
                avatar="🤖"
            ):

                st.markdown(
                    message["answer"]
                )


                # -----------------------------------------
                # Sources
                # -----------------------------------------

                st.markdown("---")

                st.markdown(
                    "### 🔎 Retrieved Sources"
                )


                unique_pages = []

                for doc in message["sources"]:

                    page = (
                        doc.metadata.get("page", 0) + 1
                    )

                    if page not in unique_pages:
                        unique_pages.append(page)


                    with st.expander(
                        f"📄 Page {page}"
                    ):

                        st.write(
                            doc.page_content
                        )


                st.caption(
                    "Answer generated using retrieved "
                    f"content from pages: "
                    f"{', '.join(map(str, unique_pages))}"
                )


else:

    # =====================================================
    # EMPTY STATE
    # =====================================================

    st.markdown(
        """
        <div class="card">

            <h2>👋 Welcome to your RAG Book Assistant</h2>

            <p>
                Start by uploading a PDF from the sidebar.
                The application will convert the document
                into searchable vector embeddings.
            </p>

            <br>

            <b>How it works:</b>

            <br><br>

            📄 Upload PDF
            →
            ✂️ Split into chunks
            →
            🧠 Generate embeddings
            →
            🗄️ Store in Chroma
            →
            🔎 Retrieve relevant content
            →
            🤖 Generate answer

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### ✨ Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">

            <h3>🔎 Semantic Search</h3>

            Finds relevant information based
            on meaning rather than exact keywords.

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">

            <h3>🧠 Context-Aware AI</h3>

            Mistral generates answers using
            retrieved document context.

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="card">

            <h3>📚 Source Tracking</h3>

            See which PDF pages were used
            to generate each answer.

            </div>
            """,
            unsafe_allow_html=True
        )