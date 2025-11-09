import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --------------------------- App Configuration ---------------------------
st.set_page_config(
    page_title="BU Chatbot - Smart Handbook Assistant",
    page_icon="🎓",
    layout="wide"
)

# --------------------------- Custom CSS Styling ---------------------------
st.markdown("""
    <style>
    body, .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    .stChatMessage {
        background-color: #2b2b2b;
        color: #f0f0f0;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #444;
        margin-bottom: 10px;
    }

    .stChatMessage.user {
        background-color: #3a3a3a;
        color: #ffffff;
    }

    .stChatMessage.assistant {
        background-color: #1f3b4d;
        color: #ffffff;
    }

    .stTextInput>div>div>input {
        background-color: #333;
        color: #fff;
        border: 1px solid #555;
    }

    .stButton button {
        background-color: #0066cc;
        color: #ffffff;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff;
    }

    .stExpanderHeader {
        color: #ffffff;
    }

    .sidebar .sidebar-content {
        background-color: #202020;
        color: white;
    }

    </style>
""", unsafe_allow_html=True)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.image("img/download.png", width=200)
    st.markdown("## 🎓 BU Chatbot")
    st.write(
        "An intelligent assistant to help you explore **Bahria University Handbook** policies & rules instantly."
    )

    st.markdown("---")
    st.info("🚀 Tip: Ask anything about attendance policy, grading, scholarships, etc.")

# --------------------------- App Core ---------------------------
st.title("BU Chatbot 📘")
st.subheader("Your AI Assistant for Bahria University Rules & Policies")

# Load document from static folder
handbook_path = "data/handbook.pdf"

# Access secrets from Streamlit's secrets management
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("API keys not found. Please ensure GROQ_API_KEY and GOOGLE_API_KEY are set in your Streamlit secrets.")
    st.stop()

if groq_api_key and google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

    with st.spinner("🔍 Processing... Please wait."):

        if "vectors" not in st.session_state:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            loader = PyPDFLoader(handbook_path)
            raw_docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_documents(raw_docs)

            st.session_state.vectors = FAISS.from_documents(documents, embeddings)

        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response.
        <context>
        {context}
        <context>
        Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Ask me..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                result = retrieval_chain.invoke({"input": user_input})
                answer = result["answer"]
                st.markdown(answer)

                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.expander("📎 Source Context"):
                    for i, doc in enumerate(result["context"]):
                        st.markdown(f"**Page {i+1}**")
                        st.write(doc.page_content)
                        st.markdown("---")
else:
    st.warning("Please enter your API keys in the sidebar to begin.")

