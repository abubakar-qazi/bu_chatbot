import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time

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
    try:
        st.image("img/download.png", width=200)
    except:
        pass
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
    try:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

        # Process documents only once
        if "vectors" not in st.session_state:
            with st.spinner("🔍 Loading and processing handbook... Please wait, this may take 1-2 minutes."):
                try:
                    # Load and split documents FIRST (before embeddings)
                    loader = PyPDFLoader(handbook_path)
                    raw_docs = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,  # Smaller chunks for faster processing
                        chunk_overlap=50
                    )
                    documents = text_splitter.split_documents(raw_docs)
                    
                    st.info(f"📄 Processing {len(documents)} document chunks...")

                    # Process in batches to avoid timeout
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=google_api_key
                    )
                    
                    # Process in smaller batches
                    batch_size = 10
                    all_vectors = []
                    
                    progress_bar = st.progress(0)
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        try:
                            if i == 0:
                                # Create initial vector store
                                st.session_state.vectors = FAISS.from_documents(batch, embeddings)
                            else:
                                # Add to existing vector store
                                batch_vectors = FAISS.from_documents(batch, embeddings)
                                st.session_state.vectors.merge_from(batch_vectors)
                            
                            # Update progress
                            progress = min((i + batch_size) / len(documents), 1.0)
                            progress_bar.progress(progress)
                            
                            # Small delay to avoid rate limits
                            time.sleep(0.5)
                        except Exception as e:
                            st.warning(f"⚠️ Batch {i//batch_size + 1} had an issue, retrying...")
                            time.sleep(2)
                            # Retry once
                            if i == 0:
                                st.session_state.vectors = FAISS.from_documents(batch, embeddings)
                            else:
                                batch_vectors = FAISS.from_documents(batch, embeddings)
                                st.session_state.vectors.merge_from(batch_vectors)
                    
                    progress_bar.progress(1.0)
                    st.success("✅ Handbook loaded successfully!")
                    time.sleep(1)
                    st.rerun()
                    
                except FileNotFoundError:
                    st.error(f"❌ Handbook file not found at: {handbook_path}")
                    st.info("Please ensure the handbook.pdf file is in the 'data' folder.")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.info("This might be due to API rate limits. Please try again in a moment.")
                    if st.button("🔄 Retry"):
                        st.rerun()
                    st.stop()

        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the Bahria University Handbook.
        If the answer is not in the context, say "I don't have information about that in the handbook."
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """)

        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("Ask me about BU policies..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = retrieval_chain.invoke({"input": user_input})
                        answer = result["answer"]
                        st.markdown(answer)

                        st.session_state.messages.append({"role": "assistant", "content": answer})

                        # Show source context
                        with st.expander("📎 View Source Context"):
                            for i, doc in enumerate(result["context"]):
                                st.markdown(f"**Source {i+1}**")
                                st.write(doc.page_content)
                                st.markdown("---")
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {str(e)}")
        st.stop()
else:
    st.warning("⚠️ Please configure your API keys in Streamlit secrets to begin.")
