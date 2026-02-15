import streamlit as st
import pandas as pd
from operator import itemgetter
import os

# LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AIIB Projects Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALIZE RAG SYSTEM
# ============================================================================
@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with FAISS vector store and Gemini"""
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            st.error("⚠️ Please set GOOGLE_API_KEY in environment variables or Streamlit secrets")
            st.info("Get your free API key at: https://makersuite.google.com/app/apikey")
            st.stop()
    
    # Load CSV
    try:
        df = pd.read_csv("AIIB_Projects.csv", sep='\t')
    except FileNotFoundError:
        st.error("❌ AIIB_Projects.csv not found. Please upload the file to the same directory.")
        st.stop()
    
    # Create documents from CSV
    docs = [
        Document(
            page_content=", ".join(f"{col}: {row[col]}" for col in df.columns),
            metadata={"row_index": idx}
        ) 
        for idx, row in df.iterrows()
    ]
    
    # Initialize embeddings (local model)
    with st.spinner("🔄 Loading embedding model..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS vector store
    with st.spinner("🔄 Building vector database..."):
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    
    # Create prompt template
    template = """You are an AI assistant with knowledge of AIIB projects and global context. 
Use the following pieces of retrieved CSV data to answer the question. 
You may also use general knowledge too add more details. Provide a cohesive answer overall.

Context:
{context}

Question: {question}
"""
    prompt = PromptTemplate.from_template(template)
    
    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build the RAG chain
    qa_chain = (
        {
            "context": itemgetter("query") | retriever | format_docs, 
            "question": itemgetter("query")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    st.sidebar.success("✅ RAG system initialized")
    
    return qa_chain, len(df)

# Initialize system
qa_chain, num_projects = initialize_rag_system()

# ============================================================================
# UI - HEADER
# ============================================================================
st.title("🏦 AIIB Projects Search Agent")
st.markdown("*Ask questions about AIIB projects using AI-powered semantic search*")
st.markdown("---")

# ============================================================================
# UI - CHATBOT
# ============================================================================
st.subheader("💬 AI Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Example questions (show when no messages yet)
if len(st.session_state.messages) == 0:
    st.markdown("**💡 Try asking:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What are the largest transport infrastructure projects", use_container_width=True):
            st.session_state.example_query = "What are the largest transport infrastructure projects"

        if st.button("What renewable energy projects are in Indonesia", use_container_width=True):
            st.session_state.example_query = "What renewable energy projects are in Indonesia"

    with col2:
        if st.button("Explain flooding-related projects in Bengal", use_container_width=True):
            st.session_state.example_query = "Explain flooding-related projects in Bengal"

        if st.button("Show non-sovereign projects in India in 2024", use_container_width=True):
            st.session_state.example_query = "Show non-sovereign projects in India in 2024"


# Handle example query clicks
if "example_query" in st.session_state:
    user_query = st.session_state.example_query
    del st.session_state.example_query
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Get response from RAG chain
    with st.spinner("🤔 Searching through projects and generating answer..."):
        try:
            response = qa_chain.invoke({"query": user_query})
        except Exception as e:
            response = f"❌ Error: {str(e)}"
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about AIIB projects..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from RAG chain
    with st.spinner("🤔 Searching through projects and generating answer..."):
        try:
            response = qa_chain.invoke({"query": prompt})
        except Exception as e:
            response = f"❌ Error: {str(e)}"
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# ============================================================================
# SIDEBAR - INFO
# ============================================================================
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This dashboard uses **Retrieval-Augmented Generation (RAG)** to answer questions about projects financed by the **Asian Infrastructure Investment Bank (AIIB)**.
    
    **Tech Stack:**
    - 🔍 **FAISS** - Vector search
    - 🤖 **HuggingFace** - Embeddings
    - ✨ **Gemini 2.5 Flash** - LLM
    - 🔗 **LangChain** - Orchestration
    """)

    st.markdown("---")
    st.markdown("### 📁 Data Source")
    st.markdown(f"**Projects loaded:** {num_projects}")
    st.markdown("**Last updated:** 15 Feb 2026")
    st.caption(
        "Data sourced from publicly available AIIB project disclosures. "
        "[Learn more about AIIB](https://www.aiib.org)."
    )
