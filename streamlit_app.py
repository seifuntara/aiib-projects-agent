import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AIIB Projects Search Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA PROCESSING
# ============================================================================
@st.cache_data
def load_and_clean_data(EUR_TO_USD=1.18):
    """Load and clean AIIB projects data"""
    try:
        # Load CSV
        df = pd.read_csv('AIIB_Projects.csv', sep=',')
        
        # Extract Project Name (remove country prefix)
        df['Project Name'] = df['Project Name'].apply(
            lambda x: (str(x).split(':', 1)[1].strip() or x) if ':' in str(x) else x
        )
        
        # Extract amount and standardize to USD
        amounts = df['Financing Amount'].str.extract(r'(\d+\.?\d*)')[0]
        is_eur = df['Financing Amount'].str.contains('EUR|€', na=False)
        df['Financing Amount'] = pd.to_numeric(amounts, errors='coerce').fillna(0)
        df.loc[is_eur, 'Financing Amount'] *= EUR_TO_USD
        df = df.rename(columns={'Financing Amount': 'Financing Amount (million USD)'})
        return df
        
    except FileNotFoundError:
        st.error("❌ AIIB_Projects.csv not found. Please upload the file to the same directory.")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# ============================================================================
# GEMINI CLIENT SETUP
# ============================================================================
@st.cache_resource
def get_gemini_client():
    """Initialize Gemini client"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            st.error("⚠️ Please set GOOGLE_API_KEY in environment variables or Streamlit secrets")
            st.info("Get your free API key at: https://makersuite.google.com/app/apikey")
            st.stop()
    
    return genai.Client(api_key=api_key)

# Initialize
client = get_gemini_client()
df = load_and_clean_data()

# Convert to Markdown for context
context_data = df.to_markdown(index=False)

# System instructions
system_prompt = f"""
You are an AI assistant with knowledge of AIIB projects. 
Use the following project data to answer questions. 
You may also use general knowledge to add more details about the projects.
For calculations, reason through all calculations internally step-by-step in your hidden layer, but output only the final verified results.
Provide a cohesive answer overall.

<project_data>
{context_data}
</project_data>
"""

# ============================================================================
# QUERY FUNCTION
# ============================================================================
def ask_gemini(user_query):
    """Send query to Gemini with system instructions"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1
            )
        )
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# UI - HEADER
# ============================================================================
st.title("🏦 AIIB Projects Search Agent")
st.markdown("*Ask questions about AIIB projects using LLM-powered contextual search*")
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

# Example questions
st.markdown("**💡 Try asking:**")

col1, col2 = st.columns(2)

with col1:
    if st.button("What renewable energy projects has AIIB approved in Indonesia?", use_container_width=True):
        st.session_state.example_query = "What renewable energy projects has AIIB approved in Indonesia?"

    if st.button("Show infrastructure projects approved by AIIB in 2021 in India", use_container_width=True):
        st.session_state.example_query = "Show infrastructure projects approved by AIIB in 2021 in India"

with col2:
    if st.button("List non-sovereign projects in Bangladesh related to transport", use_container_width=True):
        st.session_state.example_query = "List non-sovereign projects in Bangladesh related to transport"

    if st.button("Example AIIB projects related to internet connectivity", use_container_width=True):
        st.session_state.example_query = "Example AIIB projects related to internet connectivity"

# Handle example query clicks
if "example_query" in st.session_state:
    user_query = st.session_state.example_query
    del st.session_state.example_query
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Get response from Gemini
    with st.spinner("🤔 Analyzing data and generating answer..."):
        response = ask_gemini(user_query)
    
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
    
    # Get response from Gemini
    with st.spinner("🤔 Analyzing data and generating answer..."):
        response = ask_gemini(prompt)
    
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
    This dashboard uses **LLM-Powered** search to answer questions about projects financed by the **Asian Infrastructure Investment Bank (AIIB)**.
    
    **Architecture:**
    - 📊 Project data loaded into context
    - 🤖 Gemini 2.5 Flash with system instructions
    - 💡 General knowledge integration
                
    """)

    st.markdown("---")
    st.markdown("### 📊 Data Info")
    st.markdown(f"**Projects loaded:** {len(df)}")
    st.markdown("**Last updated:** 15 Feb 2026")
    st.caption(
        "Data sourced from publicly available AIIB project disclosures. "
        "[Learn more about AIIB](https://www.aiib.org/en/projects/list/index.html)."
    )