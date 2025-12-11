import os
import streamlit as st
from RAG_SCD import answer_question  # Import the main function

st.set_page_config(page_title="SCD RAG Assistant", layout="wide")
st.title("🔍 Simcorp Dimension RAG Assistant")

# Folder that contains your PDFs
PDF_FOLDER = r"./data_pdf"  # change this to your actual folder path

# Configure your available PDFs here (edit paths as needed)
PDF_OPTIONS = {
    fname: os.path.join(PDF_FOLDER, fname)
    for fname in os.listdir(PDF_FOLDER)
    if fname.lower().endswith(".pdf")
}

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: pick which PDFs to search
st.sidebar.header("Select PDFs to search")
selected = st.sidebar.multiselect("Tick PDFs to include in the search (priority = speed & accuracy):", options=list(PDF_OPTIONS.keys()), default=list(PDF_OPTIONS.keys())[:1])

selected_paths = [PDF_OPTIONS[k] for k in selected]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask about SCD workflows, modules, config..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your question and searching documents..."):
            try:
                # pass selected_paths to RAG_SCD.answer_question
                answer = answer_question(prompt, pdf_paths=selected_paths)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
