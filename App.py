import streamlit as st
from RAG_SCD import answer_question  # Import the main function

st.set_page_config(page_title="SCD RAG Assistant", layout="wide")
st.title("🔍 Simcorp Dimension RAG Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
                # This calls the FULL pipeline: understand_question() → search → query_deepseek()
                answer = answer_question(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
