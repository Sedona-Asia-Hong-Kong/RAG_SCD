import streamlit as st
from RAG_SCD import answer_question  # adjust import to your filename

st.set_page_config(page_title="SCD RAG Assistant", page_icon="📘")

st.title("SimCorp Dimension RAG Assistant")
st.caption("Ask questions about your SCD documentation PDFs.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about SCD workflows, modules, config..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking with your SCD PDFs..."):
            answer = answer_question(prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
