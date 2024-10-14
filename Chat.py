import streamlit as st
from query_data import query_rag
import time

st.set_page_config(
    page_title="Chat",
    page_icon="🤖",
)

st.title("Chat with AI")

prompt = st.chat_input("Enter your question here")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    # st.write(prompt)
    with st.spinner('Retrieving response...'):
        query = query_rag(prompt)
    response_text = query[0]
    sources = query[1]
    with st.chat_message("ai"):
        st.write(response_text)
        st.write("Sources:")
        for source in sources:
            st.write(f"- {source}")