import streamlit as st

st.title("Physical AI Textbook Chatbot")
st.write("Hello! I am your RAG chatbot. Ask me anything about the textbook.")

# Simple input box
user_input = st.text_input("Type your question here:")

if user_input:
    st.write(f"You asked: {user_input}")
