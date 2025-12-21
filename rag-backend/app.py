import streamlit as st
import pickle
import faiss
from pathlib import Path

st.title("Physical AI Textbook Chatbot")
st.write("Ask anything about the textbook!")

# Load FAISS index
index_file = Path("faiss_index/index.faiss")
doc_file = Path("faiss_index/docs.pkl")

if index_file.exists() and doc_file.exists():
    index = faiss.read_index(str(index_file))
    with open(doc_file, "rb") as f:
        documents = pickle.load(f)
    st.success("Index loaded successfully!")
else:
    st.error("FAISS index or documents not found. Run build_index.py first.")

# Input box
user_question = st.text_input("Type your question:")

if user_question:
    # Step 1: Convert question to vector (simple placeholder)
    # For now we will use random vector just to check flow
    import numpy as np
    question_vector = np.random.rand(1, 768).astype("float32")  # later replace with real embedding
    
    # Step 2: Search FAISS index
    D, I = index.search(question_vector, k=1)
    
    # Step 3: Retrieve the document
    retrieved_doc = documents[I[0][0]]
    
    # Step 4: Show answer (for now we just show the retrieved text)
    st.write("Answer from textbook content:")
    st.write(retrieved_doc)
