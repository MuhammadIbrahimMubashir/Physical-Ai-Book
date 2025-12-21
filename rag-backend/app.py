# app.py
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
import subprocess

st.set_page_config(page_title="Physical AI Textbook Chatbot", page_icon="🤖")

st.title("Physical AI Textbook Chatbot")
st.write("Ask anything about the textbook chapters (1-9).")
st.write("Powered by local FAISS index and embeddings.")

# Path to FAISS files
FAISS_DIR = "rag-backend/faiss_index"
INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
DOCS_FILE = os.path.join(FAISS_DIR, "docs.pkl")

# Generate FAISS index if not exists
if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_FILE):
    st.warning("FAISS index or documents not found. Generating now...")
    subprocess.run(["python3", "rag-backend/build_index.py"])
    st.success("Index built successfully!")

# Load FAISS index and docs
index = faiss.read_index(INDEX_FILE)
with open(DOCS_FILE, "rb") as f:
    docs = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# User input
question = st.text_input("Type your question:")

if question:
    # Convert question to embedding
    q_vec = model.encode([question]).astype("float32")

    # Search in FAISS index
    D, I = index.search(q_vec, k=1)
    answer = docs[I[0][0]]

    st.write("**Answer:**")
    st.write(answer)
