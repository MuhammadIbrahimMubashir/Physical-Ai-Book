# app.py
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

st.set_page_config(page_title="Physical AI Textbook Chatbot", page_icon="🤖")

st.title("Physical AI Textbook Chatbot")
st.write("Ask anything about the textbook chapters (1-9).")
st.write("Powered by local FAISS index and embeddings.")

# Paths
BASE_DIR = os.path.dirname(__file__)
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
DOCS_FILE = os.path.join(FAISS_DIR, "docs.pkl")

# Create folder if missing
os.makedirs(FAISS_DIR, exist_ok=True)

# Build index if missing
if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_FILE):
    st.warning("FAISS index or docs not found. Building index now...")
    import build_index  # This will run build_index.py
    st.success("✅ Index built successfully!")

# Load FAISS index and documents
try:
    index = faiss.read_index(INDEX_FILE)
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

try:
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
except Exception as e:
    st.error(f"Error loading documents: {e}")
    st.stop()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# User input
question = st.text_input("Type your question:")

if question:
    try:
        q_vec = model.encode([question]).astype("float32")
        D, I = index.search(q_vec, k=1)
        answer = docs[I[0][0]]
        st.write("**Answer:**")
        st.write(answer)
    except Exception as e:
        st.error(f"Error retrieving answer: {e}")
