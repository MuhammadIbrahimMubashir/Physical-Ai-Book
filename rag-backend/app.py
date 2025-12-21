import streamlit as st
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Functions to build/load index
# -----------------------------
def build_faiss_index(docs_path="docs", index_path="faiss_index/index.faiss", docs_pickle_path="faiss_index/docs.pkl"):
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    
    # Load all text files from docs folder
    texts = []
    file_names = []
    for root, dirs, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    file_names.append(file)
    
    # Embed texts
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and file list
    faiss.write_index(index, index_path)
    with open(docs_pickle_path, "wb") as f:
        pickle.dump(file_names, f)
    
    return index, file_names

def load_faiss_index(index_path="faiss_index/index.faiss", docs_pickle_path="faiss_index/docs.pkl"):
    if not os.path.exists(index_path) or not os.path.exists(docs_pickle_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(docs_pickle_path, "rb") as f:
        file_names = pickle.load(f)
    return index, file_names

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Physical AI Textbook Chatbot")
st.write("Ask anything about the textbook!")

# Load or build index
index, file_names = load_faiss_index()
if index is None:
    st.info("Building FAISS index. Please wait...")
    index, file_names = build_faiss_index()
    st.success("FAISS index built!")

# User input
question = st.text_input("Type your question:")

if question and index:
    # Embed user question
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_vector = model.encode([question])
    
    # Search FAISS
    D, I = index.search(question_vector, k=1)
    result_file = file_names[I[0][0]]
    
    # Show the answer (full text of the matched chapter)
    with open(os.path.join("docs", result_file), "r", encoding="utf-8") as f:
        answer_text = f.read()
    
    st.subheader("Answer from textbook:")
    st.write(answer_text)
