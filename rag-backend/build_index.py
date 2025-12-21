# build_index.py
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Folder to store FAISS index and docs
BASE_DIR = os.path.dirname(__file__)
FAISS_DIR = os.path.join(BASE_DIR, "faiss_index")
os.makedirs(FAISS_DIR, exist_ok=True)

INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
DOCS_FILE = os.path.join(FAISS_DIR, "docs.pkl")

# Load all chapters from the 'docs' folder
CHAPTERS_DIR = os.path.join(BASE_DIR, "docs")
chapter_files = sorted(os.listdir(CHAPTERS_DIR))

chapters = []
for file in chapter_files:
    if file.endswith(".md"):
        with open(os.path.join(CHAPTERS_DIR, file), "r", encoding="utf-8") as f:
            text = f.read()
            chapters.append(text)

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode chapters
embeddings = model.encode(chapters).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and documents
faiss.write_index(index, INDEX_FILE)
with open(DOCS_FILE, "wb") as f:
    pickle.dump(chapters, f)

print("✅ FAISS index and documents saved successfully!")
