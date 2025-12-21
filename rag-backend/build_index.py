import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ---------------------------
# Configuration
# ---------------------------
DOCS_PATH = "../docs"
INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500

os.makedirs(INDEX_PATH, exist_ok=True)

# ---------------------------
# Load embedding model
# ---------------------------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ---------------------------
# Read all markdown files
# ---------------------------
documents = []

for root, _, files in os.walk(DOCS_PATH):
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)

print(f"Loaded {len(documents)} documents")

# ---------------------------
# Split text into chunks
# ---------------------------
chunks = []

for doc in documents:
    for i in range(0, len(doc), CHUNK_SIZE):
        chunks.append(doc[i:i + CHUNK_SIZE])

print(f"Created {len(chunks)} text chunks")

# ---------------------------
# Create embeddings
# ---------------------------
print("Creating embeddings...")
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---------------------------
# Save index and chunks
# ---------------------------
faiss.write_index(index, os.path.join(INDEX_PATH, "index.faiss"))

with open(os.path.join(INDEX_PATH, "chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index built and saved successfully")

