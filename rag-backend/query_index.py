import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ---------------------------
# Configuration
# ---------------------------
INDEX_PATH = "faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# ---------------------------
# Load model
# ---------------------------
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ---------------------------
# Load FAISS index
# ---------------------------
print("Loading FAISS index...")
index = faiss.read_index(f"{INDEX_PATH}/index.faiss")

with open(f"{INDEX_PATH}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# ---------------------------
# Ask question
# ---------------------------
question = input("\nAsk a question about the book: ")

question_embedding = model.encode([question])

# ---------------------------
# Search
# ---------------------------
distances, indices = index.search(question_embedding, TOP_K)

print("\nAnswer (from book content only):\n")

for idx in indices[0]:
    print("-" * 40)
    print(chunks[idx])

