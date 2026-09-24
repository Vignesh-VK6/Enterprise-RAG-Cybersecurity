from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Files
input_file = Path("data/chunks.txt")
vector_db = Path("vector_db")
vector_db.mkdir(exist_ok=True)

# Load chunks
text = input_file.read_text(encoding="utf-8")

chunks = [
    chunk.strip()
    for chunk in text.split("--- CHUNK ")
    if chunk.strip()
]

print(f"Total chunks loaded: {len(chunks)}")

# Embedding model
model_name = "BAAI/bge-small-en-v1.5"

print(f"Loading embedding model: {model_name}")
model = SentenceTransformer(model_name)

# Generate embeddings
embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(embeddings, dtype="float32")

print(f"Embedding shape: {embeddings.shape}")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, str(vector_db / "faiss.index"))

# Save chunks
with open(vector_db / "chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"--- CHUNK {i} ---\n")
        f.write(chunk)
        f.write("\n\n")

print("Embeddings generated successfully!")
print("FAISS vector database saved!")
print(f"Vector database location: {vector_db}")
print(f"Total vectors: {index.ntotal}")