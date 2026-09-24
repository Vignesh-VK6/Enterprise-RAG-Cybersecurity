import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load embedding model
# -----------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# -----------------------------
# Load FAISS vector database
# -----------------------------
index = faiss.read_index("vector_db/faiss.index")

print(f"Total vectors in database: {index.ntotal}")


# -----------------------------
# Load chunks
# -----------------------------
with open("vector_db/chunks.txt", "r", encoding="utf-8") as f:
    chunks_text = f.read()

chunks = [
    chunk.strip()
    for chunk in chunks_text.split("--- CHUNK ")
    if chunk.strip()
]


# -----------------------------
# Semantic Search Function
# -----------------------------
def semantic_search(query, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    # Convert to float32
    query_embedding = query_embedding.astype("float32")

    # Search FAISS
    scores, indices = index.search(query_embedding, top_k)

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    # Display results
    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):

        print(f"\nRank {rank}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Chunk ID: {idx}")

        print("\nRetrieved Text:")
        print(chunks[idx][:1000])

        print("-" * 80)


# -----------------------------
# Test Query
# -----------------------------
query = input("\nEnter your question: ")

semantic_search(query, top_k=5)