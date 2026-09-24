import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load Model
# -----------------------------
MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# -----------------------------
# Load FAISS Database
# -----------------------------
index = faiss.read_index("vector_db/faiss.index")

print(f"Total vectors: {index.ntotal}")


# -----------------------------
# Load Chunks
# -----------------------------
with open("vector_db/chunks.txt", "r", encoding="utf-8") as f:
    chunks_text = f.read()

chunks = [
    chunk.strip()
    for chunk in chunks_text.split("--- CHUNK ")
    if chunk.strip()
]


# -----------------------------
# 20 Test Queries
# -----------------------------
queries = [
    "What is information security?",
    "What is information security governance?",
    "What is risk management?",
    "What is a security policy?",
    "What is access control?",
    "What is security awareness training?",
    "What is incident response?",
    "What is contingency planning?",
    "What is business continuity?",
    "What is security assessment?",
    "What is vulnerability management?",
    "What is configuration management?",
    "What is physical security?",
    "What is personnel security?",
    "What is security monitoring?",
    "What are the roles and responsibilities in information security?",
    "Why is security training important?",
    "What is the role of management in information security?",
    "How should an organization manage information security risks?",
    "What are security controls?"
]


# -----------------------------
# Run 20 Queries
# -----------------------------
for query_number, query in enumerate(queries, start=1):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        3
    )

    print("\n" + "=" * 80)
    print(f"QUERY {query_number}: {query}")
    print("=" * 80)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"Score: {score:.4f} | "
            f"Chunk: {idx}"
        )

        print(chunks[idx][:300].replace("\n", " "))

    print("-" * 80)


print("\n20 QUERY SEMANTIC SEARCH TEST COMPLETED!")