import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


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

print(f"Total chunks: {len(chunks)}")


# -----------------------------
# BM25
# -----------------------------
tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# -----------------------------
# Vector Model + FAISS
# -----------------------------
print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

index = faiss.read_index(
    "vector_db/faiss.index"
)


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
# Compare Retrieval Methods
# -----------------------------
for number, query in enumerate(queries, start=1):

    # BM25
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_top = np.argsort(bm25_scores)[::-1][:3]


    # Vector Search
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    vector_scores, vector_indices = index.search(
        query_embedding,
        3
    )

    vector_top = vector_indices[0]


    # Print comparison
    print("\n" + "=" * 80)
    print(f"QUERY {number}: {query}")
    print("=" * 80)

    print("\n--- BM25 KEYWORD SEARCH ---")

    for rank, idx in enumerate(bm25_top, start=1):
        print(
            f"Rank {rank} | "
            f"Chunk {idx} | "
            f"Score {bm25_scores[idx]:.4f}"
        )


    print("\n--- VECTOR SEARCH ---")

    for rank, (idx, score) in enumerate(
        zip(vector_indices[0], vector_scores[0]),
        start=1
    ):
        print(
            f"Rank {rank} | "
            f"Chunk {idx} | "
            f"Score {score:.4f}"
        )


    # Check overlap
    bm25_set = set(bm25_top)
    vector_set = set(vector_top)

    overlap = len(bm25_set.intersection(vector_set))

    print("\n--- COMPARISON ---")
    print(f"Common chunks in Top-3: {overlap}")


print("\n" + "=" * 80)
print("20 QUERY KEYWORD vs VECTOR COMPARISON COMPLETED!")
print("=" * 80)