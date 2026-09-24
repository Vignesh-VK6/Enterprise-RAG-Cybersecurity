import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
def rerank_results(query, chunks, top_k=3):

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    pairs = [
        [query, chunk]
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    results = list(
        zip(chunks, scores)
    )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# LOAD CHUNKS
# ============================================================

with open("vector_db/chunks.txt", "r", encoding="utf-8") as f:
    chunks_text = f.read()

chunks = [
    chunk.strip()
    for chunk in chunks_text.split("--- CHUNK ")
    if chunk.strip()
]

print("=" * 80)
print("RERANKING IMPLEMENTATION")
print("=" * 80)

print(f"Total chunks: {len(chunks)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

index = faiss.read_index(
    "vector_db/faiss.index"
)

print("FAISS index loaded successfully.")


# ============================================================
# LOAD RERANKER
# ============================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded successfully.")


# ============================================================
# TEST QUERY
# ============================================================

query = "What is information security risk management?"

print("\n" + "=" * 80)
print("QUERY")
print("=" * 80)
print(query)


# ============================================================
# STEP 1: INITIAL RETRIEVAL - TOP 10
# ============================================================

query_embedding = embedding_model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")


vector_scores, vector_indices = index.search(
    query_embedding,
    10
)


print("\n" + "=" * 80)
print("INITIAL RETRIEVAL - TOP 10")
print("=" * 80)

for rank, (idx, score) in enumerate(
    zip(vector_indices[0], vector_scores[0]),
    start=1
):
    print(
        f"Rank {rank} | "
        f"Chunk {idx} | "
        f"Vector Score {score:.4f}"
    )


# ============================================================
# GET TOP 10 CHUNKS
# ============================================================

top_10_indices = vector_indices[0]

top_10_chunks = [
    chunks[idx]
    for idx in top_10_indices
    if idx >= 0 and idx < len(chunks)
]


# ============================================================
# STEP 2: RERANK TOP 10
# ============================================================

print("\n" + "=" * 80)
print("RUNNING RERANKER")
print("=" * 80)

pairs = [
    (query, chunk)
    for chunk in top_10_chunks
]


rerank_scores = reranker.predict(pairs)


# ============================================================
# COMBINE CHUNK + SCORE
# ============================================================

reranked_results = list(
    zip(
        top_10_indices,
        top_10_chunks,
        rerank_scores
    )
)


# Sort by reranker score - highest first

reranked_results.sort(
    key=lambda x: x[2],
    reverse=True
)


# ============================================================
# STEP 3: TOP 3 AFTER RERANKING
# ============================================================

top_3_results = reranked_results[:3]


print("\n" + "=" * 80)
print("AFTER RERANKING - TOP 3 CONTEXTS")
print("=" * 80)


for rank, (idx, chunk, score) in enumerate(
    top_3_results,
    start=1
):

    print(
        f"\nRank {rank} | "
        f"Chunk {idx} | "
        f"Rerank Score {score:.4f}"
    )

    print("-" * 80)

    # Show first 500 characters only
    print(chunk[:500])

    print("...")


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("RETRIEVAL QUALITY COMPARISON")
print("=" * 80)

print("\nBEFORE RERANKING:")
print("FAISS retrieved Top 10 chunks.")

for rank, idx in enumerate(top_10_indices, start=1):
    print(f"Rank {rank} -> Chunk {idx}")


print("\nAFTER RERANKING:")
print("Reranker selected the Top 3 most relevant contexts.")

for rank, (idx, chunk, score) in enumerate(
    top_3_results,
    start=1
):
    print(
        f"Rank {rank} -> "
        f"Chunk {idx} -> "
        f"Rerank Score {score:.4f}"
    )


print("\n" + "=" * 80)
print("RERANKING COMPLETED SUCCESSFULLY!")
print("=" * 80)