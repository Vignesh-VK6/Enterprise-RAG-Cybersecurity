import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# -----------------------------------
# 1. Load Chunks
# -----------------------------------
with open("vector_db/chunks.txt", "r", encoding="utf-8") as f:
    chunks_text = f.read()

chunks = [
    chunk.strip()
    for chunk in chunks_text.split("--- CHUNK ")
    if chunk.strip()
]

print(f"Total chunks loaded: {len(chunks)}")


# -----------------------------------
# 2. BM25 Keyword Search
# -----------------------------------
tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# -----------------------------------
# 3. Load Embedding Model
# -----------------------------------
print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# -----------------------------------
# 4. Load FAISS Database
# -----------------------------------
index = faiss.read_index(
    "vector_db/faiss.index"
)

print(f"FAISS vectors: {index.ntotal}")


# -----------------------------------
# 5. Hybrid Search
# -----------------------------------
def hybrid_search(query, top_k=5):

    # ===== BM25 Search =====
    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    # ===== Vector Search =====
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    vector_scores, vector_indices = index.search(
        query_embedding,
        len(chunks)
    )

    vector_scores = vector_scores[0]
    vector_indices = vector_indices[0]


    # -----------------------------------
    # Normalize BM25 scores
    # -----------------------------------
    bm25_min = bm25_scores.min()
    bm25_max = bm25_scores.max()

    if bm25_max != bm25_min:
        bm25_normalized = (
            (bm25_scores - bm25_min)
            / (bm25_max - bm25_min)
        )
    else:
        bm25_normalized = np.zeros_like(
            bm25_scores
        )


    # -----------------------------------
    # Normalize Vector scores
    # -----------------------------------
    vector_min = vector_scores.min()
    vector_max = vector_scores.max()

    if vector_max != vector_min:
        vector_normalized = (
            (vector_scores - vector_min)
            / (vector_max - vector_min)
        )
    else:
        vector_normalized = np.zeros_like(
            vector_scores
        )


    # -----------------------------------
    # Combine Scores
    # -----------------------------------
    hybrid_scores = (
        0.5 * bm25_normalized
        + 0.5 * vector_normalized
    )


    # -----------------------------------
    # Get Top-K
    # -----------------------------------
    top_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]


    # -----------------------------------
    # Display Results
    # -----------------------------------
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        print(f"\nRank {rank}")
        print(f"Chunk ID: {idx}")
        print(
            f"BM25 Score: "
            f"{bm25_scores[idx]:.4f}"
        )
        print(
            f"Vector Score: "
            f"{vector_normalized[idx]:.4f}"
        )
        print(
            f"Hybrid Score: "
            f"{hybrid_scores[idx]:.4f}"
        )

        print("\nRetrieved Text:")
        print(
            chunks[idx][:500]
            .replace("\n", " ")
        )

        print("-" * 80)


# -----------------------------------
# 6. Test Query
# -----------------------------------
query = input(
    "\nEnter your question: "
)

hybrid_search(
    query,
    top_k=5
)