import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


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
print("MULTI-QUERY RETRIEVAL")
print("=" * 80)

print(f"Total chunks: {len(chunks)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
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
# ORIGINAL QUERY
# ============================================================

original_query = "What is information security risk management?"


# ============================================================
# QUERY VARIANTS
# ============================================================

query_variants = [
    original_query,

    "How does information security risk management work?",

    "What are the processes involved in information security risk management?",

    "How should an organization manage information security risks?",

    "What is the purpose of risk management in information security?"
]


# ============================================================
# DISPLAY QUERIES
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL QUERY + QUERY VARIANTS")
print("=" * 80)

for number, query in enumerate(query_variants, start=1):
    print(f"\nQuery {number}: {query}")


# ============================================================
# MULTI-QUERY RETRIEVAL
# ============================================================

all_results = []


for query_number, query in enumerate(query_variants, start=1):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        5
    )

    print("\n" + "-" * 80)
    print(f"RETRIEVAL FOR QUERY {query_number}")
    print("-" * 80)

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1
    ):

        if idx < 0 or idx >= len(chunks):
            continue

        print(
            f"Rank {rank} | "
            f"Chunk {idx} | "
            f"Score {score:.4f}"
        )

        all_results.append(
            (idx, score, query_number)
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("COMBINING RESULTS")
print("=" * 80)

# Keep best score for each unique chunk

unique_results = {}

for idx, score, query_number in all_results:

    if idx not in unique_results:
        unique_results[idx] = {
            "score": float(score),
            "query_number": query_number
        }

    else:
        if score > unique_results[idx]["score"]:
            unique_results[idx] = {
                "score": float(score),
                "query_number": query_number
            }


# ============================================================
# SORT COMBINED RESULTS
# ============================================================

combined_results = sorted(
    unique_results.items(),
    key=lambda x: x[1]["score"],
    reverse=True
)


# ============================================================
# FINAL TOP RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL MULTI-QUERY RESULTS")
print("=" * 80)

top_results = combined_results[:10]

for rank, (idx, information) in enumerate(
    top_results,
    start=1
):

    print(
        f"Rank {rank} | "
        f"Chunk {idx} | "
        f"Best Score {information['score']:.4f} | "
        f"From Query {information['query_number']}"
    )


# ============================================================
# TOP 3 CONTEXTS
# ============================================================

print("\n" + "=" * 80)
print("TOP 3 CONTEXTS FROM MULTI-QUERY RETRIEVAL")
print("=" * 80)

for rank, (idx, information) in enumerate(
    combined_results[:3],
    start=1
):

    print(
        f"\nRank {rank} | "
        f"Chunk {idx} | "
        f"Score {information['score']:.4f}"
    )

    print("-" * 80)

    print(chunks[idx][:500])

    print("...")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MULTI-QUERY RETRIEVAL COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(f"Original Query      : 1")
print(f"Query Variants      : {len(query_variants)}")
print(f"Results per Query   : 5")
print(f"Combined Unique     : {len(combined_results)}")
print(f"Final Contexts      : 3")