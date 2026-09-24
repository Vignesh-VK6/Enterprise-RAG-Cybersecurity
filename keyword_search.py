from rank_bm25 import BM25Okapi


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
# Tokenize Chunks
# -----------------------------
tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]


# -----------------------------
# Create BM25 Index
# -----------------------------
bm25 = BM25Okapi(tokenized_chunks)

print(f"Total chunks loaded: {len(chunks)}")
print("BM25 keyword search ready!")


# -----------------------------
# Keyword Search Function
# -----------------------------
def keyword_search(query, top_k=5):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[::-1][:top_k]

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for rank, idx in enumerate(top_indices, start=1):

        print(f"\nRank {rank}")
        print(f"BM25 Score: {scores[idx]:.4f}")
        print(f"Chunk ID: {idx}")

        print("\nRetrieved Text:")
        print(chunks[idx][:500])

        print("-" * 80)


# -----------------------------
# Test Query
# -----------------------------
query = input("\nEnter your question: ")

keyword_search(query, top_k=5)