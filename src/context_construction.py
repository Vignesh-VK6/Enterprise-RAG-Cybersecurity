# ============================================================
# WEEK 3 - CONTEXT CONSTRUCTION
# STEP 1: REMOVE DUPLICATE CHUNKS
# ============================================================

print("\n" + "=" * 80)
print("CONTEXT CONSTRUCTION - DUPLICATE REMOVAL")
print("=" * 80)


def remove_duplicate_chunks(results):
    """
    Remove duplicate chunks from reranked results.

    Input:
        results = [(chunk_id, chunk_text, rerank_score), ...]

    Output:
        Unique chunks only
    """

    seen = set()
    unique_results = []

    for idx, chunk, score in results:

        # Normalize text for duplicate checking
        normalized_chunk = " ".join(chunk.lower().split())

        if normalized_chunk not in seen:
            seen.add(normalized_chunk)

            unique_results.append(
                (idx, chunk, score)
            )

    return unique_results


# ============================================================
# TEST DATA
# ============================================================

# For now, use sample data to verify the duplicate-removal
# logic before connecting it to the existing reranking.py.

sample_results = [
    (101, "Information security protects information and systems.", 8.52),
    (102, "Information security protects information and systems.", 8.31),
    (103, "Risk management is an important part of information security.", 7.95),
]


# ============================================================
# REMOVE DUPLICATES
# ============================================================

unique_results = remove_duplicate_chunks(sample_results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nBEFORE DUPLICATE REMOVAL:")
print(f"Total chunks: {len(sample_results)}")

print("\nAFTER DUPLICATE REMOVAL:")
print(f"Unique chunks: {len(unique_results)}")


for rank, (idx, chunk, score) in enumerate(
    unique_results,
    start=1
):
    print("\n" + "-" * 80)
    print(f"Rank       : {rank}")
    print(f"Chunk ID   : {idx}")
    print(f"Score      : {score:.4f}")
    print(f"Content    : {chunk}")


print("\n" + "=" * 80)
print("DUPLICATE REMOVAL COMPLETED SUCCESSFULLY!")
print("=" * 80)
# ============================================================
# STEP 2: SELECT MOST RELEVANT CHUNKS
# ============================================================

def select_relevant_chunks(results, top_k=3):
    """
    Select the most relevant chunks based on reranker score.
    """

    # Highest score first
    sorted_results = sorted(
        results,
        key=lambda x: x[2],
        reverse=True
    )

    # Select Top-K
    return sorted_results[:top_k]


# ============================================================
# SELECT RELEVANT CHUNKS
# ============================================================

relevant_chunks = select_relevant_chunks(
    unique_results,
    top_k=3
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("MOST RELEVANT CHUNKS")
print("=" * 80)

print(f"\nSelected chunks: {len(relevant_chunks)}")

for rank, (idx, chunk, score) in enumerate(
    relevant_chunks,
    start=1
):
    print("\n" + "-" * 80)
    print(f"Rank       : {rank}")
    print(f"Chunk ID   : {idx}")
    print(f"Rerank Score: {score:.4f}")
    print(f"Content    : {chunk}")


print("\n" + "=" * 80)
print("RELEVANT CHUNK SELECTION COMPLETED!")
print("=" * 80)
# ============================================================
# STEP 3: RESPECT TOKEN LIMITATIONS
# ============================================================

def limit_context_tokens(results, max_chars=4000):
    """
    Limit total context size before sending it to the LLM.

    max_chars is used as a simple approximation for token control.
    """

    selected_results = []
    total_chars = 0

    for idx, chunk, score in results:

        chunk_length = len(chunk)

        # Stop if adding this chunk exceeds the limit
        if total_chars + chunk_length > max_chars:
            break

        selected_results.append(
            (idx, chunk, score)
        )

        total_chars += chunk_length

    return selected_results, total_chars


# ============================================================
# APPLY TOKEN / CONTEXT LIMIT
# ============================================================

limited_results, total_chars = limit_context_tokens(
    relevant_chunks,
    max_chars=4000
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("TOKEN LIMITATION")
print("=" * 80)

print(f"\nMaximum context characters : 4000")
print(f"Selected context characters: {total_chars}")
print(f"Selected chunks             : {len(limited_results)}")

for rank, (idx, chunk, score) in enumerate(
    limited_results,
    start=1
):
    print("\n" + "-" * 80)
    print(f"Rank        : {rank}")
    print(f"Chunk ID    : {idx}")
    print(f"Rerank Score: {score:.4f}")
    print(f"Characters  : {len(chunk)}")
    print(f"Content     : {chunk}")


print("\n" + "=" * 80)
print("TOKEN LIMITATION COMPLETED!")
print("=" * 80)
# ============================================================
# STEP 4: MAINTAIN DOCUMENT ORDERING
# ============================================================

def maintain_document_order(results):
    """
    Arrange selected chunks according to their original
    document/chunk ID order.
    """

    ordered_results = sorted(
        results,
        key=lambda x: x[0]
    )

    return ordered_results


# ============================================================
# APPLY DOCUMENT ORDERING
# ============================================================

ordered_results = maintain_document_order(
    limited_results
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("DOCUMENT ORDERING")
print("=" * 80)

for rank, (idx, chunk, score) in enumerate(
    ordered_results,
    start=1
):
    print("\n" + "-" * 80)
    print(f"Document Order : {rank}")
    print(f"Chunk ID       : {idx}")
    print(f"Rerank Score   : {score:.4f}")
    print(f"Content        : {chunk}")


print("\n" + "=" * 80)
print("DOCUMENT ORDERING COMPLETED!")
print("=" * 80)
# ============================================================
# STEP 5: PRESERVE SOURCE INFORMATION
# ============================================================

def add_source_information(results):
    """
    Attach source information to every selected chunk.
    """

    sourced_results = []

    for idx, chunk, score in results:

        source_info = {
            "document_id": "DOC001",
            "document_name": "pdf 1.pdf",
            "organization": "NIST",
            "domain": "Cybersecurity",
            "sub_domain": "Cybersecurity & Privacy",
            "chunk_id": idx
        }

        sourced_results.append(
            {
                "chunk_id": idx,
                "score": float(score),
                "content": chunk,
                "source": source_info
            }
        )

    return sourced_results


# ============================================================
# APPLY SOURCE INFORMATION
# ============================================================

final_context = add_source_information(
    ordered_results
)


# ============================================================
# DISPLAY FINAL CONTEXT
# ============================================================

print("\n" + "=" * 80)
print("SOURCE INFORMATION")
print("=" * 80)

for rank, item in enumerate(
    final_context,
    start=1
):

    print("\n" + "-" * 80)
    print(f"Context Rank : {rank}")
    print(f"Chunk ID     : {item['chunk_id']}")
    print(f"Score        : {item['score']:.4f}")
    print(f"Content      : {item['content']}")

    print("\nSource:")
    print(f"  Document ID   : {item['source']['document_id']}")
    print(f"  Document Name : {item['source']['document_name']}")
    print(f"  Organization  : {item['source']['organization']}")
    print(f"  Domain        : {item['source']['domain']}")
    print(f"  Sub-Domain    : {item['source']['sub_domain']}")
    print(f"  Chunk ID      : {item['source']['chunk_id']}")


print("\n" + "=" * 80)
print("SOURCE INFORMATION PRESERVED SUCCESSFULLY!")
print("=" * 80)