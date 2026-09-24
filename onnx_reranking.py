import os
import re
import numpy as np
import faiss
import onnxruntime as ort
from tokenizers import Tokenizer
from rank_bm25 import BM25Okapi


BASE_DIR = r"C:\Users\ELCOT\OneDrive\Desktop\project of 30days"

MODEL_PATH = (
    r"C:\Users\ELCOT\.cache\huggingface\hub"
    r"\models--BAAI--bge-small-en-v1.5"
    r"\snapshots\5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    r"\onnx\model.onnx"
)

TOKENIZER_PATH = (
    r"C:\Users\ELCOT\.cache\huggingface\hub"
    r"\models--BAAI--bge-small-en-v1.5"
    r"\snapshots\5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    r"\tokenizer.json"
)

INDEX_PATH = os.path.join(
    BASE_DIR,
    "vector_db",
    "onnx_faiss.index"
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "vector_db",
    "chunks.txt"
)


print("=" * 70)
print("              ONNX FAISS + BM25 RERANKING")
print("=" * 70)


# --------------------------------------------------
# LOAD TOKENIZER
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# --------------------------------------------------
# LOAD ONNX MODEL
# --------------------------------------------------

print("Loading BGE ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH
)

print("ONNX model loaded.")


# --------------------------------------------------
# LOAD FAISS
# --------------------------------------------------

print("\nLoading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

print("FAISS vectors:", index.ntotal)


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

print("\nLoading chunks...")

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    chunks_text = f.read()


chunks = re.split(
    r"(?=--- CHUNK \d+ ---)",
    chunks_text
)

chunks = [
    chunk.strip()
    for chunk in chunks
    if chunk.strip()
]

print("Chunks loaded:", len(chunks))


if len(chunks) != index.ntotal:

    raise ValueError(
        f"Mismatch: {len(chunks)} chunks vs "
        f"{index.ntotal} vectors"
    )

print("Chunk / Vector mapping: OK")


# --------------------------------------------------
# CREATE EMBEDDING
# --------------------------------------------------

def create_embedding(text):

    encoded = tokenizer.encode(
        text,
        add_special_tokens=True
    )

    max_length = 512

    ids = encoded.ids[:max_length]
    mask_values = encoded.attention_mask[:max_length]
    type_values = encoded.type_ids[:max_length]

    input_ids = np.array(
        [ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [mask_values],
        dtype=np.int64
    )

    token_type_ids = np.array(
        [type_values],
        dtype=np.int64
    )

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
    )

    hidden_state = outputs[0]

    mask = attention_mask[:, :, None]

    pooled = (
        hidden_state * mask
    ).sum(axis=1) / mask.sum(axis=1)

    embedding = pooled[0]

    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    return embedding.astype("float32")


# --------------------------------------------------
# TOKENIZE FOR BM25
# --------------------------------------------------

def tokenize_for_bm25(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# --------------------------------------------------
# USER QUERY
# --------------------------------------------------

print("\n" + "=" * 70)

query = input(
    "\nEnter your question: "
)


# --------------------------------------------------
# INITIAL FAISS RETRIEVAL
# --------------------------------------------------

print("\nGenerating query embedding...")

query_embedding = create_embedding(
    query
)

query_embedding = np.array(
    [query_embedding],
    dtype="float32"
)


initial_top_k = 10

scores, indices = index.search(
    query_embedding,
    initial_top_k
)


print("\n" + "=" * 70)
print("         INITIAL RETRIEVAL - TOP 10")
print("=" * 70)


candidate_chunks = []

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    chunk = chunks[idx]

    candidate_chunks.append(
        {
            "index": int(idx),
            "faiss_score": float(score),
            "text": chunk
        }
    )

    print(
        f"\n[{rank}] "
        f"Chunk: {idx} | "
        f"FAISS Score: {score:.4f}"
    )


# --------------------------------------------------
# BM25 RERANKING
# --------------------------------------------------

print("\n" + "=" * 70)
print("              BM25 RERANKING")
print("=" * 70)


tokenized_candidates = [
    tokenize_for_bm25(item["text"])
    for item in candidate_chunks
]


bm25 = BM25Okapi(
    tokenized_candidates
)


query_tokens = tokenize_for_bm25(
    query
)


bm25_scores = bm25.get_scores(
    query_tokens
)


for i, item in enumerate(
    candidate_chunks
):

    item["bm25_score"] = float(
        bm25_scores[i]
    )


reranked = sorted(
    candidate_chunks,
    key=lambda x: x["bm25_score"],
    reverse=True
)


# --------------------------------------------------
# TOP 3 AFTER RERANKING
# --------------------------------------------------

top_3 = reranked[:3]


print("\n" + "=" * 70)
print("           TOP 3 AFTER RERANKING")
print("=" * 70)


for rank, item in enumerate(
    top_3,
    start=1
):

    print(
        f"\n[{rank}] "
        f"Chunk: {item['index']}"
    )

    print(
        f"FAISS Score : "
        f"{item['faiss_score']:.4f}"
    )

    print(
        f"BM25 Score  : "
        f"{item['bm25_score']:.4f}"
    )

    print("\nContext:")

    print(
        item["text"]
    )

    print("-" * 70)


print("\nRERANKING COMPLETED")