import os
import re
import numpy as np
import faiss
import onnxruntime as ort
from tokenizers import Tokenizer


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
print("              ONNX SEMANTIC SEARCH")
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
# LOAD FAISS INDEX
# --------------------------------------------------

print("\nLoading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

print("FAISS vectors:", index.ntotal)
print("Embedding dimension:", index.d)


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
        f"Chunk / Vector mismatch: "
        f"{len(chunks)} chunks vs "
        f"{index.ntotal} vectors"
    )

print("Chunk / Vector mapping: OK")


# --------------------------------------------------
# CREATE QUERY EMBEDDING
# --------------------------------------------------

def create_embedding(text):

    encoded = tokenizer.encode(
        text,
        add_special_tokens=True
    )

    max_length = 512

    input_ids_list = encoded.ids[:max_length]

    attention_mask_list = (
        encoded.attention_mask[:max_length]
    )

    token_type_ids_list = (
        encoded.type_ids[:max_length]
    )

    input_ids = np.array(
        [input_ids_list],
        dtype=np.int64
    )

    attention_mask = np.array(
        [attention_mask_list],
        dtype=np.int64
    )

    token_type_ids = np.array(
        [token_type_ids_list],
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

        embedding = (
            embedding / norm
        )

    return embedding.astype(
        "float32"
    )


# --------------------------------------------------
# USER QUERY
# --------------------------------------------------

print("\n" + "=" * 70)

query = input(
    "\nEnter your question: "
)


print("\nGenerating query embedding...")

query_embedding = create_embedding(
    query
)

query_embedding = np.array(
    [query_embedding],
    dtype="float32"
)

print(
    "Embedding shape:",
    query_embedding.shape
)


# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

top_k = 5

scores, indices = index.search(
    query_embedding,
    top_k
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("              TOP 5 RETRIEVED RESULTS")
print("=" * 70)


for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    print(
        f"\n[{rank}] Score: {score:.4f}"
    )

    print(
        chunks[idx]
    )

    print("-" * 70)


print("\nSEMANTIC SEARCH COMPLETED")