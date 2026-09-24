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

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "vector_db",
    "chunks.txt"
)

OUTPUT_INDEX = os.path.join(
    BASE_DIR,
    "vector_db",
    "onnx_faiss.index"
)


print("=" * 70)
print("       REBUILD FAISS USING BGE ONNX")
print("=" * 70)


print("\nLoading tokenizer...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)


print("Loading BGE ONNX model...")
session = ort.InferenceSession(MODEL_PATH)
print("ONNX model loaded.")


print("\nLoading chunks...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
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

print("Total chunks:", len(chunks))


def create_embedding(text):

    encoded = tokenizer.encode(
        text,
        add_special_tokens=True
    )

    # BGE maximum sequence length
    max_length = 512

    input_ids_list = encoded.ids[:max_length]
    attention_mask_list = encoded.attention_mask[:max_length]
    token_type_ids_list = encoded.type_ids[:max_length]

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
        embedding = embedding / norm

    return embedding.astype("float32")


print("\nCreating embeddings...")

embeddings = []

total = len(chunks)

for i, chunk in enumerate(chunks):

    embedding = create_embedding(chunk)

    embeddings.append(embedding)

    if (i + 1) % 25 == 0 or i == 0:
        print(f"Processed {i + 1}/{total}")


embeddings = np.array(
    embeddings,
    dtype="float32"
)


print("\nEmbedding matrix shape:", embeddings.shape)


dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


print("FAISS vectors created:", index.ntotal)

"faiss.index"(
    index,
    OUTPUT_INDEX
)


print("\nNew FAISS index saved:")
print(OUTPUT_INDEX)


print("\n" + "=" * 70)
print("       ONNX FAISS REBUILD COMPLETED")
print("=" * 70)