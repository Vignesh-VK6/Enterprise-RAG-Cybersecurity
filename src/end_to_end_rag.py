import os
import re
import time
import numpy as np
import faiss
import onnxruntime as ort

from dotenv import load_dotenv
from google import genai
from transformers import AutoTokenizer


# ============================================================
# END-TO-END RAG WITH PERFORMANCE MONITORING
# ============================================================

load_dotenv()

print("=" * 80)
print("END-TO-END RAG WITH PERFORMANCE MONITORING")
print("=" * 80)


# ============================================================
# 1. GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)


# ============================================================
# 2. FIND BGE MODEL
# ============================================================

hf_cache = os.path.expanduser(
    "~/.cache/huggingface/hub"
)

bge_root = os.path.join(
    hf_cache,
    "models--BAAI--bge-small-en-v1.5",
    "snapshots"
)

if not os.path.exists(bge_root):
    raise FileNotFoundError(
        "BGE model not found in HuggingFace cache."
    )


snapshot_folders = [
    os.path.join(bge_root, folder)
    for folder in os.listdir(bge_root)
    if os.path.isdir(
        os.path.join(bge_root, folder)
    )
]

if not snapshot_folders:
    raise FileNotFoundError(
        "BGE snapshot folder not found."
    )


model_folder = snapshot_folders[0]

print(
    f"\nBGE Model : {model_folder}"
)


# ============================================================
# 3. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_folder
)


# ============================================================
# 4. FIND ONNX MODEL
# ============================================================

onnx_model_path = None

for root, dirs, files in os.walk(
    model_folder
):

    for file in files:

        if file.endswith(".onnx"):

            onnx_model_path = os.path.join(
                root,
                file
            )

            break

    if onnx_model_path:
        break


if not onnx_model_path:

    raise FileNotFoundError(
        "ONNX model not found inside BGE model."
    )


print(
    f"ONNX Model : {onnx_model_path}"
)


# ============================================================
# 5. LOAD ONNX MODEL
# ============================================================

session = ort.InferenceSession(
    onnx_model_path,
    providers=[
        "CPUExecutionProvider"
    ]
)


# ============================================================
# 6. LOAD FAISS INDEX
# ============================================================

faiss_path = os.path.join(
    "vector_db",
    "onnx_faiss.index"
)

chunks_path = os.path.join(
    "vector_db",
    "chunks.txt"
)


if not os.path.exists(
    faiss_path
):

    raise FileNotFoundError(
        f"FAISS index not found: {faiss_path}"
    )


if not os.path.exists(
    chunks_path
):

    raise FileNotFoundError(
        f"Chunks file not found: {chunks_path}"
    )


index = faiss.read_index(
    faiss_path
)


# ============================================================
# 7. LOAD CHUNKS USING ACTUAL FORMAT
# ============================================================

with open(
    chunks_path,
    "r",
    encoding="utf-8"
) as f:

    chunks_text = f.read()


# Actual format:
# --- CHUNK 0 ---
# 0 ---
# content
#
# --- CHUNK 1 ---
# 1 ---
# content


pattern = r"--- CHUNK \d+ ---\s*\d+ ---\s*"

chunks = re.split(
    pattern,
    chunks_text
)


# Remove empty chunks
chunks = [
    chunk.strip()
    for chunk in chunks
    if chunk.strip()
]


print(
    f"\nFAISS Vectors : {index.ntotal}"
)

print(
    f"Loaded Chunks : {len(chunks)}"
)


# ============================================================
# 8. VALIDATION
# ============================================================

if index.ntotal != len(chunks):

    print(
        "\nWARNING:"
        "\nFAISS vectors and chunk count do not match."
    )

    print(
        f"FAISS : {index.ntotal}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )


# ============================================================
# 9. EMBEDDING FUNCTION
# ============================================================

def create_embedding(text):

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    inputs = {
        "input_ids":
            encoded[
                "input_ids"
            ].astype(np.int64),

        "attention_mask":
            encoded[
                "attention_mask"
            ].astype(np.int64)
    }


    if "token_type_ids" in encoded:

        inputs[
            "token_type_ids"
        ] = encoded[
            "token_type_ids"
        ].astype(np.int64)


    outputs = session.run(
        None,
        inputs
    )


    embedding = outputs[0]


    attention_mask = encoded[
        "attention_mask"
    ]


    mask = attention_mask[
        ...,
        None
    ]


    embedding = (
        embedding * mask
    ).sum(
        axis=1
    ) / mask.sum(
        axis=1
    )


    embedding = embedding.astype(
        "float32"
    )


    faiss.normalize_L2(
        embedding
    )


    return embedding


# ============================================================
# 10. USER QUERY
# ============================================================

query = input(
    "\nEnter your question: "
)


# ============================================================
# TOTAL TIMER
# ============================================================

total_start = time.perf_counter()


# ============================================================
# 11. EMBEDDING GENERATION
# ============================================================

embedding_start = time.perf_counter()


query_embedding = create_embedding(
    query
)


embedding_end = time.perf_counter()


embedding_time = (
    embedding_end -
    embedding_start
)


# ============================================================
# 12. FAISS RETRIEVAL
# ============================================================

faiss_start = time.perf_counter()


TOP_K = 10


scores, indices = index.search(
    query_embedding,
    TOP_K
)


retrieved_docs = []


for score, idx in zip(
    scores[0],
    indices[0]
):

    if (
        idx >= 0
        and idx < len(chunks)
    ):

        retrieved_docs.append({

            "chunk":
                chunks[idx],

            "score":
                float(score),

            "index":
                int(idx)
        })


faiss_end = time.perf_counter()


faiss_time = (
    faiss_end -
    faiss_start
)


# ============================================================
# 13. BM25-STYLE RERANKING
# ============================================================

bm25_start = time.perf_counter()


query_words = set(
    query.lower().split()
)


def lexical_score(text):

    words = set(
        text.lower().split()
    )

    return len(
        query_words.intersection(
            words
        )
    )


for doc in retrieved_docs:

    doc[
        "bm25_score"
    ] = lexical_score(
        doc["chunk"]
    )


retrieved_docs.sort(
    key=lambda x: (
        x["bm25_score"],
        x["score"]
    ),
    reverse=True
)


FINAL_K = 2


final_docs = retrieved_docs[
    :FINAL_K
]


bm25_end = time.perf_counter()


bm25_time = (
    bm25_end -
    bm25_start
)


# ============================================================
# 14. BUILD CONTEXT
# ============================================================

context_parts = []


for i, doc in enumerate(
    final_docs
):

    context_parts.append(
        f"""
SOURCE {i + 1}

Chunk ID:
{doc['index']}

FAISS Score:
{doc['score']:.4f}

BM25 Score:
{doc['bm25_score']}

Content:
{doc['chunk']}
"""
    )


context = "\n".join(
    context_parts
)


# ============================================================
# 15. GROUNDED PROMPT
# ============================================================

prompt = f"""
You are a cybersecurity information assistant.

Answer the user's question ONLY using
the provided context.

Do NOT use outside knowledge.

If the answer is not available in the
provided context, say:

"The information is not available in the provided documents."

Keep the answer clear and concise.

User Question:
{query}

Retrieved Context:
{context}

Answer:
"""


# ============================================================
# 16. GEMINI LLM
# ============================================================

llm_start = time.perf_counter()


response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)


llm_end = time.perf_counter()


llm_time = (
    llm_end -
    llm_start
)


# ============================================================
# 17. TOKEN USAGE
# ============================================================

print("\n" + "=" * 80)
print("TOKEN USAGE")
print("=" * 80)


if response.usage_metadata:

    prompt_tokens = (
        response
        .usage_metadata
        .prompt_token_count
    )

    output_tokens = (
        response
        .usage_metadata
        .candidates_token_count
    )

    total_tokens = (
        response
        .usage_metadata
        .total_token_count
    )


    print(
        f"Prompt Tokens  : "
        f"{prompt_tokens}"
    )

    print(
        f"Output Tokens  : "
        f"{output_tokens}"
    )

    print(
        f"Total Tokens   : "
        f"{total_tokens}"
    )

else:

    print(
        "Token usage information "
        "not available."
    )


# ============================================================
# 18. FINAL ANSWER
# ============================================================

answer = response.text


# ============================================================
# 19. TOTAL RESPONSE TIME
# ============================================================

total_end = time.perf_counter()


total_time = (
    total_end -
    total_start
)


# ============================================================
# 20. DISPLAY ANSWER
# ============================================================

print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(answer)


# ============================================================
# 21. SOURCES
# ============================================================

print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)


for i, doc in enumerate(
    final_docs
):

    print(
        f"\nSource {i + 1}"
    )

    print(
        f"Chunk ID    : "
        f"{doc['index']}"
    )

    print(
        f"FAISS Score : "
        f"{doc['score']:.4f}"
    )

    print(
        f"BM25 Score  : "
        f"{doc['bm25_score']}"
    )


# ============================================================
# 22. PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)


print(
    f"Embedding Generation Time : "
    f"{embedding_time:.4f} seconds"
)

print(
    f"FAISS Retrieval Time      : "
    f"{faiss_time:.4f} seconds"
)

print(
    f"BM25 Reranking Time       : "
    f"{bm25_time:.4f} seconds"
)

print(
    f"LLM Generation Time       : "
    f"{llm_time:.4f} seconds"
)

print(
    f"Total Response Time       : "
    f"{total_time:.4f} seconds"
)

print(
    f"Initial Documents         : "
    f"{len(retrieved_docs)}"
)

print(
    f"Final Documents           : "
    f"{len(final_docs)}"
)

print("=" * 80)