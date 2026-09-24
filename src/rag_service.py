# src/rag_service.py

"""
Enterprise RAG Service

Pipeline:
1. Intelligent Query Router
2. Query Embedding
3. FAISS Semantic Retrieval
4. BM25 Reranking
5. Top 3 Context Selection
6. Grounded Gemini Answer Generation
7. Source Information
8. Retrieval Monitoring and Logging
"""

import time
import os
import json

import numpy as np
import faiss
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from google import genai

from src.monitoring import log_retrieval
from src.query_router import classify_query

load_dotenv()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "vector_db"
)

FAISS_INDEX_PATH = os.path.join(
    VECTOR_DB_DIR,
    "onnx_faiss.index"
)

CHUNKS_PATH = os.path.join(
    VECTOR_DB_DIR,
    "chunks.txt"
)


# ============================================================
# LOAD CHUNKS
# ============================================================

chunks = []

with open(
    CHUNKS_PATH,
    "r",
    encoding="utf-8"
) as f:

    raw_text = f.read()


parts = raw_text.split(
    "--- CHUNK "
)


for part in parts:

    part = part.strip()

    if not part:
        continue

    lines = part.splitlines()

    if len(lines) <= 1:
        continue

    chunk_text = "\n".join(
        lines[1:]
    ).strip()

    if chunk_text:
        chunks.append(
            chunk_text
        )


print(
    "Loaded chunks:",
    len(chunks)
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

index = faiss.read_index(
    FAISS_INDEX_PATH
)

print(
    "FAISS vectors:",
    index.ntotal
)


# ============================================================
# FIND BGE ONNX MODEL
# ============================================================

HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub"
)

MODEL_FOLDER = None
MODEL_PATH = None
TOKENIZER_PATH = None


for root, dirs, files in os.walk(
    HF_CACHE
):

    if "tokenizer.json" in files:

        onnx_model = None

        for search_root, search_dirs, search_files in os.walk(
            root
        ):

            if "model.onnx" in search_files:

                onnx_model = os.path.join(
                    search_root,
                    "model.onnx"
                )

                break

        if onnx_model:

            MODEL_FOLDER = root

            TOKENIZER_PATH = os.path.join(
                root,
                "tokenizer.json"
            )

            MODEL_PATH = onnx_model

            break


# ============================================================
# MODEL VALIDATION
# ============================================================

if (
    MODEL_FOLDER is None
    or TOKENIZER_PATH is None
    or MODEL_PATH is None
):

    raise FileNotFoundError(
        "BGE ONNX model/tokenizer not found in Hugging Face cache."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print(
    "Loading tokenizer..."
)

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# ============================================================
# LOAD ONNX RUNTIME
# ============================================================

print(
    "Loading BGE ONNX model..."
)

import onnxruntime as ort

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)

print(
    "BGE ONNX model loaded."
)


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(
    text: str
):

    encoded = tokenizer.encode(
        text
    )

    input_ids = np.array(
        [encoded.ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoded.attention_mask],
        dtype=np.int64
    )

    token_type_ids = np.zeros_like(
        input_ids,
        dtype=np.int64
    )

    input_names = [
        item.name
        for item in session.get_inputs()
    ]

    inputs = {}

    if "input_ids" in input_names:

        inputs[
            "input_ids"
        ] = input_ids

    if "attention_mask" in input_names:

        inputs[
            "attention_mask"
        ] = attention_mask

    if "token_type_ids" in input_names:

        inputs[
            "token_type_ids"
        ] = token_type_ids

    outputs = session.run(
        None,
        inputs
    )

    last_hidden_state = outputs[0]

    attention = attention_mask[
        :,
        :,
        None
    ]

    weighted_embeddings = (
        last_hidden_state
        * attention
    )

    summed = (
        weighted_embeddings.sum(
            axis=1
        )
    )

    counts = (
        attention.sum(
            axis=1
        )
    )

    embedding = (
        summed
        / np.clip(
            counts,
            1e-9,
            None
        )
    )

    embedding = embedding.astype(
        "float32"
    )

    norm = np.linalg.norm(
        embedding,
        axis=1,
        keepdims=True
    )

    embedding = (
        embedding
        / np.clip(
            norm,
            1e-12,
            None
        )
    )

    return embedding


# ============================================================
# BM25
# ============================================================

tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

client = None


if API_KEY:

    client = genai.Client(
        api_key=API_KEY
    )

else:

    print(
        "WARNING: GEMINI_API_KEY is not configured."
    )

    print(
        "Router and retrieval tests can still run."
    )


# ============================================================
# GROUNDED ANSWER GENERATION
# ============================================================

def generate_grounded_answer(
    question,
    contexts
):

    if client is None:

        return (
            "Gemini API key is not configured."
        )

    context_text = "\n\n".join(
        contexts
    )

    prompt = f"""
You are an enterprise RAG assistant.

Answer the user's question using ONLY the provided
document context.

Do NOT use outside knowledge.

Do NOT invent information.

If the answer cannot be directly supported by the
provided context, respond exactly with:

The information is not available in the provided documents.

Keep the answer clear and concise.

DOCUMENT CONTEXT:
-----------------
{context_text}
-----------------

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def ask_rag(
    question: str,
    conversation_history=None
):

    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start_time = time.time()

    route = None
    retrieved_documents = []
    retrieval_scores = []

    try:

        # ====================================================
        # 1. QUERY ROUTING
        # ====================================================

        route, reason = classify_query(
            question,
            conversation_history=conversation_history
        )

        print()

        print(
            "=" * 70
        )

        print(
            "INTELLIGENT QUERY ROUTER"
        )

        print(
            "=" * 70
        )

        print(
            "Query :",
            question
        )

        print(
            "Route :",
            route
        )

        print(
            "Reason:",
            reason
        )

        print(
            "=" * 70
        )


        # ====================================================
        # 2. CLARIFICATION
        # ====================================================

        if route == "CLARIFICATION":

            response_time = (
                time.time()
                - start_time
            )

            log_retrieval(
                query=question,
                route=route,
                retrieved_documents=[],
                retrieval_scores=[],
                response_time=response_time,
                error=None
            )

            return {
                "question": question,
                "answer": (
                    "Could you please clarify your question?"
                ),
                "route": route,
                "sources": []
            }


        # ====================================================
        # 3. OUTSIDE KNOWLEDGE BASE
        # ====================================================

        if route == "OUTSIDE_KNOWLEDGE_BASE":

            response_time = (
                time.time()
                - start_time
            )

            log_retrieval(
                query=question,
                route=route,
                retrieved_documents=[],
                retrieval_scores=[],
                response_time=response_time,
                error=None
            )

            return {
                "question": question,
                "answer": (
                    "The question is outside "
                    "the knowledge base."
                ),
                "route": route,
                "sources": []
            }


        # ====================================================
        # 4. CONVERSATION HISTORY
        # ====================================================

        if route == "CONVERSATION_HISTORY":

            history_text = ""

            if conversation_history:

                for turn in conversation_history[-5:]:

                    user_message = turn.get(
                        "user",
                        ""
                    )

                    assistant_message = turn.get(
                        "assistant",
                        ""
                    )

                    history_text += (
                        f"User: {user_message}\n"
                        f"Assistant: {assistant_message}\n\n"
                    )

            response_time = (
                time.time()
                - start_time
            )

            log_retrieval(
                query=question,
                route=route,
                retrieved_documents=[],
                retrieval_scores=[],
                response_time=response_time,
                error=None
            )

            return {
                "question": question,
                "answer": (
                    "This question requires the "
                    "previous conversation context."
                ),
                "route": route,
                "conversation_context": history_text,
                "sources": []
            }


        # ====================================================
        # 5. DOCUMENT RETRIEVAL
        # ====================================================

        query_embedding = create_embedding(
            question
        )


        # ====================================================
        # 6. FAISS TOP 10
        # ====================================================

        faiss_scores, faiss_ids = index.search(
            query_embedding,
            10
        )


        candidates = []


        for i in range(
            len(faiss_ids[0])
        ):

            chunk_id = int(
                faiss_ids[0][i]
            )

            if chunk_id < 0:
                continue

            if chunk_id >= len(chunks):
                continue

            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "faiss_score": float(
                        faiss_scores[0][i]
                    ),
                    "text": chunks[chunk_id]
                }
            )


        # ====================================================
        # 7. BM25 RERANKING
        # ====================================================

        candidate_texts = [
            item["text"]
            for item in candidates
        ]

        candidate_tokens = [
            text.lower().split()
            for text in candidate_texts
        ]

        candidate_bm25 = BM25Okapi(
            candidate_tokens
        )

        query_tokens = (
            question.lower().split()
        )

        bm25_scores = (
            candidate_bm25.get_scores(
                query_tokens
            )
        )


        for i, item in enumerate(
            candidates
        ):

            item[
                "bm25_score"
            ] = float(
                bm25_scores[i]
            )


        # ====================================================
        # 8. BM25 SORT
        # ====================================================

        candidates.sort(
            key=lambda x: x[
                "bm25_score"
            ],
            reverse=True
        )


        # ====================================================
        # 9. TOP 3
        # ====================================================

        top_results = candidates[:3]


        # ====================================================
        # 10. MONITORING DATA
        # ====================================================

        retrieved_documents = [
            f"chunk_{result['chunk_id']}"
            for result in top_results
        ]

        retrieval_scores = [
            result["bm25_score"]
            for result in top_results
        ]


        # ====================================================
        # 11. DEBUG OUTPUT
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            "RETRIEVED TOP 3 CONTEXT"
        )

        print(
            "=" * 70
        )


        for result in top_results:

            print()

            print(
                "Chunk ID     :",
                result["chunk_id"]
            )

            print(
                "FAISS Score  :",
                round(
                    result["faiss_score"],
                    4
                )
            )

            print(
                "BM25 Score   :",
                round(
                    result["bm25_score"],
                    4
                )
            )

            print()

            print(
                "Text:"
            )

            print(
                result["text"][:2000]
            )

            print()

            print(
                "-" * 70
            )


        # ====================================================
        # 12. BUILD CONTEXT
        # ====================================================

        contexts = [
            result["text"]
            for result in top_results
        ]


        # ====================================================
        # 13. GEMINI ANSWER
        # ====================================================

        answer = generate_grounded_answer(
            question,
            contexts
        )


        # ====================================================
        # 14. SOURCES
        # ====================================================

        sources = []


        for result in top_results:

            sources.append(
                {
                    "chunk_id": result[
                        "chunk_id"
                    ],
                    "faiss_score": result[
                        "faiss_score"
                    ],
                    "bm25_score": result[
                        "bm25_score"
                    ],
                    "source": (
                        "NIST - pdf 1.pdf"
                    )
                }
            )


        # ====================================================
        # 15. RESPONSE TIME
        # ====================================================

        response_time = (
            time.time()
            - start_time
        )


        # ====================================================
        # 16. MONITORING LOG
        # ====================================================

        log_retrieval(
            query=question,
            route=route,
            retrieved_documents=retrieved_documents,
            retrieval_scores=retrieval_scores,
            response_time=response_time,
            error=None
        )


        print()

        print(
            "RETRIEVAL MONITORING"
        )

        print(
            "Response Time :",
            round(
                response_time,
                4
            ),
            "seconds"
        )

        print(
            "Documents     :",
            retrieved_documents
        )

        print(
            "Scores        :",
            [
                round(score, 4)
                for score in retrieval_scores
            ]
        )


        # ====================================================
        # 17. FINAL RESULT
        # ====================================================

        return {
            "question": question,
            "answer": answer,
            "route": route,
            "sources": sources,
            "response_time": response_time
        }


    # ========================================================
    # ERROR MONITORING
    # ========================================================

    except Exception as error:

        response_time = (
            time.time()
            - start_time
        )

        error_message = str(
            error
        )

        log_retrieval(
            query=question,
            route=route,
            retrieved_documents=retrieved_documents,
            retrieval_scores=retrieval_scores,
            response_time=response_time,
            error=error_message
        )

        print()

        print(
            "=" * 70
        )

        print(
            "RAG ERROR"
        )

        print(
            "=" * 70
        )

        print(
            "Error:",
            error_message
        )

        print(
            "Response Time:",
            round(
                response_time,
                4
            ),
            "seconds"
        )

        print(
            "=" * 70
        )

        raise


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    result = ask_rag(
        "What is information security?"
    )

    print()

    print(
        "=" * 70
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )