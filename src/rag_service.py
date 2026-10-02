import os
import re
import time
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

from src.query_router import classify_query
from src.llm_answer_generation import generate_answer
from src.monitoring import log_retrieval


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "vector_db",
    "chunks.txt"
)

FAISS_INDEX_PATH = os.path.join(
    BASE_DIR,
    "vector_db",
    "onnx_faiss.index"
)


# ============================================================
# BGE ONNX MODEL
# ============================================================

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


# ============================================================
# LOAD CHUNKS
# ============================================================

with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks_text = file.read()


# IMPORTANT:
# chunks.txt uses "--- CHUNK N ---" markers.
chunks = re.split(
    r"(?=--- CHUNK \d+ ---)",
    chunks_text
)

chunks = [
    chunk.strip()
    for chunk in chunks
    if chunk.strip()
]

print(f"Loaded chunks: {len(chunks)}")


# ============================================================
# LOAD FAISS
# ============================================================

faiss_index = faiss.read_index(
    FAISS_INDEX_PATH
)

print(
    f"FAISS vectors: {faiss_index.ntotal}"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading BGE tokenizer...")

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print("Tokenizer loaded.")


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print("Loading BGE ONNX model...")

import onnxruntime as ort

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("BGE ONNX model loaded.")


# ============================================================
# TOKENIZATION FOR BM25
# ============================================================

def tokenize_text(text):
    """
    Convert text into simple word tokens.

    Example:
    "What is information security?"
    ->
    ["what", "is", "information", "security"]
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ============================================================
# BUILD BM25 OVER ALL 630 CHUNKS
# ============================================================

bm25_documents = [
    tokenize_text(chunk)
    for chunk in chunks
]

bm25 = BM25Okapi(
    bm25_documents
)

print(
    f"BM25 documents: {len(bm25_documents)}"
)


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def create_embedding(text):

    encoded = tokenizer.encode(
        text,
        add_special_tokens=True
    )

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

    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(
        "float32"
    )


# ============================================================
# EXTRACT CHUNK ID
# ============================================================

def extract_chunk_id(chunk_text):

    match = re.search(
        r"--- CHUNK (\d+) ---",
        chunk_text
    )

    if match:
        return int(match.group(1))

    return -1


# ============================================================
# FAISS SEARCH
# ============================================================

def faiss_search(question, top_k=20):

    query_embedding = create_embedding(
        question
    )

    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        if index >= len(chunks):
            continue

        results.append(
            {
                "chunk_id": extract_chunk_id(
                    chunks[index]
                ),
                "text": chunks[index],
                "faiss_score": float(score),
                "bm25_score": 0.0,
                "rrf_score": 0.0
            }
        )

    return results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(question, top_k=20):

    query_tokens = tokenize_text(
        question
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in ranked_indices:

        if index >= len(chunks):
            continue

        results.append(
            {
                "chunk_id": extract_chunk_id(
                    chunks[index]
                ),
                "text": chunks[index],
                "faiss_score": 0.0,
                "bm25_score": float(
                    scores[index]
                ),
                "rrf_score": 0.0
            }
        )

    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    faiss_results,
    bm25_results,
    k=60
):

    fused = {}

    # --------------------------------------------------------
    # FAISS RESULTS
    # --------------------------------------------------------

    for rank, item in enumerate(
        faiss_results,
        start=1
    ):

        chunk_id = item["chunk_id"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "text": item["text"],
                "faiss_score": item["faiss_score"],
                "bm25_score": 0.0,
                "rrf_score": 0.0
            }

        fused[chunk_id]["rrf_score"] += (
            1.0 / (k + rank)
        )

    # --------------------------------------------------------
    # BM25 RESULTS
    # --------------------------------------------------------

    for rank, item in enumerate(
        bm25_results,
        start=1
    ):

        chunk_id = item["chunk_id"]

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "text": item["text"],
                "faiss_score": 0.0,
                "bm25_score": item["bm25_score"],
                "rrf_score": 0.0
            }

        else:

            fused[chunk_id]["bm25_score"] = (
                item["bm25_score"]
            )

        fused[chunk_id]["rrf_score"] += (
            1.0 / (k + rank)
        )

    return list(
        fused.values()
    )


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(question):

    """
    Add simple domain terms when useful.

    This does NOT generate an answer.
    It only improves retrieval.
    """

    q = question.lower().strip()

    expansions = []

    if (
        "information security" in q
        or "info security" in q
    ):
        expansions.extend(
            [
                "information security",
                "protect information",
                "protect information systems",
                "confidentiality integrity availability",
                "security requirements",
                "security controls"
            ]
        )

    if "governance" in q:
        expansions.extend(
            [
                "information security governance",
                "security governance",
                "management structure",
                "business objectives",
                "manage risk"
            ]
        )

    if "security awareness" in q:
        expansions.extend(
            [
                "security awareness",
                "awareness program",
                "security knowledge"
            ]
        )

    if "risk management" in q:
        expansions.extend(
            [
                "risk management",
                "risk assessment",
                "identifying analyzing responding to risk"
            ]
        )

    if expansions:

        return question + " " + " ".join(
            expansions
        )

    return question


# ============================================================
# PHRASE BONUS
# ============================================================

def calculate_phrase_bonus(
    question,
    chunk_text
):

    query_tokens = tokenize_text(
        question
    )

    if not query_tokens:
        return 0.0

    chunk_lower = chunk_text.lower()

    normalized_question = " ".join(
        query_tokens
    )

    normalized_chunk = " ".join(
        tokenize_text(chunk_text)
    )

    # Exact normalized phrase
    if normalized_question in normalized_chunk:

        return 0.05

    # Important phrase:
    # information security
    if (
        "information security" in question.lower()
        and "information security" in chunk_lower
    ):

        return 0.03

    matched_tokens = sum(
        1
        for token in query_tokens
        if token in normalized_chunk
    )

    overlap_ratio = (
        matched_tokens /
        len(set(query_tokens))
    )

    return min(
        overlap_ratio * 0.02,
        0.02
    )


# ============================================================
# DEFINITION BONUS
# ============================================================

def calculate_definition_bonus(
    question,
    chunk_text
):

    """
    Give a small retrieval bonus when the question
    asks for a definition and the chunk contains
    definition-style language.

    This does NOT create information.
    """

    question_lower = question.lower()
    chunk_lower = chunk_text.lower()

    definition_question = any(
        phrase in question_lower
        for phrase in [
            "what is",
            "what are",
            "define",
            "definition of",
            "meaning of",
            "refers to"
        ]
    )

    if not definition_question:
        return 0.0

    definition_patterns = [
        "can be defined as",
        "is defined as",
        "are defined as",
        "refers to",
        "is the process of",
        "is a process of",
        "is an aggregate of",
        "constitutes the",
        "is the",
        "is a"
    ]

    for pattern in definition_patterns:

        if pattern in chunk_lower:

            return 0.025

    return 0.0


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieval(
    question,
    top_k=3
):

    print()
    print(
        "========== HYBRID RETRIEVAL =========="
    )

    # --------------------------------------------------------
    # QUERY EXPANSION
    # --------------------------------------------------------

    expanded_query = expand_query(
        question
    )

    if expanded_query != question:

        print(
            "Query expansion applied."
        )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    faiss_results = faiss_search(
        expanded_query,
        top_k=20
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_results = bm25_search(
        expanded_query,
        top_k=20
    )

    print(
        f"FAISS candidates: {len(faiss_results)}"
    )

    print(
        f"BM25 candidates: {len(bm25_results)}"
    )

    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        faiss_results,
        bm25_results
    )

    # --------------------------------------------------------
    # FINAL SCORING
    # --------------------------------------------------------

    for item in fused_results:

        phrase_bonus = calculate_phrase_bonus(
            question,
            item["text"]
        )

        definition_bonus = calculate_definition_bonus(
            question,
            item["text"]
        )

        item["phrase_bonus"] = (
            phrase_bonus
        )

        item["definition_bonus"] = (
            definition_bonus
        )

        item["final_score"] = (
            item["rrf_score"]
            + phrase_bonus
            + definition_bonus
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    fused_results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    final_results = fused_results[
        :top_k
    ]

    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "Top Hybrid Chunks:"
    )

    for rank, item in enumerate(
        final_results,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"Chunk {item['chunk_id']} | "
            f"FAISS: {item['faiss_score']:.4f} | "
            f"BM25: {item['bm25_score']:.4f} | "
            f"RRF: {item['rrf_score']:.6f} | "
            f"Phrase Bonus: {item['phrase_bonus']:.4f} | "
            f"Definition Bonus: {item['definition_bonus']:.4f} | "
            f"Final: {item['final_score']:.6f}"
        )

    print(
        "========================================"
    )

    return final_results


# ============================================================
# CONTEXT CONSTRUCTION
# ============================================================

def build_context(results):

    context_parts = []

    for item in results:

        context_parts.append(
            item["text"]
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# SOURCE INFORMATION
# ============================================================

def build_sources(results):

    sources = []

    for item in results:

        sources.append(
            {
                "source": "pdf 1.pdf",
                "chunk_id": item["chunk_id"],
                "faiss_score": item["faiss_score"],
                "bm25_score": item["bm25_score"],
                "rrf_score": item["rrf_score"],
                "final_score": item["final_score"]
            }
        )

    return sources


# ============================================================
# MAIN RAG SERVICE
# ============================================================

def ask_rag(
    question,
    conversation_history=None
):

    start_time = time.time()

    if conversation_history is None:
        conversation_history = []

    # --------------------------------------------------------
    # QUERY ROUTING
    # --------------------------------------------------------

    router_result = classify_query(
        question,
        conversation_history
    )

    # Support tuple result
    if isinstance(
        router_result,
        tuple
    ):

        route = router_result[0]

        reason = (
            router_result[1]
            if len(router_result) > 1
            else ""
        )

    # Support dictionary result
    elif isinstance(
        router_result,
        dict
    ):

        route = router_result.get(
            "route",
            "DOCUMENT_RETRIEVAL"
        )

        reason = router_result.get(
            "reason",
            ""
        )

    # Support string result
    else:

        route = str(
            router_result
        )

        reason = ""

    print()
    print(
        f"Query: {question}"
    )

    print(
        f"Route: {route}"
    )

    print(
        f"Reason: {reason}"
    )

    # --------------------------------------------------------
    # OUTSIDE KNOWLEDGE BASE
    # --------------------------------------------------------

    if route == "OUTSIDE_KNOWLEDGE_BASE":

        answer = (
            "The information is not available "
            "in the provided documents."
        )

        response_time = (
            time.time() - start_time
        )

        return {
            "answer": answer,
            "route": route,
            "reason": reason,
            "response_time": response_time,
            "sources": [],
            "retrieved_context": ""
        }

    # --------------------------------------------------------
    # CLARIFICATION
    # --------------------------------------------------------

    if route == "CLARIFICATION":

        answer = (
            "Could you please clarify what "
            "you are referring to?"
        )

        response_time = (
            time.time() - start_time
        )

        return {
            "answer": answer,
            "route": route,
            "reason": reason,
            "response_time": response_time,
            "sources": [],
            "retrieved_context": ""
        }

    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    # For now, retrieve from the document knowledge base
    # when a follow-up question needs factual grounding.
    #
    # The conversation history is still passed to the router.

    # --------------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------------

    results = hybrid_retrieval(
        question,
        top_k=3
    )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    retrieved_context = build_context(
        results
    )

    # --------------------------------------------------------
    # GENERATE GROUNDED ANSWER
    # --------------------------------------------------------

    try:

        answer = generate_answer(
            question,
            retrieved_context
        )

    except TypeError:

        # Compatibility fallback if the existing function
        # expects keyword arguments.

        answer = generate_grounded_answer(
            question=question,
            context=retrieved_context
        )

    # --------------------------------------------------------
    # RESPONSE TIME
    # --------------------------------------------------------

    response_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = build_sources(
        results
    )

    # --------------------------------------------------------
    # MONITORING
    # --------------------------------------------------------

    try:

        log_retrieval(
            query=question,
            retrieved_documents=sources,
            response_time=response_time,
            error=None
        )

    except Exception as monitoring_error:

        print(
            "Monitoring warning:",
            monitoring_error
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {
        "answer": answer,
        "route": route,
        "reason": reason,
        "response_time": response_time,
        "sources": sources,
        "retrieved_context": retrieved_context
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_question = (
        "What is information security?"
    )

    result = ask_rag(
        test_question
    )

    print()
    print(
        "========== FINAL ANSWER =========="
    )

    print(
        result["answer"]
    )

    print()
    print(
        "Route:",
        result["route"]
    )

    print(
        "Response Time:",
        f"{result['response_time']:.3f}",
        "seconds"
    )

    print()
    print(
        "Sources:"
    )

    for source in result["sources"]:

        print(
            source
        )