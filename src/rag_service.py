"""
Enterprise RAG Service
Cybersecurity & Privacy Knowledge Assistant

Pipeline:
User Query
    ↓
Query Router
    ↓
Conversation Follow-up Resolution
    ↓
Query Expansion
    ↓
Hybrid Retrieval
    ├── FAISS Vector Search
    └── BM25 Keyword Search
    ↓
RRF Fusion
    ↓
Conservative Definition / Relevance Bonus
    ↓
Top-K Context Construction
    ↓
Grounded Gemini Answer
    ↓
Source Citations
    ↓
Monitoring Logs
"""

import os
import re
import time
import glob

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
import onnxruntime as ort
from dotenv import load_dotenv

from src.query_router import classify_query
from src.llm_answer_generation import generate_answer
from src.monitoring import log_retrieval
from src.conversational_rag import resolve_follow_up


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHUNKS_FILE = os.path.join(
    BASE_DIR,
    "vector_db",
    "chunks.txt"
)

FAISS_INDEX_FILE = os.path.join(
    BASE_DIR,
    "vector_db",
    "onnx_faiss.index"
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K_FAISS = 50
TOP_K_BM25 = 50
TOP_K_FINAL = 3

DEFINITION_TOP_K = 10

RRF_K = 60

FALLBACK_ANSWER = (
    "The information is not available in the provided documents."
)

load_dotenv()


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not os.path.exists(CHUNKS_FILE):

        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()

    pattern = r"--- CHUNK \d+ ---"

    parts = re.split(
        pattern,
        content
    )

    chunks = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        chunks.append(part)

    return chunks


chunks = load_chunks()

print(
    f"Loaded chunks: {len(chunks)}"
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

if not os.path.exists(
    FAISS_INDEX_FILE
):

    raise FileNotFoundError(
        f"FAISS index not found: "
        f"{FAISS_INDEX_FILE}"
    )


faiss_index = faiss.read_index(
    FAISS_INDEX_FILE
)

print(
    f"FAISS vectors: {faiss_index.ntotal}"
)


# ============================================================
# LOAD BGE TOKENIZER
# ============================================================

print(
    "Loading BGE tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

print(
    "Tokenizer loaded."
)


# ============================================================
# FIND ONNX MODEL
# ============================================================

def find_onnx_model():

    hf_cache = os.path.expanduser(
        "~/.cache/huggingface/hub"
    )

    search_pattern = os.path.join(
        hf_cache,
        "models--BAAI--bge-small-en-v1.5",
        "snapshots",
        "*",
        "onnx",
        "model.onnx"
    )

    model_files = glob.glob(
        search_pattern
    )

    if not model_files:

        raise FileNotFoundError(
            "BGE ONNX model.onnx "
            "not found in Hugging Face cache."
        )

    return model_files[0]


print(
    "Loading BGE ONNX model..."
)

ONNX_MODEL_PATH = find_onnx_model()

print(
    f"ONNX model path: "
    f"{ONNX_MODEL_PATH}"
)


# ============================================================
# LOAD ONNX SESSION
# ============================================================

onnx_session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)

print(
    "BGE ONNX model loaded."
)

ONNX_INPUTS = [
    input_node.name
    for input_node
    in onnx_session.get_inputs()
]

print(
    f"ONNX inputs: {ONNX_INPUTS}"
)


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):
    """
    Create a normalized 384-dimensional
    BGE embedding.
    """

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )

    model_inputs = {}

    for input_name in ONNX_INPUTS:

        if input_name in encoded:

            model_inputs[
                input_name
            ] = encoded[input_name]

        elif input_name == "token_type_ids":

            model_inputs[
                input_name
            ] = np.zeros_like(
                encoded["input_ids"],
                dtype=np.int64
            )

    outputs = onnx_session.run(
        None,
        model_inputs
    )

    token_embeddings = outputs[0]

    attention_mask = (
        encoded["attention_mask"]
    )

    mask = (
        attention_mask[..., None]
        .astype(np.float32)
    )

    masked_embeddings = (
        token_embeddings * mask
    )

    summed = masked_embeddings.sum(
        axis=1
    )

    counts = np.clip(
        mask.sum(axis=1),
        a_min=1e-9,
        a_max=None
    )

    sentence_embedding = (
        summed / counts
    )

    sentence_embedding = (
        sentence_embedding.astype(
            np.float32
        )
    )

    norm = np.linalg.norm(
        sentence_embedding,
        axis=1,
        keepdims=True
    )

    sentence_embedding = (
        sentence_embedding
        / np.clip(
            norm,
            1e-12,
            None
        )
    )

    return sentence_embedding


# ============================================================
# BM25
# ============================================================

tokenized_documents = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_documents
)

print(
    f"BM25 documents: "
    f"{len(tokenized_documents)}"
)


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(question):
    """
    Lightweight query expansion.

    Keeps the original query and adds
    useful terminology without making
    the query overly broad.
    """

    question_lower = (
        question.lower()
    )

    expansions = []

    if "information security" in question_lower:

        expansions.extend([
            "protect information",
            "information security controls",
            "confidentiality integrity availability",
            "protect information systems"
        ])

    elif "risk management" in question_lower:

        expansions.extend([
            "risk assessment",
            "risk mitigation",
            "threat vulnerability"
        ])

    elif "security awareness" in question_lower:

        expansions.extend([
            "security training",
            "security awareness program",
            "security knowledge"
        ])

    if expansions:

        return (
            question
            + " "
            + " ".join(expansions)
        )

    return question


# ============================================================
# DEFINITION QUESTION DETECTION
# ============================================================

def is_definition_question(question):

    question_lower = (
        question.lower().strip()
    )

    return (
        question_lower.startswith(
            "what is "
        )
        or question_lower.startswith(
            "what are "
        )
        or question_lower.startswith(
            "define "
        )
        or question_lower.startswith(
            "definition of "
        )
    )


# ============================================================
# INFORMATION SECURITY EVIDENCE SCORE
# ============================================================

def information_security_definition_score(
    chunk
):
    """
    Conservative evidence score.

    Does not reward a chunk merely because
    it contains 'information security'.

    Rewards supporting evidence such as:

    - protection of information
    - unauthorized access
    - confidentiality
    - integrity
    - availability
    - security controls
    """

    text = chunk.lower()

    if "information security" not in text:

        return 0.0

    score = 0.0

    protection_patterns = [

        "protect information",
        "protecting information",
        "protection of information",
        "protect information systems",
        "protecting information systems",
        "unauthorized access",
        "unauthorized use",
        "unauthorized disclosure",
        "unauthorized modification",
        "unauthorized destruction",
    ]

    for pattern in protection_patterns:

        if pattern in text:

            score += 0.10

    cia_patterns = [

        "confidentiality",
        "integrity",
        "availability",
    ]

    cia_count = sum(
        1
        for pattern in cia_patterns
        if pattern in text
    )

    if cia_count == 1:

        score += 0.06

    elif cia_count == 2:

        score += 0.12

    elif cia_count >= 3:

        score += 0.18

    context_patterns = [

        "information security program",
        "information security requirements",
        "information security policy",
        "information security controls",
        "security requirements",
        "security controls",
    ]

    for pattern in context_patterns:

        if pattern in text:

            score += 0.04

    return min(
        score,
        1.0
    )


# ============================================================
# DEFINITION BONUS
# ============================================================

def calculate_definition_bonus(
    question,
    chunk
):

    if not is_definition_question(
        question
    ):

        return 0.0

    question_lower = (
        question.lower()
    )

    if "information security" in question_lower:

        evidence_score = (
            information_security_definition_score(
                chunk
            )
        )

        return (
            evidence_score * 0.08
        )

    text = chunk.lower()

    direct_definition_patterns = [

        "is defined as",
        "are defined as",
        "refers to",
        "means",
        "is the process of",
        "is a process",
    ]

    score = 0.0

    for pattern in direct_definition_patterns:

        if pattern in text:

            score += 0.08

    return min(
        score,
        0.15
    )


# ============================================================
# SPECIAL DEFINITION SEARCH
# ============================================================

def definition_search(
    question,
    max_candidates=DEFINITION_TOP_K
):
    """
    Conservative definition search.
    """

    if not is_definition_question(
        question
    ):

        return []

    question_lower = (
        question.lower()
    )

    if "information security" not in question_lower:

        return []

    candidates = []

    for index, chunk in enumerate(
        chunks
    ):

        score = (
            information_security_definition_score(
                chunk
            )
        )

        if score >= 0.20:

            candidates.append(
                (
                    index,
                    score
                )
            )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return candidates[
        :max_candidates
    ]


# ============================================================
# PHRASE BONUS
# ============================================================

def calculate_phrase_bonus(
    question,
    chunk
):

    question_lower = (
        question.lower()
    )

    chunk_lower = (
        chunk.lower()
    )

    bonus = 0.0

    important_phrases = [

        "information security",
        "risk management",
        "security controls",
        "security awareness",
        "security planning",
        "confidentiality",
        "integrity",
        "availability",
    ]

    for phrase in important_phrases:

        if (
            phrase in question_lower
            and phrase in chunk_lower
        ):

            bonus += 0.03

    return min(
        bonus,
        0.12
    )


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieval(question):

    print(
        "\n========== HYBRID RETRIEVAL =========="
    )

    expanded_query = (
        expand_query(question)
    )

    if expanded_query != question:

        print(
            "Query expansion applied."
        )

    # --------------------------------------------------------
    # FAISS VECTOR SEARCH
    # --------------------------------------------------------

    query_embedding = (
        create_embedding(
            expanded_query
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    faiss_distances, faiss_indices = (
        faiss_index.search(
            query_embedding,
            TOP_K_FAISS
        )
    )

    faiss_results = {}

    for rank, (
        index,
        score
    ) in enumerate(
        zip(
            faiss_indices[0],
            faiss_distances[0]
        ),
        start=1
    ):

        index = int(index)

        if (
            index < 0
            or index >= len(chunks)
        ):

            continue

        faiss_results[index] = {

            "rank": rank,

            "score": float(
                score
            )
        }

    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    bm25_query_tokens = (
        question.lower().split()
    )

    bm25_scores_all = (
        bm25.get_scores(
            bm25_query_tokens
        )
    )

    bm25_top_indices = (
        np.argsort(
            bm25_scores_all
        )[::-1][:TOP_K_BM25]
    )

    bm25_results = {}

    for rank, index in enumerate(
        bm25_top_indices,
        start=1
    ):

        index = int(index)

        bm25_results[index] = {

            "rank": rank,

            "score": float(
                bm25_scores_all[index]
            )
        }

    # --------------------------------------------------------
    # DEFINITION SEARCH
    # --------------------------------------------------------

    definition_candidates = (
        definition_search(
            question
        )
    )

    definition_map = {

        index: score

        for index, score
        in definition_candidates
    }

    print(
        f"FAISS candidates: "
        f"{len(faiss_results)}"
    )

    print(
        f"BM25 candidates: "
        f"{len(bm25_results)}"
    )

    print(
        f"Definition candidates: "
        f"{len(definition_candidates)}"
    )

    # --------------------------------------------------------
    # MERGE CANDIDATES
    # --------------------------------------------------------

    candidate_indices = set()

    candidate_indices.update(
        faiss_results.keys()
    )

    candidate_indices.update(
        bm25_results.keys()
    )

    candidate_indices.update(
        definition_map.keys()
    )

    ranked_results = []

    for index in candidate_indices:

        faiss_rank = (
            faiss_results.get(
                index,
                {}
            ).get(
                "rank"
            )
        )

        faiss_score = (
            faiss_results.get(
                index,
                {}
            ).get(
                "score",
                0.0
            )
        )

        bm25_rank = (
            bm25_results.get(
                index,
                {}
            ).get(
                "rank"
            )
        )

        bm25_score = (
            bm25_results.get(
                index,
                {}
            ).get(
                "score",
                0.0
            )
        )

        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        rrf_score = 0.0

        if faiss_rank is not None:

            rrf_score += (
                1.0
                / (
                    RRF_K
                    + faiss_rank
                )
            )

        if bm25_rank is not None:

            rrf_score += (
                1.0
                / (
                    RRF_K
                    + bm25_rank
                )
            )

        # ----------------------------------------------------
        # PHRASE BONUS
        # ----------------------------------------------------

        phrase_bonus = (
            calculate_phrase_bonus(
                question,
                chunks[index]
            )
        )

        # ----------------------------------------------------
        # DEFINITION BONUS
        # ----------------------------------------------------

        definition_bonus = (
            calculate_definition_bonus(
                question,
                chunks[index]
            )
        )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (
            rrf_score
            + phrase_bonus
            + definition_bonus
        )

        ranked_results.append({

            "index": index,

            "faiss_score": faiss_score,

            "bm25_score": bm25_score,

            "rrf_score": rrf_score,

            "phrase_bonus": phrase_bonus,

            "definition_bonus":
                definition_bonus,

            "final_score": final_score
        })

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    ranked_results.sort(
        key=lambda item:
            item["final_score"],
        reverse=True
    )

    top_results = (
        ranked_results[
            :TOP_K_FINAL
        ]
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\nTop Hybrid Chunks:"
    )

    for rank, result in enumerate(
        top_results,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"Chunk {result['index']} | "
            f"FAISS: "
            f"{result['faiss_score']:.4f} | "
            f"BM25: "
            f"{result['bm25_score']:.4f} | "
            f"RRF: "
            f"{result['rrf_score']:.6f} | "
            f"Phrase Bonus: "
            f"{result['phrase_bonus']:.4f} | "
            f"Definition Bonus: "
            f"{result['definition_bonus']:.4f} | "
            f"Final: "
            f"{result['final_score']:.6f}"
        )

    print(
        "========================================\n"
    )

    return top_results


# ============================================================
# CONTEXT CONSTRUCTION
# ============================================================

def build_context(results):

    context_parts = []

    print(
        "========== RETRIEVED CONTEXT =========="
    )

    for result in results:

        index = result["index"]

        chunk = chunks[index]

        context_parts.append(
            f"--- CHUNK {index} ---\n"
            f"{chunk}"
        )

        print(
            f"--- CHUNK {index} ---"
        )

        print(
            chunk[:2500]
        )

        print()

    print(
        "========================================"
    )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# SOURCE CITATIONS
# ============================================================

def build_sources(results):

    sources = []

    for result in results:

        index = result["index"]

        sources.append({

            "chunk": index,

            "score": round(
                result["final_score"],
                6
            ),

            "source": "pdf 1.pdf"
        })

    return sources


# ============================================================
# RETRIEVAL SCORES
# ============================================================

def get_retrieval_scores(results):

    return [

        round(
            result["final_score"],
            6
        )

        for result in results
    ]


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def ask_rag(
    question,
    conversation_history=None
):

    start_time = time.time()

    route = None

    try:

        # ----------------------------------------------------
        # QUERY ROUTER
        # ----------------------------------------------------

        route_result = classify_query(
            question,
            conversation_history
        )

        if isinstance(
            route_result,
            tuple
        ):

            route = route_result[0]

            reason = (
                route_result[1]
                if len(route_result) > 1
                else ""
            )

        elif isinstance(
            route_result,
            dict
        ):

            route = route_result.get(
                "route",
                "DOCUMENT_RETRIEVAL"
            )

            reason = route_result.get(
                "reason",
                ""
            )

        else:

            route = str(
                route_result
            )

            reason = ""

        print(
            f"\nQuery: {question}"
        )

        print(
            f"Route: {route}"
        )

        if reason:

            print(
                f"Reason: {reason}"
            )

        # ----------------------------------------------------
        # OUTSIDE KNOWLEDGE BASE
        # ----------------------------------------------------

        if (
            route
            == "OUTSIDE_KNOWLEDGE_BASE"
        ):

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

                "answer":
                    FALLBACK_ANSWER,

                "route": route,

                "sources": [],

                "retrieval_scores": []
            }

        # ----------------------------------------------------
        # CLARIFICATION
        # ----------------------------------------------------

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

                "answer": (
                    "Could you please clarify "
                    "what you are referring to?"
                ),

                "route": route,

                "sources": [],

                "retrieval_scores": []
            }

        # ----------------------------------------------------
        # FOLLOW-UP RESOLUTION
        # ----------------------------------------------------

        retrieval_question = question

        if conversation_history:

            try:

                retrieval_question = (
                    resolve_follow_up(
                        question,
                        conversation_history
                    )
                )

                print(
                    f"Resolved Question: "
                    f"{retrieval_question}"
                )

            except Exception as resolution_error:

                print(
                    "Follow-up resolution "
                    f"warning: "
                    f"{resolution_error}"
                )

                retrieval_question = question

        # ----------------------------------------------------
        # DOCUMENT RETRIEVAL
        # ----------------------------------------------------

        results = hybrid_retrieval(
            retrieval_question
        )

        if not results:

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

                "answer":
                    FALLBACK_ANSWER,

                "route": route,

                "sources": [],

                "retrieval_scores": [],

                "response_time":
                    response_time
            }

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        retrieved_context = (
            build_context(
                results
            )
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = (
            build_sources(
                results
            )
        )

        retrieval_scores = (
            get_retrieval_scores(
                results
            )
        )

        # ----------------------------------------------------
        # GEMINI GROUNDED ANSWER
        # ----------------------------------------------------

        answer = generate_answer(
            retrieval_question,
            retrieved_context
        )

        # ----------------------------------------------------
        # RESPONSE TIME
        # ----------------------------------------------------

        response_time = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # MONITORING
        # ----------------------------------------------------

        log_retrieval(

            query=question,

            route=route,

            retrieved_documents=[

                f"chunk_{result['index']}"

                for result in results
            ],

            retrieval_scores=
                retrieval_scores,

            response_time=
                response_time,

            error=None
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "answer": answer,

            "route": route,

            "sources": sources,

            "retrieval_scores":
                retrieval_scores,

            "response_time":
                response_time,

            "resolved_question":
                retrieval_question
        }

    except Exception as error:

        response_time = (
            time.time()
            - start_time
        )

        print(
            f"\nRAG ERROR: {error}"
        )

        try:

            log_retrieval(

                query=question,

                route=route,

                retrieved_documents=[],

                retrieval_scores=[],

                response_time=
                    response_time,

                error=str(error)
            )

        except Exception:

            pass

        return {

            "answer":
                FALLBACK_ANSWER,

            "route": route,

            "sources": [],

            "retrieval_scores": [],

            "error": str(error),

            "response_time":
                response_time
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("\n")

    print(
        "=" * 70
    )

    print(
        "ENTERPRISE RAG SERVICE TEST"
    )

    print(
        "=" * 70
    )

    test_question = (
        "What is information security?"
    )

    result = ask_rag(
        test_question
    )

    print("\n")

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
        result.get(
            "answer",
            FALLBACK_ANSWER
        )
    )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "ROUTE"
    )

    print(
        "=" * 70
    )

    print(
        result.get(
            "route"
        )
    )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "SOURCES"
    )

    print(
        "=" * 70
    )

    for source in result.get(
        "sources",
        []
    ):

        print(
            source
        )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "RAG TEST COMPLETED"
    )

    print(
        "=" * 70
    )