"""
Enterprise RAG Retrieval Benchmark

Evaluates the retrieval pipeline without calling Gemini.

Pipeline:
Question
   ↓
Query Router
   ↓
BGE Embedding
   ↓
FAISS Top 10
   ↓
BM25 Reranking
   ↓
Top 3 Contexts
   ↓
Retrieval Evaluation

Outputs:
- benchmark_results.csv
- benchmark_summary.json
"""

import os
import sys
import json
import csv
import time
import re
from rank_bm25 import BM25Okapi

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# LOAD EXISTING RAG COMPONENTS
# ============================================================

print()
print("=" * 70)
print("ENTERPRISE RAG RETRIEVAL BENCHMARK")
print("=" * 70)

print()
print("Loading existing RAG components...")

from src.rag_service import (
    chunks,
    index,
    create_embedding,
    bm25
)

from src.query_router import classify_query


print()
print("Loaded chunks:", len(chunks))
print("FAISS vectors:", index.ntotal)


# ============================================================
# FILE PATHS
# ============================================================

QUESTIONS_PATH = os.path.join(
    BASE_DIR,
    "benchmark",
    "benchmark_questions.json"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "benchmark",
    "benchmark_results.csv"
)

SUMMARY_PATH = os.path.join(
    BASE_DIR,
    "benchmark",
    "benchmark_summary.json"
)


# ============================================================
# LOAD QUESTIONS
# ============================================================

print()
print("Loading benchmark questions...")

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as f:

    benchmark_data = json.load(f)


if isinstance(
    benchmark_data,
    list
):

    questions = benchmark_data

elif isinstance(
    benchmark_data,
    dict
):

    questions = benchmark_data.get(
        "questions",
        []
    )

else:

    raise ValueError(
        "Invalid benchmark JSON format."
    )


print(
    "Total benchmark questions:",
    len(questions)
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "what",
    "which",
    "how",
    "why",
    "does",
    "do",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "from",
    "by",
    "as",
    "it",
    "this",
    "that",
    "be",
    "can",
    "used",
    "use"
}


# ============================================================
# KEYWORDS
# ============================================================

def extract_keywords(text):

    normalized = normalize_text(
        text
    )

    words = normalized.split()

    return {
        word
        for word in words
        if len(word) >= 4
        and word not in STOPWORDS
    }


# ============================================================
# RETRIEVAL RELEVANCE
# ============================================================

def calculate_context_relevance(
    expected_answer,
    context
):

    expected_keywords = extract_keywords(
        expected_answer
    )

    context_keywords = extract_keywords(
        context
    )

    if not expected_keywords:
        return 0.0

    overlap = (
        expected_keywords
        & context_keywords
    )

    score = (
        len(overlap)
        / len(expected_keywords)
    )

    return round(
        score * 100,
        2
    )


# ============================================================
# RUN RETRIEVAL
# ============================================================

def retrieve(
    question
):

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    query_embedding = create_embedding(
        question
    )


    # --------------------------------------------------------
    # FAISS TOP 10
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # BM25 RERANKING
    # --------------------------------------------------------

    candidate_texts = [
        item["text"]
        for item in candidates
    ]

    candidate_tokens = [
        text.lower().split()
        for text in candidate_texts
    ]


    if not candidate_tokens:

        return []


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


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[
            "bm25_score"
        ],
        reverse=True
    )


    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    return candidates[:3]


# ============================================================
# BENCHMARK
# ============================================================

results = []

total_retrieval_success = 0
total_top1_success = 0

total_relevance = 0.0
total_latency = 0.0

route_document_count = 0
route_outside_count = 0
error_count = 0


print()
print("=" * 70)
print("STARTING RETRIEVAL BENCHMARK")
print("=" * 70)


for number, item in enumerate(
    questions,
    start=1
):

    question_id = item.get(
        "id",
        number
    )

    question = item.get(
        "question",
        ""
    )

    expected_answer = item.get(
        "expected_answer",
        ""
    )


    print()
    print("-" * 70)

    print(
        f"QUESTION {number}/{len(questions)}"
    )

    print(
        "ID:",
        question_id
    )

    print(
        "Question:",
        question
    )

    print("-" * 70)


    start_time = time.time()


    try:

        # ----------------------------------------------------
        # ROUTER
        # ----------------------------------------------------

        route, reason = classify_query(
            question,
            conversation_history=None
        )


        if route == "DOCUMENT_RETRIEVAL":

            route_document_count += 1

        elif route == "OUTSIDE_KNOWLEDGE_BASE":

            route_outside_count += 1


        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        retrieved = []


        if route == "DOCUMENT_RETRIEVAL":

            retrieved = retrieve(
                question
            )


        latency = (
            time.time()
            - start_time
        )


        # ----------------------------------------------------
        # RELEVANCE SCORES
        # ----------------------------------------------------

        relevance_scores = []

        for result in retrieved:

            score = calculate_context_relevance(
                expected_answer,
                result["text"]
            )

            relevance_scores.append(
                score
            )


        # ----------------------------------------------------
        # TOP 1 / TOP 3 SUCCESS
        # ----------------------------------------------------

        top1_success = 0
        top3_success = 0


        if relevance_scores:

            if relevance_scores[0] >= 20:

                top1_success = 1


            if max(
                relevance_scores
            ) >= 20:

                top3_success = 1


        total_top1_success += (
            top1_success
        )

        total_retrieval_success += (
            top3_success
        )


        if relevance_scores:

            best_relevance = max(
                relevance_scores
            )

        else:

            best_relevance = 0.0


        total_relevance += (
            best_relevance
        )

        total_latency += (
            latency
        )


        # ----------------------------------------------------
        # CHUNK IDS
        # ----------------------------------------------------

        chunk_ids = [
            str(
                result["chunk_id"]
            )
            for result in retrieved
        ]


        # ----------------------------------------------------
        # BM25 SCORES
        # ----------------------------------------------------

        bm25_scores = [
            round(
                result["bm25_score"],
                4
            )
            for result in retrieved
        ]


        # ----------------------------------------------------
        # SAVE ROW
        # ----------------------------------------------------

        row = {
            "id": question_id,
            "question": question,
            "expected_answer": expected_answer,
            "route": route,
            "route_reason": reason,
            "retrieved_chunks": ",".join(
                chunk_ids
            ),
            "bm25_scores": ",".join(
                map(
                    str,
                    bm25_scores
                )
            ),
            "top1_relevance_percent": (
                round(
                    relevance_scores[0],
                    2
                )
                if relevance_scores
                else 0
            ),
            "best_top3_relevance_percent": (
                round(
                    best_relevance,
                    2
                )
            ),
            "top1_success": top1_success,
            "top3_success": top3_success,
            "retrieval_latency_seconds": round(
                latency,
                4
            ),
            "error": ""
        }


        results.append(
            row
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            "Route:",
            route
        )

        print(
            "Retrieved Chunks:",
            chunk_ids
        )

        print(
            "Top-1 Relevance:",
            f"{row['top1_relevance_percent']}%"
        )

        print(
            "Best Top-3 Relevance:",
            f"{row['best_top3_relevance_percent']}%"
        )

        print(
            "Top-1 Success:",
            top1_success
        )

        print(
            "Top-3 Success:",
            top3_success
        )

        print(
            "Retrieval Latency:",
            f"{latency:.4f} seconds"
        )


    except Exception as error:

        latency = (
            time.time()
            - start_time
        )

        error_count += 1


        print()
        print(
            "BENCHMARK ERROR:",
            str(error)
        )


        results.append(
            {
                "id": question_id,
                "question": question,
                "expected_answer": expected_answer,
                "route": "",
                "route_reason": "",
                "retrieved_chunks": "",
                "bm25_scores": "",
                "top1_relevance_percent": 0,
                "best_top3_relevance_percent": 0,
                "top1_success": 0,
                "top3_success": 0,
                "retrieval_latency_seconds": round(
                    latency,
                    4
                ),
                "error": str(error)
            }
        )


# ============================================================
# SAVE CSV
# ============================================================

print()
print("=" * 70)
print("SAVING RETRIEVAL RESULTS")
print("=" * 70)


fieldnames = [
    "id",
    "question",
    "expected_answer",
    "route",
    "route_reason",
    "retrieved_chunks",
    "bm25_scores",
    "top1_relevance_percent",
    "best_top3_relevance_percent",
    "top1_success",
    "top3_success",
    "retrieval_latency_seconds",
    "error"
]


with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# FINAL METRICS
# ============================================================

total_questions = len(
    questions
)


top1_success_rate = (
    total_top1_success
    / total_questions
    * 100
)


top3_success_rate = (
    total_retrieval_success
    / total_questions
    * 100
)


average_relevance = (
    total_relevance
    / total_questions
)


average_latency = (
    total_latency
    / total_questions
)


document_route_rate = (
    route_document_count
    / total_questions
    * 100
)


outside_route_rate = (
    route_outside_count
    / total_questions
    * 100
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "benchmark_name":
        "Enterprise RAG Retrieval Benchmark",

    "total_questions":
        total_questions,

    "successful_runs":
        total_questions - error_count,

    "error_count":
        error_count,

    "document_retrieval_routes":
        route_document_count,

    "outside_knowledge_base_routes":
        route_outside_count,

    "document_route_rate_percent":
        round(
            document_route_rate,
            2
        ),

    "outside_route_rate_percent":
        round(
            outside_route_rate,
            2
        ),

    "top1_retrieval_success_rate_percent":
        round(
            top1_success_rate,
            2
        ),

    "top3_retrieval_success_rate_percent":
        round(
            top3_success_rate,
            2
        ),

    "average_best_top3_relevance_percent":
        round(
            average_relevance,
            2
        ),

    "average_retrieval_latency_seconds":
        round(
            average_latency,
            4
        ),

    "results_file":
        "benchmark/benchmark_results.csv"
}


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print()
print("=" * 70)
print("RETRIEVAL BENCHMARK COMPLETED")
print("=" * 70)

print()

print(
    "Total Questions                 :",
    total_questions
)

print(
    "Successful Runs                 :",
    total_questions - error_count
)

print(
    "Errors                          :",
    error_count
)

print(
    "Document Retrieval Routes       :",
    route_document_count
)

print(
    "Outside Knowledge Routes        :",
    route_outside_count
)

print(
    "Top-1 Retrieval Success Rate    :",
    f"{top1_success_rate:.2f}%"
)

print(
    "Top-3 Retrieval Success Rate    :",
    f"{top3_success_rate:.2f}%"
)

print(
    "Average Top-3 Relevance         :",
    f"{average_relevance:.2f}%"
)

print(
    "Average Retrieval Latency       :",
    f"{average_latency:.4f} seconds"
)

print()
print(
    "Results:",
    RESULTS_PATH
)

print(
    "Summary:",
    SUMMARY_PATH
)

print()
print("=" * 70)
print("RETRIEVAL BENCHMARK FINISHED")
print("=" * 70)