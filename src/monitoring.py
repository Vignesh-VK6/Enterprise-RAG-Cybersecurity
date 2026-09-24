# src/monitoring.py

"""
Retrieval Monitoring and Logging Module

Logs:
1. User query
2. Query route
3. Retrieved documents
4. Retrieval scores
5. Response time
6. Errors
7. Timestamp
"""

import os
import json
from datetime import datetime


# ============================================================
# LOG FILE
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.path.join(BASE_DIR, "logs")

LOG_FILE = os.path.join(
    LOG_DIR,
    "retrieval_logs.jsonl"
)


# ============================================================
# CREATE LOG DIRECTORY
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# CREATE MONITORING LOG
# ============================================================

def log_retrieval(
    query,
    route=None,
    retrieved_documents=None,
    retrieval_scores=None,
    response_time=None,
    error=None
):
    """
    Save one RAG request as a JSON log entry.
    """

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "route": route,
        "retrieved_documents": retrieved_documents or [],
        "retrieval_scores": retrieval_scores or [],
        "response_time_seconds": response_time,
        "error": error
    }

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(log_entry, ensure_ascii=False)
            + "\n"
        )


# ============================================================
# READ ALL LOGS
# ============================================================

def load_logs():
    """
    Load all monitoring logs.
    """

    if not os.path.exists(LOG_FILE):
        return []

    logs = []

    with open(
        LOG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                logs.append(json.loads(line))

            except json.JSONDecodeError:
                continue

    return logs


# ============================================================
# BASIC ANALYTICS
# ============================================================

def get_analytics():
    """
    Calculate basic monitoring statistics.
    """

    logs = load_logs()

    if not logs:

        return {
            "total_queries": 0,
            "average_response_time": 0,
            "average_retrieval_score": 0,
            "error_count": 0
        }

    total_queries = len(logs)

    response_times = [
        log["response_time_seconds"]
        for log in logs
        if isinstance(
            log.get("response_time_seconds"),
            (int, float)
        )
    ]

    retrieval_scores = []

    for log in logs:

        scores = log.get(
            "retrieval_scores",
            []
        )

        if scores:

            retrieval_scores.extend(
                score
                for score in scores
                if isinstance(
                    score,
                    (int, float)
                )
            )

    error_count = sum(
        1
        for log in logs
        if log.get("error")
    )

    average_response_time = (
        sum(response_times) / len(response_times)
        if response_times
        else 0
    )

    average_retrieval_score = (
        sum(retrieval_scores) / len(retrieval_scores)
        if retrieval_scores
        else 0
    )

    return {
        "total_queries": total_queries,
        "average_response_time": average_response_time,
        "average_retrieval_score": average_retrieval_score,
        "error_count": error_count
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RETRIEVAL MONITORING MODULE TEST")
    print("=" * 70)

    # Test log
    log_retrieval(
        query="What is information security?",
        route="DOCUMENT_RETRIEVAL",
        retrieved_documents=[
            "chunk_101",
            "chunk_76",
            "chunk_287"
        ],
        retrieval_scores=[
            8.52,
            7.95,
            7.41
        ],
        response_time=1.42,
        error=None
    )

    print("\nLog created successfully.")

    print("\nLog file:")
    print(LOG_FILE)

    # Load logs
    logs = load_logs()

    print("\nTotal logs:", len(logs))

    # Analytics
    analytics = get_analytics()

    print("\n" + "-" * 70)
    print("BASIC ANALYTICS")
    print("-" * 70)

    print(
        "Total Queries :",
        analytics["total_queries"]
    )

    print(
        "Average Response Time :",
        round(
            analytics["average_response_time"],
            3
        ),
        "seconds"
    )

    print(
        "Average Retrieval Score :",
        round(
            analytics["average_retrieval_score"],
            3
        )
    )

    print(
        "Error Count :",
        analytics["error_count"]
    )

    print("\n" + "=" * 70)
    print("MONITORING TEST COMPLETED")
    print("=" * 70)