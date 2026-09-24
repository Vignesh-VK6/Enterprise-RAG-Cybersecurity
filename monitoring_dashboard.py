import streamlit as st
import json
import os
import pandas as pd

# ============================================================
# MONITORING DASHBOARD
# ============================================================

st.set_page_config(
    page_title="RAG Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(
    BASE_DIR,
    "logs",
    "retrieval_logs.jsonl"
)


# ============================================================
# LOAD LOGS
# ============================================================

def load_logs():

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


logs = load_logs()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Enterprise RAG Monitoring Dashboard")

st.caption(
    "Retrieval performance, query routing, response time and error monitoring"
)


# ============================================================
# NO LOGS
# ============================================================

if not logs:

    st.warning(
        "No monitoring logs found. Run the RAG service first."
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

total_queries = len(logs)


response_times = [
    log.get("response_time_seconds")
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


error_count = sum(
    1
    for log in logs
    if log.get("error")
)


# ============================================================
# ROUTE DISTRIBUTION
# ============================================================

route_counts = {}

for log in logs:

    route = log.get(
        "route",
        "UNKNOWN"
    )

    route_counts[route] = (
        route_counts.get(route, 0) + 1
    )


# ============================================================
# METRICS
# ============================================================

st.subheader("📌 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Queries",
        total_queries
    )

with col2:

    st.metric(
        "Avg Response Time",
        f"{average_response_time:.3f} sec"
    )

with col3:

    st.metric(
        "Avg Retrieval Score",
        f"{average_retrieval_score:.3f}"
    )

with col4:

    st.metric(
        "Error Count",
        error_count
    )


# ============================================================
# ROUTE DISTRIBUTION
# ============================================================

st.subheader("🧭 Query Route Distribution")

if route_counts:

    route_df = pd.DataFrame(
        {
            "Route": list(route_counts.keys()),
            "Queries": list(route_counts.values())
        }
    )

    st.bar_chart(
        route_df.set_index("Route")
    )


# ============================================================
# RESPONSE TIME
# ============================================================

st.subheader("⏱️ Response Time")

if response_times:

    response_df = pd.DataFrame(
        {
            "Query Number": range(
                1,
                len(response_times) + 1
            ),
            "Response Time (seconds)": response_times
        }
    )

    st.line_chart(
        response_df.set_index("Query Number")
    )


# ============================================================
# RETRIEVAL SCORES
# ============================================================

st.subheader("🔎 Retrieval Scores")

if retrieval_scores:

    score_df = pd.DataFrame(
        {
            "Retrieval": range(
                1,
                len(retrieval_scores) + 1
            ),
            "Score": retrieval_scores
        }
    )

    st.line_chart(
        score_df.set_index("Retrieval")
    )


# ============================================================
# RECENT QUERIES
# ============================================================

st.subheader("💬 Recent Queries")

recent_logs = logs[-10:]

recent_data = []

for log in reversed(recent_logs):

    recent_data.append(
        {
            "Timestamp": log.get(
                "timestamp",
                ""
            ),
            "Query": log.get(
                "query",
                ""
            ),
            "Route": log.get(
                "route",
                ""
            ),
            "Response Time": log.get(
                "response_time_seconds",
                ""
            ),
            "Error": log.get(
                "error",
                ""
            )
        }
    )


recent_df = pd.DataFrame(
    recent_data
)

st.dataframe(
    recent_df,
    use_container_width=True
)


# ============================================================
# RETRIEVED DOCUMENTS
# ============================================================

st.subheader("📚 Retrieved Documents")

document_rows = []

for log in logs:

    documents = log.get(
        "retrieved_documents",
        []
    )

    scores = log.get(
        "retrieval_scores",
        []
    )

    for i, document in enumerate(documents):

        score = (
            scores[i]
            if i < len(scores)
            else None
        )

        document_rows.append(
            {
                "Query": log.get(
                    "query",
                    ""
                ),
                "Route": log.get(
                    "route",
                    ""
                ),
                "Document": document,
                "Score": score
            }
        )


if document_rows:

    document_df = pd.DataFrame(
        document_rows
    )

    st.dataframe(
        document_df,
        use_container_width=True
    )

else:

    st.info(
        "No retrieved documents available."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Enterprise RAG System • Retrieval Monitoring Module"
)