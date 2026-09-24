# src/query_router.py

"""
Intelligent Query Router for Enterprise RAG

Routes user queries into:
1. DOCUMENT_RETRIEVAL
2. CONVERSATION_HISTORY
3. CLARIFICATION
4. OUTSIDE_KNOWLEDGE_BASE
"""


# ============================================================
# KNOWLEDGE BASE TOPICS
# ============================================================

KB_KEYWORDS = {
    "information security",
    "security",
    "cybersecurity",
    "risk management",
    "risk assessment",
    "risk mitigation",
    "security controls",
    "security governance",
    "security policy",
    "security awareness",
    "security certification",
    "security accreditation",
    "information system",
    "access control",
    "incident response",
    "contingency planning",
    "business continuity",
    "disaster recovery",
    "system development life cycle",
    "sdlc",
    "privacy",
    "federal information security",
    "fisma",
    "nist",
}


# ============================================================
# OUTSIDE KNOWLEDGE BASE TOPICS
# ============================================================

OUTSIDE_KB_KEYWORDS = {
    "cricket",
    "football",
    "soccer",
    "basketball",
    "movie",
    "movies",
    "actor",
    "actress",
    "celebrity",
    "politics",
    "election",
    "stock market",
    "weather",
    "restaurant",
    "recipe",
    "travel",
    "shopping",
    "bitcoin",
    "cryptocurrency",
}


# ============================================================
# FOLLOW-UP / CONVERSATION WORDS
# ============================================================

FOLLOW_UP_WORDS = {
    "it",
    "its",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "his",
    "her",
    "above",
    "previous",
    "earlier",
    "mentioned",
}


# ============================================================
# VAGUE QUERIES
# ============================================================

VAGUE_QUERIES = {
    "explain that",
    "explain this",
    "tell me more",
    "more details",
    "what about that",
    "what about this",
    "can you explain",
    "please explain",
    "elaborate",
    "continue",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_query(query: str) -> str:
    """
    Normalize query for routing.
    """
    return " ".join(query.lower().strip().split())


def contains_kb_topic(query: str):
    """
    Check whether query contains a knowledge-base topic.
    """
    for keyword in KB_KEYWORDS:
        if keyword in query:
            return keyword

    return None


def contains_outside_topic(query: str):
    """
    Check whether query contains an outside knowledge-base topic.
    """
    for keyword in OUTSIDE_KB_KEYWORDS:
        if keyword in query:
            return keyword

    return None


def contains_follow_up_reference(query: str):
    """
    Check whether query contains a conversational reference
    such as 'its', 'that', 'this', etc.
    """
    words = set(query.split())

    for word in FOLLOW_UP_WORDS:
        if word in words:
            return word

    return None


# ============================================================
# MAIN ROUTER
# ============================================================

def classify_query(query: str, conversation_history=None):
    """
    Classify the user query.

    Returns:
        route
        reason
    """

    normalized_query = normalize_query(query)

    # --------------------------------------------------------
    # 1. EMPTY QUERY
    # --------------------------------------------------------

    if not normalized_query:
        return (
            "CLARIFICATION",
            "The query is empty and needs clarification."
        )

    # --------------------------------------------------------
    # 2. VAGUE QUERY
    # --------------------------------------------------------

    if normalized_query in VAGUE_QUERIES:
        return (
            "CLARIFICATION",
            "The query is too vague and needs clarification."
        )

    # --------------------------------------------------------
    # 3. CONVERSATION FOLLOW-UP
    # --------------------------------------------------------

    follow_up_word = contains_follow_up_reference(normalized_query)

    if follow_up_word and conversation_history:
        return (
            "CONVERSATION_HISTORY",
            f"The query contains a conversational reference: '{follow_up_word}'."
        )

    # --------------------------------------------------------
    # 4. OUTSIDE KNOWLEDGE BASE
    # --------------------------------------------------------

    outside_topic = contains_outside_topic(normalized_query)

    if outside_topic:
        return (
            "OUTSIDE_KNOWLEDGE_BASE",
            f"The query contains a topic outside the current knowledge base: '{outside_topic}'."
        )

    # --------------------------------------------------------
    # 5. KNOWLEDGE BASE
    # --------------------------------------------------------

    kb_topic = contains_kb_topic(normalized_query)

    if kb_topic:
        return (
            "DOCUMENT_RETRIEVAL",
            f"The query matches the knowledge base topic: '{kb_topic}'."
        )

    # --------------------------------------------------------
    # 6. FOLLOW-UP WITHOUT HISTORY
    # --------------------------------------------------------

    if follow_up_word:
        return (
            "CONVERSATION_HISTORY",
            f"The query appears to be a follow-up question containing '{follow_up_word}'."
        )

    # --------------------------------------------------------
    # 7. DEFAULT
    # --------------------------------------------------------

    return (
        "OUTSIDE_KNOWLEDGE_BASE",
        "No relevant knowledge-base topic was detected."
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INTELLIGENT QUERY ROUTER - TEST")
    print("=" * 70)

    test_history = [
        {
            "user": "What is information security?",
            "assistant": "Information security protects information and information systems."
        }
    ]

    test_queries = [
        "What is information security?",
        "What are its objectives?",
        "Explain that",
        "Who won yesterday's cricket match?",
        "What is risk management?",
    ]

    for query in test_queries:

        route, reason = classify_query(
            query,
            conversation_history=test_history
        )

        print("\nQuery :", query)
        print("Route :", route)
        print("Reason:", reason)

    print("\n" + "=" * 70)