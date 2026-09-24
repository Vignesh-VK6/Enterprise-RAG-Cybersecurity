# ============================================================
# HALLUCINATION CONTROL
# Step 17.1 - Context Grounding + Similarity Threshold
# ============================================================

DEFAULT_SIMILARITY_THRESHOLD = 0.60


def check_similarity_threshold(score, threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """
    Check whether retrieved context is sufficiently similar
    to the user's question.
    """

    if score is None:
        return False

    return score >= threshold


def validate_context_relevance(results, threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """
    Validate whether retrieved contexts are relevant enough
    to answer the user's question.

    results format:
    [
        {"text": "...", "score": 0.75},
        {"text": "...", "score": 0.68}
    ]
    """

    if not results:
        return {
            "relevant": False,
            "reason": "No retrieved context found."
        }

    valid_results = []

    for result in results:
        score = result.get("score")

        if check_similarity_threshold(score, threshold):
            valid_results.append(result)

    if not valid_results:
        return {
            "relevant": False,
            "reason": "Retrieved context is below similarity threshold."
        }

    return {
        "relevant": True,
        "reason": "Relevant context found.",
        "results": valid_results
    }


def get_safe_response(validation_result):
    """
    Return an 'I don't know' response when relevant
    context is not available.
    """

    if not validation_result["relevant"]:
        return (
            "I don't know based on the provided documents. "
            "The available context is not sufficient to answer this question."
        )

    return None