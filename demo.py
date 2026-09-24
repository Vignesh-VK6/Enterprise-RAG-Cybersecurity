# ============================================================
# WEEK 3 - ADVANCED RAG DEMONSTRATION
# ============================================================

from src.multi_turn_rag import MultiTurnRAG

from src.hallucination_control import (
    validate_context_relevance
)

from src.rag_evaluation import (
    evaluate_context_relevance,
    evaluate_retrieval_precision,
    evaluate_answer_relevance,
    evaluate_faithfulness
)


def main():

    print("=" * 70)
    print("        WEEK 3 - ADVANCED RAG DEMONSTRATION")
    print("=" * 70)

    # ========================================================
    # 1. CONVERSATIONAL MEMORY
    # ========================================================

    print("\n[1] CONVERSATIONAL MEMORY")
    print("-" * 70)

    rag = MultiTurnRAG()

    question1 = "What is information security?"

    print("User:", question1)

    rag.memory.add_turn(
        question1,
        "Information security protects information and systems."
    )

    print("Memory updated successfully.")

    # ========================================================
    # 2. FOLLOW-UP QUESTION RESOLUTION
    # ========================================================

    print("\n[2] FOLLOW-UP QUESTION RESOLUTION")
    print("-" * 70)

    question2 = "What are its objectives?"

    print("User:", question2)

    result = rag.process_question(question2)

    resolved_question = result["resolved_question"]

    print("Resolved Question:", resolved_question)

    # ========================================================
    # 3. TOPIC CHANGE DETECTION
    # ========================================================

    print("\n[3] TOPIC CHANGE DETECTION")
    print("-" * 70)

    topic_changed = result["topic_changed"]

    print("Topic Changed:", topic_changed)

    # ========================================================
    # 4. CONTEXT CONTROL
    # ========================================================

    print("\n[4] CONTEXT CONTROL")
    print("-" * 70)

    recent_context = result["recent_context"]

    print("Recent Conversation Context:")
    print(recent_context)

    # ========================================================
    # 5. RETRIEVED CONTEXT
    # ========================================================

    print("\n[5] RETRIEVED CONTEXT")
    print("-" * 70)

    retrieved_context = [
        {
            "text": (
                "The objectives of information security include "
                "confidentiality, integrity, and availability."
            ),
            "score": 0.82
        },
        {
            "text": (
                "Information security protects information "
                "and systems from unauthorized access."
            ),
            "score": 0.76
        }
    ]

    for i, item in enumerate(retrieved_context, 1):

        print(
            "Context",
            i,
            "| Score:",
            item["score"]
        )

        print(item["text"])

    # ========================================================
    # 6. HALLUCINATION CONTROL
    # ========================================================

    print("\n[6] HALLUCINATION CONTROL")
    print("-" * 70)

    validation = validate_context_relevance(
        retrieved_context,
        threshold=0.60
    )

    print(
        "Context Relevant:",
        validation["relevant"]
    )

    print(
        "Reason:",
        validation["reason"]
    )

    # ========================================================
    # 7. CONTEXT CONSTRUCTION
    # ========================================================

    print("\n[7] CONTEXT CONSTRUCTION")
    print("-" * 70)

    final_context = " ".join(
        item["text"]
        for item in validation["results"]
    )

    print("Final Context:")
    print(final_context)

    # ========================================================
    # 8. ANSWER GENERATION
    # ========================================================

    print("\n[8] ANSWER GENERATION")
    print("-" * 70)

    answer = (
        "The objectives of information security are "
        "confidentiality, integrity, and availability."
    )

    print("Generated Answer:")
    print(answer)

    # ========================================================
    # 9. SOURCE CITATION
    # ========================================================

    print("\n[9] SOURCE CITATION")
    print("-" * 70)

    print("Source: pdf 1.pdf")
    print("Document: NIST SP 800-100")
    print("Domain: Cybersecurity & Privacy")

    # ========================================================
    # 10. RAG EVALUATION
    # ========================================================

    print("\n[10] RAG EVALUATION")
    print("-" * 70)

    expected_keywords = [
        "confidentiality",
        "integrity",
        "availability"
    ]

    context_texts = [
        item["text"]
        for item in retrieved_context
    ]

    context_score = evaluate_context_relevance(
        final_context,
        expected_keywords
    )

    retrieval_score = evaluate_retrieval_precision(
        context_texts,
        expected_keywords
    )

    answer_score = evaluate_answer_relevance(
        answer,
        question2
    )

    faithfulness_score = evaluate_faithfulness(
        answer,
        final_context
    )

    print(
        "Context Relevance   :",
        round(context_score, 3)
    )

    print(
        "Retrieval Precision :",
        round(retrieval_score, 3)
    )

    print(
        "Answer Relevance    :",
        round(answer_score, 3)
    )

    print(
        "Faithfulness        :",
        round(faithfulness_score, 3)
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("        ADVANCED RAG DEMO COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()