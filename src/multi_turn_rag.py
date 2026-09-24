# ============================================================
# MULTI-TURN RAG - STEP 5
# ============================================================

from .conversational_rag import (
    ConversationMemory,
    resolve_follow_up,
    detect_topic_change,
    get_relevant_context
)


# ============================================================
# MULTI-TURN RAG PIPELINE
# ============================================================

class MultiTurnRAG:

    def __init__(self):

        self.memory = ConversationMemory(max_turns=5)

    def process_question(self, question):

        history = self.memory.get_history()

        # ----------------------------------------------------
        # STEP 1 - FOLLOW-UP RESOLUTION
        # ----------------------------------------------------

        resolved_question = resolve_follow_up(
            question,
            history
        )

        # ----------------------------------------------------
        # STEP 2 - TOPIC CHANGE DETECTION
        # ----------------------------------------------------

        topic_changed = detect_topic_change(
            resolved_question,
            history
        )

        # ----------------------------------------------------
        # STEP 3 - CONTEXT CONTROL
        # ----------------------------------------------------

        recent_context = get_relevant_context(
            history,
            max_turns=2
        )

        return {
            "original_question": question,
            "resolved_question": resolved_question,
            "topic_changed": topic_changed,
            "recent_context": recent_context
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    rag = MultiTurnRAG()

    # First question
    result1 = rag.process_question(
        "What is information security?"
    )

    print("\n" + "=" * 60)
    print("MULTI-TURN RAG - STEP 5")
    print("=" * 60)

    print("\nQuestion 1:")
    print(result1["original_question"])

    print("Resolved:")
    print(result1["resolved_question"])

    print("Topic Changed:")
    print(result1["topic_changed"])


    # Add first conversation turn
    rag.memory.add_turn(
        "What is information security?",
        "Information security protects information and systems."
    )


    # Follow-up question
    result2 = rag.process_question(
        "What are its objectives?"
    )

    print("\n" + "=" * 60)
    print("FOLLOW-UP QUESTION")
    print("=" * 60)

    print("\nOriginal:")
    print(result2["original_question"])

    print("\nResolved:")
    print(result2["resolved_question"])

    print("\nTopic Changed:")
    print(result2["topic_changed"])


    print("\n# STEP 5.1 INTEGRATION TEST COMPLETED!")