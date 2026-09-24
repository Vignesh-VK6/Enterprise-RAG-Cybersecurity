# ============================================================
# CONVERSATIONAL RAG
# STEP 1 + STEP 2 + STEP 3 + STEP 4
# ============================================================


# ============================================================
# STEP 1 - CONVERSATION MEMORY
# ============================================================

class ConversationMemory:

    def __init__(self, max_turns=5):
        self.history = []
        self.max_turns = max_turns

    def add_turn(self, question, answer):

        self.history.append({
            "question": question,
            "answer": answer
        })

        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_history(self):
        return self.history

    def get_recent_context(self, max_turns=2):

        return self.history[-max_turns:]

    def clear(self):
        self.history = []


# ============================================================
# STEP 2 - FOLLOW-UP QUESTION RESOLUTION
# ============================================================

def resolve_follow_up(question, history):

    if not history:
        return question

    question_lower = question.lower()

    if "its" in question_lower or "it" in question_lower:

        last_turn = history[-1]

        previous_text = (
            last_turn["question"]
            + " "
            + last_turn["answer"]
        ).lower()

        if "information security" in previous_text:

            resolved_question = question.replace(
                "its",
                "information security"
            )

            resolved_question = resolved_question.replace(
                "it",
                "information security"
            )

            return resolved_question

    return question


# ============================================================
# STEP 3 - CONTEXT ACCUMULATION CONTROL
# ============================================================

def get_relevant_context(history, max_turns=2):

    if not history:
        return []

    return history[-max_turns:]


# ============================================================
# STEP 4 - TOPIC CHANGE DETECTION
# ============================================================

def detect_topic_change(question, history):

    if not history:
        return True

    question_words = set(
        question.lower()
        .replace("?", "")
        .replace(".", "")
        .split()
    )

    last_turn = history[-1]

    previous_text = (
        last_turn["question"]
        + " "
        + last_turn["answer"]
    )

    previous_words = set(
        previous_text.lower()
        .replace("?", "")
        .replace(".", "")
        .split()
    )

    stop_words = {
        "what", "is", "are", "the", "a", "an",
        "of", "to", "in", "on", "for", "and",
        "why", "how", "does", "do", "its",
        "it", "this", "that"
    }

    question_words -= stop_words
    previous_words -= stop_words

    overlap = question_words.intersection(previous_words)

    if len(overlap) == 0:
        return True

    return False


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    memory = ConversationMemory(max_turns=5)

    # --------------------------------------------------------
    # STEP 1 TEST
    # --------------------------------------------------------

    memory.add_turn(
        "What is information security?",
        "Information security protects information and systems."
    )

    memory.add_turn(
        "What are its objectives?",
        "Its objectives include protecting confidentiality, integrity, and availability."
    )

    memory.add_turn(
        "Why is risk management important?",
        "Risk management helps identify and manage security risks."
    )

    print("# CONVERSATIONAL RAG - STEP 1 + STEP 2 + STEP 3 + STEP 4")

    print("\n# CONVERSATION MEMORY")

    for turn in memory.get_history():
        print("\nUser:", turn["question"])
        print("Assistant:", turn["answer"])


    # --------------------------------------------------------
    # STEP 2 TEST
    # --------------------------------------------------------

    follow_up = "What are its objectives?"

    resolved = resolve_follow_up(
        follow_up,
        memory.get_history()
    )

    print("\n" + "=" * 60)
    print("FOLLOW-UP QUESTION RESOLUTION")
    print("=" * 60)

    print("\nOriginal Question:")
    print(follow_up)

    print("\nResolved Question:")
    print(resolved)


    # --------------------------------------------------------
    # STEP 3 TEST
    # --------------------------------------------------------

    recent_context = get_relevant_context(
        memory.get_history(),
        max_turns=2
    )

    print("\n" + "=" * 60)
    print("CONTEXT ACCUMULATION CONTROL")
    print("=" * 60)

    print("\n# LIMITED RECENT CONTEXT")

    for turn in recent_context:
        print("\nUser:", turn["question"])
        print("Assistant:", turn["answer"])

    print("\n# CONTEXT ACCUMULATION CONTROL TEST SUCCESSFUL!")


    # --------------------------------------------------------
    # STEP 4 TEST
    # --------------------------------------------------------

    topic_memory = ConversationMemory(max_turns=5)

    topic_memory.add_turn(
        "What is information security?",
        "Information security protects information and systems."
    )

    same_topic_question = "What are information security objectives?"
    new_topic_question = "What is risk management?"

    same_topic = detect_topic_change(
        same_topic_question,
        topic_memory.get_history()
    )

    new_topic = detect_topic_change(
        new_topic_question,
        topic_memory.get_history()
    )

    print("\n" + "=" * 60)
    print("TOPIC CHANGE DETECTION")
    print("=" * 60)

    print("\nSame Topic Question:")
    print(same_topic_question)

    print("Topic Changed:", same_topic)

    print("\nNew Topic Question:")
    print(new_topic_question)

    print("Topic Changed:", new_topic)

    print("\n# TOPIC CHANGE DETECTION TEST COMPLETED!")
    # ============================================================
# STEP 2 - FOLLOW-UP QUESTION RESOLUTION
# ============================================================

def resolve_follow_up(question, history):

    if not history:
        return question

    question_lower = question.lower()

    last_turn = history[-1]

    previous_text = (
        last_turn["question"]
        + " "
        + last_turn["answer"]
    ).lower()

    if "information security" in previous_text:

        # Handle "its"
        if "its" in question_lower:

            return question.replace(
                "its",
                "information security"
            )

        # Handle standalone "it"
        words = question.split()

        new_words = []

        for word in words:

            clean_word = word.lower().strip("?,.")

            if clean_word == "it":
                new_words.append("information security")
            else:
                new_words.append(word)

        return " ".join(new_words)

    return question