# ============================================================
# RAG EVALUATION FRAMEWORK
# ============================================================

EVALUATION_DATASET = [
    {
        "question": "What is information security?",
        "expected_keywords": ["information", "security", "protect"]
    },
    {
        "question": "What are the objectives of information security?",
        "expected_keywords": ["confidentiality", "integrity", "availability"]
    },
    {
        "question": "What is risk management?",
        "expected_keywords": ["risk", "management"]
    },
    {
        "question": "Why is access control important?",
        "expected_keywords": ["access", "control", "security"]
    },
    {
        "question": "What is a security policy?",
        "expected_keywords": ["security", "policy"]
    }
]


def get_evaluation_dataset():
    return EVALUATION_DATASET


def calculate_keyword_overlap(text, keywords):

    text = text.lower()

    if not keywords:
        return 0.0

    matched = 0

    for keyword in keywords:
        if keyword.lower() in text:
            matched += 1

    return matched / len(keywords)


def evaluate_context_relevance(context, expected_keywords):

    return calculate_keyword_overlap(
        context,
        expected_keywords
    )


def evaluate_retrieval_precision(contexts, expected_keywords):

    if not contexts:
        return 0.0

    relevant_count = 0

    for context in contexts:

        score = calculate_keyword_overlap(
            context,
            expected_keywords
        )

        if score > 0:
            relevant_count += 1

    return relevant_count / len(contexts)


def evaluate_answer_relevance(answer, question):

    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())

    stop_words = {
        "what", "is", "are", "the",
        "a", "an", "of", "why",
        "how", "does"
    }

    question_words = question_words - stop_words

    if not question_words:
        return 0.0

    matched = question_words.intersection(answer_words)

    return len(matched) / len(question_words)


def evaluate_faithfulness(answer, context):

    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    stop_words = {
        "the", "is", "are", "a",
        "an", "and", "of", "to",
        "in", "from"
    }

    answer_words = answer_words - stop_words

    if not answer_words:
        return 0.0

    supported = answer_words.intersection(context_words)

    return len(supported) / len(answer_words)


# ============================================================
# MAIN EVALUATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("INITIAL RAG EVALUATION")
    print("=" * 60)

    dataset = get_evaluation_dataset()

    evaluation_cases = [

        {
            "context": (
                "Information security protects information "
                "and systems from unauthorized access."
            ),
            "answer": (
                "Information security protects information "
                "and systems from unauthorized access."
            )
        },

        {
            "context": (
                "The objectives of information security include "
                "confidentiality, integrity, and availability."
            ),
            "answer": (
                "The objectives are confidentiality, integrity, "
                "and availability."
            )
        },

        {
            "context": (
                "Risk management identifies, assesses, and manages "
                "risks to information systems."
            ),
            "answer": (
                "Risk management identifies and manages "
                "risks to information systems."
            )
        },

        {
            "context": (
                "Access control helps protect systems by ensuring "
                "only authorized users can access information."
            ),
            "answer": (
                "Access control protects systems by allowing "
                "authorized users to access information."
            )
        },

        {
            "context": (
                "A security policy defines rules and requirements "
                "for protecting organizational information."
            ),
            "answer": (
                "A security policy defines rules for protecting "
                "organizational information."
            )
        }
    ]

    results = []

    for i, item in enumerate(dataset):

        question = item["question"]
        keywords = item["expected_keywords"]

        context = evaluation_cases[i]["context"]
        answer = evaluation_cases[i]["answer"]

        context_score = evaluate_context_relevance(
            context,
            keywords
        )

        retrieval_score = evaluate_retrieval_precision(
            [
                context,
                "Random unrelated text about weather."
            ],
            keywords
        )

        answer_score = evaluate_answer_relevance(
            answer,
            question
        )

        faithfulness_score = evaluate_faithfulness(
            answer,
            context
        )

        results.append({
            "context": context_score,
            "retrieval": retrieval_score,
            "answer": answer_score,
            "faithfulness": faithfulness_score
        })

        print()
        print("Q" + str(i + 1) + ": " + question)

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
    # AVERAGE SCORES
    # ========================================================

    avg_context = sum(
        r["context"] for r in results
    ) / len(results)

    avg_retrieval = sum(
        r["retrieval"] for r in results
    ) / len(results)

    avg_answer = sum(
        r["answer"] for r in results
    ) / len(results)

    avg_faithfulness = sum(
        r["faithfulness"] for r in results
    ) / len(results)

    print()
    print("=" * 60)
    print("AVERAGE RAG EVALUATION SCORES")
    print("=" * 60)

    print(
        "Average Context Relevance   :",
        round(avg_context, 3)
    )

    print(
        "Average Retrieval Precision :",
        round(avg_retrieval, 3)
    )

    print(
        "Average Answer Relevance    :",
        round(avg_answer, 3)
    )

    print(
        "Average Faithfulness        :",
        round(avg_faithfulness, 3)
    )

    print()
    print("Evaluation completed successfully.")