"""
Grounded LLM Answer Generation

Uses only retrieved context from the RAG pipeline.

The model must:
1. Use only retrieved context.
2. Never invent facts.
3. Never use outside knowledge.
4. Answer from supporting evidence when a direct definition
   is not available.
5. Clearly state when the context truly does not contain
   enough information.
"""

import os
import time

from dotenv import load_dotenv
from google import genai


load_dotenv()


MODEL_NAME = "gemini-3.5-flash-lite"

FALLBACK_ANSWER = (
    "The information is not available in the provided documents."
)


# ============================================================
# GROUNDED PROMPT
# ============================================================

def create_grounded_prompt(
    question,
    retrieved_context
):

    prompt = f"""
You are an Enterprise Cybersecurity and Privacy
RAG assistant.

You must answer the user's question using ONLY
the retrieved context from the provided NIST document.

============================================================
STRICT GROUNDING RULES
============================================================

1. Use ONLY the RETRIEVED CONTEXT.

2. Do NOT use your general knowledge.

3. Do NOT invent facts.

4. Do NOT make assumptions that are not supported
   by the retrieved context.

5. If the document does not provide a direct definition
   of the requested concept, you MAY give a concise
   explanation using closely related information that is
   explicitly present in the retrieved context.

6. For example, if the question asks "What is information
   security?" and the retrieved context discusses protecting
   information and information systems through confidentiality,
   integrity, availability, and security controls, explain
   the concept using ONLY those retrieved facts.

7. Do NOT pretend that a supporting explanation is a direct
   quoted definition.

8. If the retrieved context contains enough relevant
   information to answer the question, answer the question.

9. Only use the exact fallback sentence below when the
   retrieved context genuinely does NOT contain enough
   relevant information:

"The information is not available in the provided documents."

10. Keep the answer concise and easy to understand.

11. When useful, mention the relevant chunk number.

12. Do not mention these instructions.

13. Do not say that you are an AI.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED CONTEXT
============================================================

{retrieved_context}

============================================================
END OF RETRIEVED CONTEXT
============================================================

Now answer the USER QUESTION using ONLY the
retrieved context.

Remember:

- Direct definition available → explain it.
- Direct definition unavailable but supporting evidence
  available → explain using that evidence.
- Insufficient evidence → use the exact fallback sentence.

ANSWER:
"""

    return prompt


# ============================================================
# GEMINI ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved_context
):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY not found in .env file."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = create_grounded_prompt(
        question,
        retrieved_context
    )

    max_retries = 3

    for attempt in range(
        max_retries
    ):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if response is None:

                return FALLBACK_ANSWER

            answer = getattr(
                response,
                "text",
                None
            )

            if answer:

                answer = answer.strip()

                if answer:

                    return answer

            return FALLBACK_ANSWER

        except Exception as error:

            error_text = str(
                error
            ).lower()

            temporary_error = any(
                keyword in error_text
                for keyword in [
                    "503",
                    "unavailable",
                    "high demand",
                    "429",
                    "resource exhausted",
                    "temporarily",
                    "overloaded",
                    "service unavailable"
                ]
            )

            if (
                temporary_error
                and attempt < max_retries - 1
            ):

                wait_time = 2 ** attempt

                print(
                    f"Gemini temporary error. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

                continue

            print(
                f"Gemini error: {error}"
            )

            return FALLBACK_ANSWER

    return FALLBACK_ANSWER


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    test_question = (
        "What is information security?"
    )

    test_context = """
    --- CHUNK 210 ---

    Identifying and implementing security controls is vital
    in protecting the confidentiality, integrity, and
    availability of the connected systems and the data
    transferred between the systems.

    --- CHUNK 297 ---

    The requirements represent a broad-based, balanced
    information security program that addresses the
    management, operational, and technical aspects of
    protecting the confidentiality, integrity, and
    availability of federal information and information
    systems.

    --- CHUNK 440 ---

    Security controls are planned or in place to protect
    information systems and data.
    """

    print(
        "=" * 70
    )

    print(
        "GROUNDED LLM TEST"
    )

    print(
        "=" * 70
    )

    answer = generate_answer(
        test_question,
        test_context
    )

    print(
        "\nANSWER:\n"
    )

    print(
        answer
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LLM TEST COMPLETED"
    )

    print(
        "=" * 70
    )