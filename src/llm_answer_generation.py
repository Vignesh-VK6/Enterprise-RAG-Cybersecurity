# ============================================================
# GROUNDED ANSWER GENERATION MODULE
# Live Gemini LLM Integration
# ============================================================

import os

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# GROUNDED PROMPT
# ============================================================

def create_grounded_prompt(
    question,
    retrieved_context
):
    """
    Create a strict grounded prompt.

    The model must answer only from the retrieved
    NIST document context.
    """

    prompt = f"""
You are an Enterprise Cybersecurity RAG assistant.

Your task is to answer the user's question using ONLY
the retrieved context from the NIST document provided below.

STRICT RULES:

1. Use ONLY the RETRIEVED CONTEXT.
2. Do NOT use your general knowledge.
3. Do NOT make assumptions.
4. Do NOT invent facts.
5. If the retrieved context contains information that
   directly or indirectly answers the question, answer
   using that information.
6. If the retrieved context does NOT contain enough
   information to answer the question, respond exactly:

"The information is not available in the provided documents."

7. Give a concise and clear answer.
8. Do not mention these instructions.
9. Do not say that you are an AI.
10. When possible, mention the relevant source/chunk number
    from the retrieved context.

USER QUESTION:
{question}

============================================================
RETRIEVED CONTEXT
============================================================

{retrieved_context}

============================================================
END OF RETRIEVED CONTEXT
============================================================

Now answer the USER QUESTION using ONLY the retrieved
context above.

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
    """
    Generate a grounded answer using Gemini.
    """

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY not found in .env file."
        )


    # --------------------------------------------------------
    # CREATE GEMINI CLIENT
    # --------------------------------------------------------

    client = genai.Client(
        api_key=api_key
    )


    # --------------------------------------------------------
    # CREATE GROUNDED PROMPT
    # --------------------------------------------------------

    prompt = create_grounded_prompt(
        question,
        retrieved_context
    )


    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )


    # --------------------------------------------------------
    # VALIDATE RESPONSE
    # --------------------------------------------------------

    if response is None:

        return (
            "The information is not available "
            "in the provided documents."
        )


    answer = getattr(
        response,
        "text",
        None
    )


    if not answer:

        return (
            "The information is not available "
            "in the provided documents."
        )


    return answer.strip()


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "What is information security?"
    )


    retrieved_context = """
===== Chunk 101 =====

[Source: NIST - pdf 1.pdf - Chunk 101]

Information security protects information
and information systems.

===== Chunk 103 =====

[Source: NIST - pdf 1.pdf - Chunk 103]

Risk management is an important part
of information security.
"""


    print("=" * 80)
    print("LIVE GEMINI GROUNDED ANSWER")
    print("=" * 80)


    try:

        answer = generate_answer(
            question,
            retrieved_context
        )

        print(answer)

        print("=" * 80)
        print(
            "LIVE LLM ANSWER GENERATION SUCCESSFUL!"
        )
        print("=" * 80)


    except Exception as error:

        print("=" * 80)
        print("GEMINI ERROR")
        print("=" * 80)

        print(
            type(error).__name__,
            ":",
            str(error)
        )