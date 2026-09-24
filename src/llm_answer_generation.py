# GROUNDED ANSWER GENERATION MODULE
# Live Gemini LLM Integration

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def create_grounded_prompt(question, retrieved_context):
    """
    Create a grounded prompt using only retrieved context.
    """

    prompt = f"""
You are a cybersecurity question-answering assistant.

IMPORTANT RULES:

1. Answer ONLY using the information provided in the
   retrieved context.
2. Do NOT use outside knowledge.
3. Do NOT invent or assume information.
4. If the answer is not available in the context,
   clearly say:
   "The information is not available in the provided documents."
5. Give a clear and concise answer.
6. Include the source information provided with the retrieved context.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{retrieved_context}

ANSWER:
"""

    return prompt


def generate_answer(question, retrieved_context):
    """
    Generate a grounded answer using Gemini.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file."
        )

    client = genai.Client(api_key=api_key)

    prompt = create_grounded_prompt(
        question,
        retrieved_context
    )

    response = client.models.generate_content(
       model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    question = "What is information security?"

    retrieved_context = """
[Source: NIST - pdf 1.pdf - Chunk 101]

Information security protects information and systems.

[Source: NIST - pdf 1.pdf - Chunk 103]

Risk management is an important part of information security.
"""

    answer = generate_answer(
        question,
        retrieved_context
    )

    print("=" * 80)
    print("LIVE GEMINI GROUNDED ANSWER")
    print("=" * 80)

    print(answer)

    print("=" * 80)
    print("LIVE LLM ANSWER GENERATION SUCCESSFUL!")
    print("=" * 80)