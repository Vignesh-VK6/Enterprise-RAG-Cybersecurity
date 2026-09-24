import streamlit as st
from src.rag_service import ask_rag

# ============================================================
# ENTERPRISE RAG CHATBOT
# ============================================================

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise RAG Chatbot")

st.write(
    "Ask questions from the NIST cybersecurity knowledge base."
)

st.divider()

# ============================================================
# SESSION MEMORY
# ============================================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask your question",
    placeholder="Example: What is risk management?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🔍 Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Processing your question..."):

            try:

                result = ask_rag(
                    question,
                    st.session_state.conversation_history
                )

                # ------------------------------------------------
                # SAVE CONVERSATION
                # ------------------------------------------------

                st.session_state.conversation_history.append(
                    {
                        "user": question,
                        "assistant": result.get(
                            "answer",
                            ""
                        )
                    }
                )

                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                st.subheader("💬 Answer")

                st.write(
                    result.get(
                        "answer",
                        "No answer generated."
                    )
                )

                # ------------------------------------------------
                # ROUTE
                # ------------------------------------------------

                st.subheader("🧭 Query Route")

                st.info(
                    result.get(
                        "route",
                        "UNKNOWN"
                    )
                )

                # ------------------------------------------------
                # RESPONSE TIME
                # ------------------------------------------------

                if result.get("response_time") is not None:

                    st.metric(
                        "Response Time",
                        f"{result['response_time']:.3f} seconds"
                    )

                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                sources = result.get(
                    "sources",
                    []
                )

                if sources:

                    st.subheader("📚 Sources")

                    for source in sources:

                        st.write(
                            f"• {source.get('source', 'Unknown source')} "
                            f"| Chunk {source.get('chunk_id', 'N/A')}"
                        )

                # ------------------------------------------------
                # RETRIEVED CONTEXT
                # ------------------------------------------------

                if result.get("retrieved_context"):

                    with st.expander(
                        "🔎 View Retrieved Context"
                    ):

                        st.write(
                            result["retrieved_context"]
                        )

            except Exception as error:

                st.error(
                    f"An error occurred: {error}"
                )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state.conversation_history:

    st.divider()

    st.subheader("💭 Conversation History")

    for item in reversed(
        st.session_state.conversation_history
    ):

        st.markdown(
            f"**You:** {item['user']}"
        )

        st.markdown(
            f"**Assistant:** {item['assistant']}"
        )

        st.divider()