import streamlit as st
from src.rag_service import ask_rag


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Enterprise RAG Chatbot")

st.write(
    "Ask questions from the NIST cybersecurity knowledge base."
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is information security?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🔍 Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            # ------------------------------------------------
            # CALL RAG SYSTEM
            # ------------------------------------------------

            result = ask_rag(
                question,
                conversation_history=st.session_state.conversation_history
            )


            # ------------------------------------------------
            # GET RESULT VALUES
            # ------------------------------------------------

            answer = result.get(
                "answer",
                "No answer was generated."
            )

            route = result.get(
                "route",
                "UNKNOWN"
            )

            response_time = result.get(
                "response_time",
                0
            )

            sources = result.get(
                "sources",
                []
            )

            retrieved_context = result.get(
                "retrieved_context"
            )


            # =================================================
            # ANSWER
            # =================================================

            st.subheader("💬 Answer")

            st.write(answer)


            # =================================================
            # QUERY ROUTE
            # =================================================

            st.subheader("🧭 Query Route")

            st.write(route)


            # =================================================
            # RESPONSE TIME
            # =================================================

            st.subheader("Response Time")

            st.write(
                f"{response_time:.3f} seconds"
            )


            # =================================================
            # SOURCES
            # =================================================

            if sources:

                st.subheader("📚 Sources")

                for source in sources:

                    source_name = source.get(
                        "source",
                        "Unknown"
                    )

                    chunk_id = source.get(
                        "chunk_id",
                        "Unknown"
                    )

                    st.write(
                        f"• NIST - {source_name} | Chunk {chunk_id}"
                    )

            else:

                st.subheader("📚 Sources")

                st.write("No sources returned.")


            # =================================================
            # RETRIEVED CONTEXT
            # =================================================

            st.subheader("🔎 Retrieved Context")

            with st.expander(
                "Click here to view retrieved context",
                expanded=True
            ):

                if retrieved_context:

                    st.text_area(
                        "Retrieved Context",
                        value=str(retrieved_context),
                        height=400
                    )

                else:

                    st.warning(
                        "No retrieved context was returned by the RAG service."
                    )


            # =================================================
            # SAVE CONVERSATION
            # =================================================

            st.session_state.conversation_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )


        except Exception:

            st.error(
                "Sorry, something went wrong while processing "
                "your question. Please try again."
            )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state.conversation_history:

    st.divider()

    st.subheader("💭 Conversation History")

    for item in st.session_state.conversation_history:

        st.markdown(
            f"**You:** {item['question']}"
        )

        st.markdown(
            f"**Assistant:** {item['answer']}"
        )

        st.write("---")