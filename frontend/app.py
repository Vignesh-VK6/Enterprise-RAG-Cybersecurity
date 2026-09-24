import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🔐",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔐 Enterprise RAG System")
st.caption("Cybersecurity & Privacy Knowledge Assistant")

st.divider()

# --------------------------------------------------
# SIDEBAR - DOCUMENT MANAGEMENT
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Document Management")

    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("⬆️ Upload Document", use_container_width=True):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            try:

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files
                )

                if response.status_code == 200:

                    st.success("Document uploaded successfully!")

                    st.session_state["uploaded_filename"] = uploaded_file.name

                else:

                    st.error(response.text)

            except requests.exceptions.ConnectionError:

                st.error("❌ FastAPI server is not running.")

    # --------------------------------------------------
    # PROCESS DOCUMENT
    # --------------------------------------------------

    if "uploaded_filename" in st.session_state:

        if st.button(
            "⚙️ Process Document",
            use_container_width=True
        ):

            filename = st.session_state["uploaded_filename"]

            try:

                response = requests.post(
                    f"{API_URL}/process",
                    params={"filename": filename}
                )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Document processed successfully!")

                    st.write(
                        f"📄 Pages: **{result.get('pages')}**"
                    )

                    st.write(
                        f"📝 Characters: **{result.get('characters')}**"
                    )

                else:

                    st.error(response.text)

            except requests.exceptions.ConnectionError:

                st.error("❌ FastAPI server is not running.")

    st.divider()

    # --------------------------------------------------
    # API HEALTH
    # --------------------------------------------------

    if st.button(
        "❤️ Check API Health",
        use_container_width=True
    ):

        try:

            response = requests.get(
                f"{API_URL}/health"
            )

            if response.status_code == 200:

                st.success("API is healthy ✅")

            else:

                st.error("API health check failed.")

        except requests.exceptions.ConnectionError:

            st.error("❌ FastAPI server is not running.")

# --------------------------------------------------
# MAIN CHAT
# --------------------------------------------------

st.subheader("💬 Ask Your Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is security certification?"
)

if st.button(
    "🚀 Ask RAG",
    type="primary"
):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question
                }
            )

            if response.status_code == 200:

                result = response.json()

                st.success(
                    "Answer generated successfully!"
                )

                # ------------------------------------------
                # ANSWER
                # ------------------------------------------

                st.markdown("### 🤖 Answer")

                st.write(
                    result.get("answer", "")
                )

                # ------------------------------------------
                # SOURCES
                # ------------------------------------------

                st.markdown("### 📚 Sources")

                sources = result.get(
                    "sources",
                    []
                )

                if sources:

                    for source in sources:

                        st.info(
                            f"**Chunk:** {source.get('chunk_id')}  \n"
                            f"**Source:** {source.get('source')}  \n"
                            f"**FAISS Score:** {source.get('faiss_score')}  \n"
                            f"**BM25 Score:** {source.get('bm25_score')}"
                        )

                else:

                    st.write(
                        "No sources available."
                    )

            else:

                st.error(
                    f"API Error: {response.status_code}"
                )

                st.write(
                    response.text
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to FastAPI."
            )

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

st.divider()

st.subheader("🕘 Conversation History")

if st.button(
    "🔄 Load Chat History"
):

    try:

        response = requests.get(
            f"{API_URL}/chat-history"
        )

        if response.status_code == 200:

            history_data = response.json()

            if history_data["count"] == 0:

                st.info(
                    "No conversation history yet."
                )

            else:

                for item in reversed(
                    history_data["history"]
                ):

                    with st.expander(
                        f"❓ {item['question']}"
                    ):

                        st.markdown(
                            "**Answer:**"
                        )

                        st.write(
                            item["answer"]
                        )

        else:

            st.error(
                "Unable to load chat history."
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ FastAPI server is not running."
        )