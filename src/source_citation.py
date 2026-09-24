import json
import os


# ============================================================
# SOURCE CITATION SYSTEM
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAPPING_PATH = os.path.join(
    BASE_DIR,
    "data",
    "chunk_page_mapping.json"
)


def load_chunk_mapping():
    """Load chunk → page mapping from JSON file."""

    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    return mapping


def get_page_number(chunk_id, mapping):
    """Find the page number for a given chunk ID."""

    for item in mapping:

        if str(item["chunk_id"]) == str(chunk_id):

            return item.get("page_number")

    return None


def create_source_citation(document_name, chunk_id, mapping):
    """Create citation using actual page number."""

    page_number = get_page_number(chunk_id, mapping)

    citation = f"Source: {document_name}"

    if page_number is not None:
        citation += f" — Page {page_number}"

    citation += f" — Chunk {chunk_id}"

    return citation


def attach_citation(answer, document_name, chunk_id, mapping):
    """Attach source citation to the answer."""

    citation = create_source_citation(
        document_name,
        chunk_id,
        mapping
    )

    return f"{answer}\n\n{citation}"


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SOURCE CITATION SYSTEM")
    print("=" * 60)

    # Load mapping
    mapping = load_chunk_mapping()

    print(f"Mapping records loaded : {len(mapping)}")

    # Find first chunk having a real page number
    test_chunk = None

    for item in mapping:

        if item.get("page_number") is not None:

            test_chunk = item["chunk_id"]
            break

    if test_chunk is None:

        print("No page mapping found.")

    else:

        answer = "Information security protects information and systems."

        final_answer = attach_citation(
            answer,
            "pdf 1.pdf",
            test_chunk,
            mapping
        )

        print("\n# CITATION-ENABLED ANSWER")
        print(final_answer)

        print("\n# SOURCE CITATION INTEGRATION SUCCESSFUL!")