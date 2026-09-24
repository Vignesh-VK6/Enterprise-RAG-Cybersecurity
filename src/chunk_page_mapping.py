import json
import re


PAGE_METADATA_PATH = (
    r"C:\Users\ELCOT\OneDrive\Desktop\project of 30days"
    r"\data\page_metadata.json"
)

CHUNKS_PATH = (
    r"C:\Users\ELCOT\OneDrive\Desktop\project of 30days"
    r"\data\chunks.txt"
)

OUTPUT_PATH = (
    r"C:\Users\ELCOT\OneDrive\Desktop\project of 30days"
    r"\data\chunk_page_mapping.json"
)


def normalize_text(text):

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_pages():

    with open(
        PAGE_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def extract_chunks(text):

    """
    Extract chunks from chunks.txt.

    Handles common formats such as:
    Chunk 1
    Chunk 2
    etc.
    """

    pattern = r"(?i)chunk\s*(\d+)\s*[:\-]?\s*(.*?)(?=chunk\s*\d+\s*[:\-]?|\Z)"

    matches = re.findall(
        pattern,
        text,
        flags=re.DOTALL
    )

    chunks = []

    for chunk_id, chunk_text in matches:

        chunks.append({
            "chunk_id": int(chunk_id),
            "text": chunk_text.strip()
        })

    return chunks


def find_page(chunk_text, pages):

    chunk_text = normalize_text(chunk_text)

    # Remove very short fragments
    sentences = re.split(
        r"[.!?]",
        chunk_text
    )

    candidates = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) >= 50:

            candidates.append(
                sentence[:120]
            )

    candidates.sort(
        key=len,
        reverse=True
    )

    for candidate in candidates:

        for page in pages:

            page_text = normalize_text(
                page["text"]
            )

            if candidate in page_text:

                return page["page_number"]

    return None


if __name__ == "__main__":

    print("=" * 70)
    print("FULL CHUNK → PAGE MAPPING")
    print("=" * 70)

    pages = load_pages()
    chunks_text = open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ).read()

    chunks = extract_chunks(
        chunks_text
    )

    print(f"Pages loaded : {len(pages)}")
    print(f"Chunks detected : {len(chunks)}")

    mappings = []

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        page_number = find_page(
            chunk["text"],
            pages
        )

        mappings.append({
            "chunk_id": chunk["chunk_id"],
            "document_name": "pdf 1.pdf",
            "page_number": page_number
        })

        if i % 50 == 0:

            print(
                f"Processed {i}/{len(chunks)} chunks"
            )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mappings,
            f,
            indent=4,
            ensure_ascii=False
        )

    matched = sum(
        1
        for item in mappings
        if item["page_number"] is not None
    )

    print()
    print("=" * 70)
    print("MAPPING SUMMARY")
    print("=" * 70)

    print(f"Total chunks : {len(mappings)}")
    print(f"Matched      : {matched}")
    print(
        f"Unmatched    : "
        f"{len(mappings) - matched}"
    )

    print()
    print(f"Output file : {OUTPUT_PATH}")

    print("=" * 70)
    print(
        "FULL CHUNK → PAGE MAPPING COMPLETED!"
    )
    print("=" * 70)