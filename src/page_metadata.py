import os
import json
from pypdf import PdfReader


PDF_PATH = r"C:\Users\ELCOT\OneDrive\Desktop\project of 30days\data\pdf 1.pdf"
OUTPUT_PATH = r"C:\Users\ELCOT\OneDrive\Desktop\project of 30days\data\page_metadata.json"


def extract_page_metadata(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    return pages


if __name__ == "__main__":

    print("=" * 70)
    print("PAGE METADATA EXTRACTION")
    print("=" * 70)

    pages = extract_page_metadata(PDF_PATH)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            pages,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Total pages : {len(pages)}")
    print(f"Output file : {OUTPUT_PATH}")

    print("=" * 70)
    print("PAGE METADATA CREATED SUCCESSFULLY!")
    print("=" * 70)