from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "cleaned_text.txt"
OUTPUT_FILE = BASE_DIR / "data" / "chunks.json"


# --------------------------------------------------
# 2. Read cleaned text
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    text = file.read()

print("=" * 70)
print("TEXT CHUNKING")
print("=" * 70)

print(f"Input file : {INPUT_FILE}")
print(f"Total characters : {len(text):,}")


# --------------------------------------------------
# 3. Create text splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)


# --------------------------------------------------
# 4. Split text into chunks
# --------------------------------------------------

chunks = text_splitter.split_text(text)

print(f"Total chunks : {len(chunks)}")


# --------------------------------------------------
# 5. Add metadata to every chunk
# --------------------------------------------------

chunk_data = []

for i, chunk in enumerate(chunks):

    chunk_data.append({
        "chunk_id": f"DOC001_CHUNK_{i + 1:04d}",
        "document_name": "pdf 1.pdf",
        "organization": "NIST",
        "domain": "Cybersecurity",
        "sub_domain": "Cybersecurity & Privacy",
        "publication_date": "2006-10",
        "chunk_index": i,
        "text": chunk
    })


# --------------------------------------------------
# 6. Save chunks
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(chunk_data, file, indent=4, ensure_ascii=False)


# --------------------------------------------------
# 7. Show sample chunks
# --------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE CHUNKS")
print("=" * 70)

for chunk in chunk_data[:3]:

    print(f"\nChunk ID: {chunk['chunk_id']}")
    print("-" * 50)
    print(chunk["text"][:500])


# --------------------------------------------------
# 8. Completion message
# --------------------------------------------------

print("\n" + "=" * 70)
print("CHUNKING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print(f"Output file : {OUTPUT_FILE}")
print(f"Total chunks: {len(chunk_data)}")