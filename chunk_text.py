from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

input_file = Path("data/cleaned_text.txt")
output_file = Path("data/chunks.txt")

text = input_file.read_text(encoding="utf-8")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

with open(output_file, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"--- CHUNK {i} ---\n")
        f.write(chunk)
        f.write("\n\n")

print(f"Total chunks created: {len(chunks)}")
print(f"Saved to: {output_file}")