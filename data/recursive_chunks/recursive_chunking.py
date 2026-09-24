from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Input file
input_path = "data/cleaned_text.txt"

# Output folder
output_dir = "data/recursive_chunks"

# Create folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read cleaned text
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Recursive Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Create chunks
chunks = splitter.split_text(text)

# Save each chunk
for i, chunk in enumerate(chunks, start=1):
    file_path = os.path.join(output_dir, f"chunk_{i}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(chunk)

print("Recursive chunking completed successfully!")
print("Total recursive chunks:", len(chunks))
print("Chunk size:", 1000)
print("Chunk overlap:", 200)
print("Saved to:", output_dir)