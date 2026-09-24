import os

input_path = "data/cleaned_text.txt"
output_folder = "data/chunks"

os.makedirs(output_folder, exist_ok=True)

# Read cleaned text
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Chunk settings
chunk_size = 1000
chunk_overlap = 200

chunks = []

start = 0

while start < len(text):

    end = start + chunk_size
    chunk = text[start:end]

    if chunk.strip():
        chunks.append(chunk)

    start = end - chunk_overlap

# Save chunks
for i, chunk in enumerate(chunks, start=1):

    output_path = os.path.join(
        output_folder,
        f"chunk_{i}.txt"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(chunk)

print("Fixed-size chunking completed!")
print("Chunk size:", chunk_size)
print("Chunk overlap:", chunk_overlap)
print("Total chunks:", len(chunks))
print("Saved in:", output_folder)