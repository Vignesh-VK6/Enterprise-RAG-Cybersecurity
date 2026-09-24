from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

# Input file
input_path = "data/cleaned_text.txt"

# Output folder
output_dir = "data/semantic_chunks"

os.makedirs(output_dir, exist_ok=True)

# Read cleaned text
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Split into paragraphs
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

print("Total paragraphs:", len(paragraphs))
print("Loading embedding model...")

# Local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(paragraphs)

# Semantic similarity threshold
threshold = 0.65

chunks = []
current_chunk = paragraphs[0]

for i in range(1, len(paragraphs)):
    similarity = cosine_similarity(
        [embeddings[i - 1]],
        [embeddings[i]]
    )[0][0]

    if similarity >= threshold:
        current_chunk += "\n\n" + paragraphs[i]
    else:
        chunks.append(current_chunk)
        current_chunk = paragraphs[i]

# Add final chunk
chunks.append(current_chunk)

# Save chunks
for i, chunk in enumerate(chunks, start=1):
    file_path = os.path.join(
        output_dir,
        f"chunk_{i}.txt"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(chunk)

print("Semantic chunking completed successfully!")
print("Total semantic chunks:", len(chunks))
print("Similarity threshold:", threshold)
print("Saved to:", output_dir)