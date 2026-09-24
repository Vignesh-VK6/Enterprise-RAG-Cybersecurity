import re

input_path = "data/extracted_text.txt"
output_path = "data/cleaned_text.txt"

# Read extracted text
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Remove excessive spaces
text = re.sub(r"[ \t]+", " ", text)

# Remove excessive blank lines
text = re.sub(r"\n\s*\n+", "\n\n", text)

# Remove leading/trailing spaces from each line
text = "\n".join(line.strip() for line in text.splitlines())

# Save cleaned text
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text.strip())

print("Text cleaning completed successfully!")
print("Original characters:", len(text))
print("Cleaned file saved at:", output_path)