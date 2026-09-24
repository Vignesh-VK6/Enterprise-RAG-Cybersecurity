from pypdf import PdfReader
import os
import json

pdf_path = "data/pdf 1.pdf"

reader = PdfReader(pdf_path)

# PDF metadata
pdf_metadata = reader.metadata

# File information
file_size = os.path.getsize(pdf_path)

metadata = {
    "file_name": os.path.basename(pdf_path),
    "file_size_bytes": file_size,
    "number_of_pages": len(reader.pages),
    "title": pdf_metadata.title if pdf_metadata else None,
    "author": pdf_metadata.author if pdf_metadata else None,
    "subject": pdf_metadata.subject if pdf_metadata else None,
    "creator": pdf_metadata.creator if pdf_metadata else None,
    "producer": pdf_metadata.producer if pdf_metadata else None,
    "creation_date": str(pdf_metadata.creation_date) if pdf_metadata and pdf_metadata.creation_date else None,
    "modification_date": str(pdf_metadata.modification_date) if pdf_metadata and pdf_metadata.modification_date else None
}

# Print metadata
print("\n--- DOCUMENT METADATA ---")

for key, value in metadata.items():
    print(f"{key}: {value}")

# Save metadata
output_path = "data/metadata.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("\nMetadata extracted successfully!")
print("Saved at:", output_path)