from pypdf import PdfReader
import os

pdf_path = "data/pdf 1.pdf"

print("Checking document...")
print("File:", pdf_path)

# Check file exists
if not os.path.exists(pdf_path):
    print("ERROR: PDF file not found.")
    exit()

# Check file size
file_size = os.path.getsize(pdf_path)

if file_size == 0:
    print("ERROR: PDF file is empty.")
    exit()

print("File size:", file_size, "bytes")

# Try opening the PDF
try:
    reader = PdfReader(pdf_path)

    print("PDF opened successfully.")
    print("Total pages:", len(reader.pages))

    if len(reader.pages) == 0:
        print("ERROR: PDF contains no pages.")
        exit()

except Exception as e:
    print("ERROR: PDF is corrupted or cannot be opened.")
    print("Details:", e)
    exit()

# Check text extraction
total_text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        total_text += page_text

if len(total_text.strip()) == 0:
    print("WARNING: PDF contains no extractable text.")
else:
    print("Text extraction check: PASSED")
    print("Extracted characters:", len(total_text))

print("\nDocument validation completed!")