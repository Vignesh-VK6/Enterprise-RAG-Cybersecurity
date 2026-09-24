from pypdf import PdfReader

pdf_path = "data/pdf 1.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("Total Pages:", len(reader.pages))
print("Total Characters:", len(text))

print("\n--- FIRST 2000 CHARACTERS ---\n")
print(text[:2000])

output_path = "data/extracted_text.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Text saved successfully!")
output_path = "data/extracted_text.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Text saved successfully!")
