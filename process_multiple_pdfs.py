from pypdf import PdfReader
import os

input_folder = "raw_documents/PDF"
output_folder = "data/multiple_extracted"

os.makedirs(output_folder, exist_ok=True)

pdf_files = [
    file for file in os.listdir(input_folder)
    if file.lower().endswith(".pdf")
]

print("Total PDF files found:", len(pdf_files))

for pdf_file in pdf_files:

    pdf_path = os.path.join(input_folder, pdf_file)

    print("\nProcessing:", pdf_file)

    try:
        reader = PdfReader(pdf_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        output_name = os.path.splitext(pdf_file)[0] + ".txt"
        output_path = os.path.join(output_folder, output_name)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print("Pages:", len(reader.pages))
        print("Characters:", len(text))
        print("Saved:", output_path)

    except Exception as e:
        print("ERROR:", pdf_file)
        print("Details:", e)

print("\nMultiple PDF processing completed!")