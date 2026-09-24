import csv

METADATA_FILE = "Metadata/document_metadata.csv.txt"


# Load metadata
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    documents = list(reader)

print("=" * 70)
print("METADATA FILTERING")
print("=" * 70)

print(f"\nTotal documents: {len(documents)}")


def metadata_filter(
    document_name=None,
    domain=None,
    sub_domain=None,
    publication_date=None
):
    results = documents

    if document_name:
        results = [
            doc for doc in results
            if document_name.lower() in doc["document_name"].lower()
        ]

    if domain:
        results = [
            doc for doc in results
            if domain.lower() == doc["domain"].lower()
        ]

    if sub_domain:
        results = [
            doc for doc in results
            if sub_domain.lower() == doc["sub_domain"].lower()
        ]

    if publication_date:
        results = [
            doc for doc in results
            if publication_date == doc["publication_date"]
        ]

    return results


# --------------------------------------------------
# TEST 1 - Document Name
# --------------------------------------------------

print("\n" + "=" * 70)
print("TEST 1: DOCUMENT NAME FILTER")
print("=" * 70)

results = metadata_filter(
    document_name="pdf 1.pdf"
)

for doc in results:
    print(f"Document : {doc['document_name']}")
    print(f"Organization : {doc['organization']}")


# --------------------------------------------------
# TEST 2 - Domain
# --------------------------------------------------

print("\n" + "=" * 70)
print("TEST 2: DOMAIN FILTER")
print("=" * 70)

results = metadata_filter(
    domain="Cybersecurity"
)

for doc in results:
    print(f"Document : {doc['document_name']}")
    print(f"Domain   : {doc['domain']}")


# --------------------------------------------------
# TEST 3 - Sub Domain
# --------------------------------------------------

print("\n" + "=" * 70)
print("TEST 3: SUB-DOMAIN FILTER")
print("=" * 70)

results = metadata_filter(
    sub_domain="Cybersecurity & Privacy"
)

for doc in results:
    print(f"Document : {doc['document_name']}")
    print(f"Sub-Domain : {doc['sub_domain']}")


# --------------------------------------------------
# TEST 4 - Date
# --------------------------------------------------

print("\n" + "=" * 70)
print("TEST 4: DATE FILTER")
print("=" * 70)

results = metadata_filter(
    publication_date="2006-10"
)

for doc in results:
    print(f"Document : {doc['document_name']}")
    print(f"Publication Date : {doc['publication_date']}")


print("\n" + "=" * 70)
print("METADATA FILTERING TEST COMPLETED!")
print("=" * 70)