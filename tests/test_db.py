from services.data_base.database_service import (
    get_raw_documents_by_ids,
)


def main():
    documents = get_raw_documents_by_ids(
        [
            "28",
            "34",
            "55",
        ]
    )

    print("======================================")
    print("DATABASE TEST")
    print("======================================")

    for document in documents:
        print("\n--------------------------------------")
        print(f"DOC ID: {document['doc_id']}")
        print("--------------------------------------")
        print(document["text"][:500])


if __name__ == "__main__":
    main()