import itertools
from os import name
from services.dataset.dataset_loader import (
    load_lotte,
)
from services.evaluation.evaluation_service import build_qrels_by_query_id


def main():
    dataset = load_lotte()

    print("======================================")
    print("LOTTE QUERIES AND QRELS TEST")
    print("======================================")

    print(f"Documents: {dataset.docs_count():,}")
    print(f"Queries:   {dataset.queries_count():,}")
    print(f"Qrels:     {dataset.qrels_count():,}")

    relevance_values = {
        qrel.relevance
        for qrel in dataset.qrels_iter()
    }

    print(
        f"Relevance values: "
        f"{sorted(relevance_values)}"
    )

    print("\n======================================")
    print("FIRST 5 QUERIES")
    print("======================================")

    for query in itertools.islice(
        dataset.queries_iter(),
        5,
    ):
        print(f"Query ID: {query.query_id}")
        print(f"Text:     {query.text}")
        print("--------------------------------------")

    print("\n======================================")
    print("FIRST 10 QRELS")
    print("======================================")

    for qrel in itertools.islice(
        dataset.qrels_iter(),
        10,
    ):
        print(
            f"query_id={qrel.query_id}, "
            f"doc_id={qrel.doc_id}, "
            f"relevance={qrel.relevance}"
        )

    qrels_by_query_id = build_qrels_by_query_id(
        dataset
    )

    print("\n======================================")
    print("QRELS DICTIONARY TEST")
    print("======================================")

    print(
        f"Queries having Qrels: "
        f"{len(qrels_by_query_id):,}"
    )

    print("\nQrels for Query ID 0:")

    print(
        qrels_by_query_id.get(
            "0",
            {},
        )
    )

    print("\nQrels for Query ID 1:")

    print(
        qrels_by_query_id.get(
            "1",
            {},
        )
    )


if __name__ == "__main__":
    main()