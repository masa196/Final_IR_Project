import itertools
import random

from services.dataset.dataset_loader import (
    load_lotte,
    load_quora
)

from services.preprocessing.preprocessing_service import (
    preprocess_lotte,
    preprocess_quora
)


DATASET = "lotte"

if DATASET == "lotte":
    dataset = load_lotte()
    preprocess = preprocess_lotte
else:
    dataset = load_quora()
    preprocess = preprocess_quora

print(f"✅ {DATASET.upper()} Dataset loaded successfully")


# ==========================================================
# تجربة وثيقة
# ==========================================================
random_doc_index = random.randint(0, 1000)

doc = next(
    itertools.islice(
        dataset.docs_iter(),
        random_doc_index,
        None
    )
)

print("\n--- DOCUMENT ID ---")
print(doc.doc_id)

print("\n--- RAW DOCUMENT ---")
print(doc.text[:300])

print("\n--- PROCESSED DOCUMENT ---")
print(preprocess(doc.text)[:50])


# ==========================================================
# تجربة Query
# ==========================================================
queries = list(dataset.queries_iter())

query = random.choice(queries)

print("\n--- QUERY ID ---")
print(query.query_id)

print("\n--- RAW QUERY ---")
print(query.text)

print("\n--- PROCESSED QUERY ---")
print(preprocess(query.text))