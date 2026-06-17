import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================================
# قراءة الوثائق المعالجة واحدة واحدة
# ==========================================================

def iterate_processed_documents(
    documents_file: Path,
    document_ids: list[str],
    statistics: dict,
    max_documents: int | None = None,
    print_every: int = 1000,
):
    """
    قراءة الوثائق المعالجة من JSONL واحدة واحدة.

    نعيد النص المنظف بالشكل:
    "game player combat"

    ونحتفظ بترتيب doc_ids حتى نعرف لاحقاً
    أي صف من TF-IDF Matrix يعود لأي وثيقة.
    """

    with documents_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if (
                max_documents is not None
                and line_number > max_documents
            ):
                break

            record = json.loads(line)

            doc_id = str(record["doc_id"])
            tokens = record["tokens"]

            document_ids.append(doc_id)

            statistics["documents_count"] += 1

            if not tokens:
                statistics["empty_documents"] += 1

            if line_number % print_every == 0:
                print(
                    f"Loaded documents: "
                    f"{line_number:,}"
                )

            # مثال:
            # ["machine", "learning", "course"]
            #
            # تصبح:
            # "machine learning course"
            yield " ".join(tokens)


# ==========================================================
# بناء TF-IDF
# ==========================================================

def build_tfidf_index(
    documents_file: Path,
    inverted_index_file: Path,
    output_directory: Path,
    max_documents: int | None = None,
    print_every: int = 1000,
):
    """
    بناء TF-IDF Matrix وحفظ الملفات الناتجة.
    """

    if not documents_file.exists():
        raise FileNotFoundError(
            f"Documents file was not found: "
            f"{documents_file}"
        )

    if not inverted_index_file.exists():
        raise FileNotFoundError(
            f"Inverted Index file was not found: "
            f"{inverted_index_file}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading vocabulary from Inverted Index...")

    inverted_index = joblib.load(
        inverted_index_file
    )

    # تحويل كلمات Inverted Index إلى Vocabulary.
    #
    # مثال:
    # {
    #     "game": 0,
    #     "player": 1,
    #     "combat": 2,
    # }
    vocabulary = {
        term: index
        for index, term in enumerate(
            inverted_index.keys()
        )
    }

    print(
        f"Vocabulary loaded: "
        f"{len(vocabulary):,} terms"
    )

    document_ids = []

    statistics = {
        "documents_count": 0,
        "empty_documents": 0,
    }

    # tokenizer=str.split مهم جداً.
    #
    # لأنه يحافظ على Tokens كما خزّناها تماماً:
    #
    # project_manager → Token واحدة
    # 3_5             → Token واحدة
    # 5               → لا يتم حذفها
    vectorizer = TfidfVectorizer(
        vocabulary=vocabulary,
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        sublinear_tf=True,
        ngram_range=(1, 1),
        norm="l2",
        dtype=np.float32,
    )

    print("\nBuilding TF-IDF Matrix...")

    start_time = time.time()

    tfidf_matrix = vectorizer.fit_transform(
        iterate_processed_documents(
            documents_file=documents_file,
            document_ids=document_ids,
            statistics=statistics,
            max_documents=max_documents,
            print_every=print_every,
        )
    )

    elapsed_seconds = round(
        time.time() - start_time,
        2,
    )

    # ======================================================
    # حفظ الملفات
    # ======================================================

    vectorizer_file = (
        output_directory
        / "vectorizer.joblib"
    )

    matrix_file = (
        output_directory
        / "tfidf_matrix.npz"
    )

    document_ids_file = (
        output_directory
        / "document_ids.json"
    )

    report_file = (
        output_directory
        / "report.json"
    )

    print("\nSaving vectorizer.joblib...")

    joblib.dump(
        vectorizer,
        vectorizer_file,
    )

    print("✅ vectorizer.joblib saved")

    print("Saving tfidf_matrix.npz...")

    save_npz(
        matrix_file,
        tfidf_matrix,
    )

    print("✅ tfidf_matrix.npz saved")

    print("Saving document_ids.json...")

    with document_ids_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            document_ids,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("✅ document_ids.json saved")

    report = {
        "dataset": "lotte",
        "model": "vsm_tfidf",
        "mode": (
            "full"
            if max_documents is None
            else "test"
        ),
        "documents_count": (
            statistics["documents_count"]
        ),
        "empty_documents": (
            statistics["empty_documents"]
        ),
        "vocabulary_size": len(vocabulary),
        "matrix_rows": tfidf_matrix.shape[0],
        "matrix_columns": tfidf_matrix.shape[1],
        "non_zero_values": int(
            tfidf_matrix.nnz
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("✅ report.json saved")

    return report