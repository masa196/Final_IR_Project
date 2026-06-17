import json
from pathlib import Path
import joblib
from sentence_transformers import  SentenceTransformer

def build_embedding(
    inverted_index_directory: Path,
    output_directory: Path,
    get_raw_documents_by_ids_func,
    batch_size: int = 10000,
) -> dict:


    inverted_index_directory = Path(inverted_index_directory)
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("======================================")
    print("BUILDING EMBEDDING ")
    print("======================================")

    print("Loading document ids...")

    with (
        inverted_index_directory
        / "document_ids.json"
    ).open("r", encoding="utf-8") as file:
        document_ids = [
            str(doc_id)
            for doc_id in json.load(file)
        ]

    number_of_documents = len(document_ids)

    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_embeddings = []
    valid_document_ids = []

    print("Calculating Embedding matrix values in batches...")
    
    
    for i in range(0, number_of_documents, batch_size):
        batch_ids = document_ids[i : i + batch_size]
        
       
        raw_documents = get_raw_documents_by_ids_func(batch_ids)
        
        batch_texts = [doc["text"] for doc in raw_documents]
        batch_valid_ids = [str(doc["doc_id"]) for doc in raw_documents]

        if not batch_texts:
            continue

        
        batch_embeddings = model.encode(
            batch_texts, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        
        all_embeddings.append(batch_embeddings)
        valid_document_ids.extend(batch_valid_ids)
        
        print(f"Processed: {min(i + batch_size, number_of_documents):,}/{number_of_documents:,}")

    import numpy as np
    final_embedding_matrix = np.vstack(all_embeddings)

    report = {
        "number_of_documents": len(valid_document_ids),
        "embedding_dimension": final_embedding_matrix.shape[1],
        "matrix_shape": list(final_embedding_matrix.shape),
    }

    print("Saving Embedding files...")

    joblib.dump(
        final_embedding_matrix,
        output_directory / "document_embeddings.joblib",
    )

    joblib.dump(
        valid_document_ids,
        output_directory / "embedding_document_ids.joblib",
    )

    with (
        output_directory
        / "report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("✅ EMBEDDING  BUILT SUCCESSFULLY")

    return report