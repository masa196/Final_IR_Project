FEATURE_NAMES = [
    "bm25_score",
    "embedding_similarity",
    "term_overlap_count",
    "term_overlap_ratio",
    "doc_length",
    "norm_doc_length",
    "query_length",
    "mean_idf_matched",
    "max_idf_matched",
    "min_idf_matched",
    "sum_idf_matched",
    "bm25_rank",
]

CANDIDATE_K = 100

N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.1
MIN_SAMPLES_LEAF = 5
RANDOM_STATE = 42

LTR_INDEX_NAME = "ltr"
