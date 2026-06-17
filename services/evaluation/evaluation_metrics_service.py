from math import log2


# ==========================================================
# Precision@K
# ==========================================================
from math import log2
def precision_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> float:
    """
    حساب Precision@K.

    retrieved_doc_ids:
        الوثائق التي أعادها محرك البحث مرتبة من الأفضل إلى الأسوأ.

    relevant_doc_ids:
        الوثائق الصحيحة الموجودة في Qrels للسؤال الحالي.

    k:
        عدد النتائج الأولى التي نريد تقييمها.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    # أخذ أفضل K نتائج فقط.
    top_k_results = retrieved_doc_ids[:k]

    # حساب الوثائق الصحيحة الموجودة ضمن أفضل K نتائج.
    relevant_retrieved_count = len(
        set(top_k_results)
        & relevant_doc_ids
    )

    return relevant_retrieved_count / k


# ==========================================================
# Recall@K
# ==========================================================

def recall_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> float:
    """
    حساب Recall@K.

    retrieved_doc_ids:
        الوثائق التي أعادها محرك البحث مرتبة من الأفضل إلى الأسوأ.

    relevant_doc_ids:
        جميع الوثائق الصحيحة الموجودة في Qrels للسؤال الحالي.

    k:
        عدد النتائج الأولى التي نريد تقييمها.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    # في حال لم توجد وثائق صحيحة للسؤال،
    # نتجنب القسمة على صفر.
    if not relevant_doc_ids:
        return 0.0

    # أخذ أفضل K نتائج فقط.
    top_k_results = retrieved_doc_ids[:k]

    # حساب الوثائق الصحيحة التي تم استرجاعها.
    relevant_retrieved_count = len(
        set(top_k_results)
        & relevant_doc_ids
    )

    return (
        relevant_retrieved_count
        / len(relevant_doc_ids)
    )


# ==========================================================
# Average Precision@K
# ==========================================================

def average_precision_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> float:
    """
    حساب Average Precision@K لسؤال واحد.

    نهتم بموقع الوثائق الصحيحة داخل الترتيب.
    كلما ظهرت الوثائق الصحيحة مبكراً، ارتفعت النتيجة.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    # أخذ أفضل K نتائج فقط.
    top_k_results = retrieved_doc_ids[:k]

    relevant_found_count = 0
    precision_sum = 0.0

    # rank يبدأ من 1 لأن أول نتيجة هي Rank 1 وليست Rank 0.
    for rank, doc_id in enumerate(
        top_k_results,
        start=1,
    ):
        # إذا كانت الوثيقة صحيحة حسب Qrels.
        if doc_id in relevant_doc_ids:
            relevant_found_count += 1

            # حساب Precision عند هذا الموضع فقط.
            precision_at_current_rank = (
                relevant_found_count
                / rank
            )

            precision_sum += (
                precision_at_current_rank
            )

    # إذا لم نجد أي وثيقة صحيحة ضمن أول K نتائج.
    if relevant_found_count == 0:
        return 0.0

    return (
        precision_sum
        / relevant_found_count
    )

# ==========================================================
# Mean Average Precision@K
# ==========================================================

def mean_average_precision_at_k(
    evaluation_cases: list[
        tuple[list[str], set[str]]
    ],
    k: int = 10,
) -> float:
    """
    حساب MAP@K لعدة Queries.

    كل عنصر داخل evaluation_cases يحتوي:
    (
        الوثائق التي أعادها النظام بالترتيب,
        الوثائق الصحيحة الموجودة في Qrels
    )
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    if not evaluation_cases:
        return 0.0

    average_precision_scores = []

    for retrieved_doc_ids, relevant_doc_ids in (
        evaluation_cases
    ):
        score = average_precision_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_doc_ids=relevant_doc_ids,
            k=k,
        )

        average_precision_scores.append(
            score
        )

    return (
        sum(average_precision_scores)
        / len(average_precision_scores)
    )


# ==========================================================
# دالة مساعدة لحساب DCG من درجات الملاءمة
# ==========================================================

def _dcg_from_relevance_scores(
    relevance_scores: list[float],
) -> float:
    """
    حساب DCG اعتماداً على قائمة درجات الملاءمة المرتبة.

    مثال:
    [0, 1, 2, 0, 2]

    كل عنصر يمثل درجة ملاءمة الوثيقة في موقعها الحالي.
    """

    dcg_score = 0.0

    for rank, relevance in enumerate(
        relevance_scores,
        start=1,
    ):
        gain = (2 ** relevance) - 1

        discount = log2(
            rank + 1
        )

        dcg_score += (
            gain / discount
        )

    return dcg_score    

# ==========================================================
# DCG@K
# ==========================================================

def dcg_at_k(
    retrieved_doc_ids: list[str],
    relevance_by_doc_id: dict[str, float],
    k: int = 10,
) -> float:
    """
    حساب DCG@K للنتائج التي أعادها محرك البحث.

    retrieved_doc_ids:
        الوثائق المسترجعة مرتبة من الأفضل إلى الأسوأ.

    relevance_by_doc_id:
        درجة ملاءمة كل وثيقة حسب Qrels.

        مثال:
        {
            "A": 0,
            "B": 1,
            "C": 2,
        }

    k:
        عدد النتائج الأولى المراد تقييمها.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    top_k_results = retrieved_doc_ids[:k]

    relevance_scores = [
        relevance_by_doc_id.get(
            doc_id,
            0.0,
        )
        for doc_id in top_k_results
    ]

    return _dcg_from_relevance_scores(
        relevance_scores
    )



# ==========================================================
# nDCG@K
# ==========================================================

def ndcg_at_k(
    retrieved_doc_ids: list[str],
    relevance_by_doc_id: dict[str, float],
    k: int = 10,
) -> float:
    """
    حساب nDCG@K.

    نقارن DCG الفعلية بترتيب مثالي IDCG.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    actual_dcg = dcg_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevance_by_doc_id=relevance_by_doc_id,
        k=k,
    )

    ideal_relevance_scores = sorted(
        relevance_by_doc_id.values(),
        reverse=True,
    )[:k]

    ideal_dcg = _dcg_from_relevance_scores(
        ideal_relevance_scores
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

