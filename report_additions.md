## 🔍 خامساً: صقل الاستعلام وتحسينه (Query Refinement)

تم بناء نظام متعدد المراحل (Multi-Stage Pipeline) لتحسين الاستعلامات قبل تمريرها إلى محركات البحث (TF-IDF, BM25, Embedding, Hybrid, LTR). يتألف النظام من ثلاث مراحل رئيسية plus نظام إكمال تلقائي (Autocomplete) قائم على هيكل بيانات الـ Trie.

### المرحلة الأولى: تصحيح الأخطاء الإملائية (Spell Correction)
تقوم الدالة `correct_spelling()` بمعالجة كل كلمة على حدة:
* تُهمل الكلمات القصيرة (طولها ≤ 3) والكلمات الموجودة مسبقاً في قاموس NLTK الإنجليزي.
* للكلمات غير المعروفة، تُجرى عملية **إلغاء الجمع (Unpluralization)** أولاً：-strip لاحقات الجمع الشائعة مثل `-ies` → `-y`، `-es`، `-s` للتحقق من وجود الصيغة الأساسية في القاموس.
* إذا فشلت عملية إلغاء الجمع، يتم توليد جميع مرشحي المسافة الصرفية الواحدة (Edit Distance = 1) عبر الدالة `_edits1()` التي تشمل الحذف، التبديل، الاستبدال، والإدراج، ثم اختيار المرشح الأقرب طولاً.

```python
def _edits1(word: str) -> set[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {L + R[1:] for L, R in splits if R}
    transposes = {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1}
    replaces = {L + c + R[1:] for L, R in splits if R for c in letters}
    inserts = {L + c + R for L, R in splits for c in letters}
    return deletes | transposes | replaces | inserts


def correct_spelling(text: str) -> str:
    tokens = text.lower().split()
    corrected = []
    for token in tokens:
        if not token.isalpha() or len(token) <= 3 or token in ENGLISH_WORDS:
            corrected.append(token)
            continue
        inflected = _unpluralize(token)
        if inflected:
            corrected.append(inflected)
            continue
        candidates = {c for c in _edits1(token) & ENGLISH_WORDS if c[0] == token[0]}
        if candidates:
            corrected.append(min(candidates, key=lambda c: (abs(len(c) - len(token)), c)))
        else:
            corrected.append(token)
    return " ".join(corrected)
```

### المرحلة الثانية: توسيع المرادفات (Synonym Expansion)
بعد التصحيح، تُمرَّر النتيجة عبر خط المعالجة القبلية الكامل (`preprocess_lotte`) ثم يتم توسيع كل كلمة بحد أقصى مرادفتَين (`MAX_SYNONYMS_PER_TOKEN = 2`) من مزامير WordNet. تُحفظ الكلمات الأصلية دائماً إلى جانب المرادفات المضافة.

```python
def expand_synonyms(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        syns = wordnet.synsets(token)
        added = 0
        for synset in syns:
            if added >= MAX_SYNONYMS_PER_TOKEN:
                break
            for lemma in synset.lemmas():
                name = lemma.name().lower()
                if name != token and name.isalpha() and added < MAX_SYNONYMS_PER_TOKEN:
                    expanded.append(name)
                    added += 1
    return expanded
```

### المرحلة الثالثة: تعزيز بناءً على سجل البحث (History-Based Term Boosting)
تحافظ فئة `SearchHistory` على عدّاد (`Counter`) لكلمات الاستعلامات السابقة في جلسة البحث. يتم استخراج أعلى 5 كلمات تكراراً، ثم تُضاف كل منها مرتَين (`HISTORY_BOOST_COUNT = 2`) إلى قائمة الكلمات الموسّعة، مما يرفع وزنها التكراري (TF) بشكل غير مباشر عند حساب الترتيب.

```python
class SearchHistory:
    _term_frequencies: Counter = Counter()

    @classmethod
    def record_query(cls, tokens: list[str]):
        cls._term_frequencies.update(tokens)

    @classmethod
    def get_top_terms(cls, n: int = 5) -> list[str]:
        return [term for term, _ in cls._term_frequencies.most_common(n)]


def boost_from_history(tokens: list[str]) -> list[str]:
    boosted = list(tokens)
    top_history_terms = SearchHistory.get_top_terms(5)
    for term in top_history_terms:
        boosted.extend([term] * HISTORY_BOOST_COUNT)
    return boosted
```

### الدالة الموحدة (Refine Pipeline)
تجمع الدالة `refine_query()` المراحل الثلاث في خط أنابيب واحد:

```python
def refine_query(query_text: str) -> str:
    corrected = correct_spelling(query_text)
    tokens = preprocess_lotte(corrected)
    expanded = expand_synonyms(tokens)
    boosted = boost_from_history(expanded)
    SearchHistory.record_query(expanded)
    return " ".join(boosted)
```

### نظام الإكمال التلقائي (Trie Autocomplete)
تم بناء شجرة بادئة (Trie) من جميع المصطلحات الموجودة في الفهرس العكسي عند بدء تشغيل النظام. تدعم الدالة `suggest()` استكمال البادئة (Prefix Completion) باستخدام انتشار عمق DFS مع حد أقصى للنتائج. يُستخدم عبر نقطة النهاية `POST /api/suggest` وتستدعيه واجهة Streamlit عند إدخال المستخدم حرفين فأكثر.

```python
class VocabularyTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def suggest(self, prefix: str, limit: int = 10) -> list[str]:
        node = self._traverse(prefix)
        if node is None:
            return []
        results: list[str] = []
        self._dfs(node, prefix, results, limit)
        return results


def build_trie_from_inverted_index() -> VocabularyTrie:
    inverted_index = joblib.load(INDEX_DIR / "inverted_index.joblib")
    trie = VocabularyTrie()
    for term in inverted_index.keys():
        trie.insert(term)
    return trie
```

### التكامل مع طبقة البحث
يحتوي كل طلب بحث على علامة `use_refinement` (قيمة افتراضية `true`). عند التفعيل، تستدعي طبقة التحكم `refine_query()` قبل تمرير الاستعلام إلى نموذج الترتيب. تستثنى نماذج التضمين الدلالي (`Embedding`) من استخدام الاستعلام المُصَلّى، وتعمل على الاستعلام الخام مباشرة. يتضمن الاستجابة حقلَي `refined_query` و `enhanced` لعرض النتيجة في الواجهة.

```python
def search_documents(request: SearchRequest) -> SearchResponse:
    if request.use_refinement:
        try:
            refined_query = refine_query(request.query)
            query_modified = refined_query != request.query
        except Exception as exc:
            refined_query = request.query
            query_modified = False
    else:
        refined_query = request.query
        query_modified = False
```

---

## 🤖 سادساً: الترتيب بالتعلم الآلي (Learning to Rank - LTR)

تم بناء نظام ترتيب بالتعلم الآلي من نوع **Pointwise** يستخدم نموذج `GradientBoostingRegressor` من مكتبة `scikit-ltm` لإعادة ترتيب الوثائق المرشحة. يعمل النظام على مرحلتين: توليد المرشحين أولاً ثم إعادة ترتيبهم.

### المرحلة الأولى: توليد المرشحين عبر BM25
يُستخدم BM25 كمحرك استرجاع أولي (First-Stage Retriever)؛ حيث يتم استخراج أفضل **100** وثيقة (`CANDIDATE_K = 100`) لكل استعلام من الفهرس العكسي باستخدام درجات BM25، لتُشكّل مجموعة المرشحين التي سيعيد النموذج ترتيبها.

### المرحلة الثانية: استخراج الميزات وإعادة الترتيب
تقوم الدالة `LTRFeatureExtractor` باستخراج متجه ميزات رقمي من **12 ميزة** لكل زوج (استعلام، وثيقة مرشحة)：

| # | اسم الميزة | الوصف |
|---|-----------|-------|
| 1 | `bm25_score` | درجة BM25 لهذا الزوج |
| 2 | `embedding_similarity` | تشابه جيب التمام بين تضمين الاستعلام والوثيقة |
| 3 | `term_overlap_count` | عدد كلمات الاستعلام الموجودة في الوثيقة |
| 4 | `term_overlap_ratio` | نسبة التداخل إلى طول الاستعلام الكلي |
| 5 | `doc_length` | طول الوثيقة بعدد المصطلحات |
| 6 | `norm_doc_length` | الطول مُقسَّم على متوسط أطوال الوثائق |
| 7 | `query_length` | عدد رموز الاستعلام المعالج |
| 8 | `mean_idf_matched` | متوسط IDF للمصطلحات المطابقة |
| 9 | `max_idf_matched` | أقصى IDF للمصطلحات المطابقة |
| 10 | `min_idf_matched` | أدنى IDF للمصطلحات المطابقة |
| 11 | `sum_idf_matched` | مجموع قيم IDF للمصطلحات المطابقة |
| 12 | `bm25_rank` | ترتيب الوثيقة في قائمة BM25 الأولية |

```python
def extract_features(
    self, query_text: str, candidate_doc_ids: list[str],
    bm25_scores: dict[str, float], bm25_ranks: dict[str, int],
) -> np.ndarray:
    query_tokens = preprocess_lotte(query_text)
    query_length = len(query_tokens)
    unique_query_terms = set(query_tokens)

    embedding_similarities = self._compute_embedding_similarities(
        query_text, candidate_doc_ids
    )
    overlap_and_idf = self._compute_overlap_and_idf_stats(
        unique_query_terms, candidate_doc_ids
    )

    feature_matrix = []
    for rank_idx, doc_id in enumerate(candidate_doc_ids):
        dl = self.document_lengths.get(doc_id, 0)
        norm_dl = dl / self.avg_document_length if self.avg_document_length > 0 else 1.0
        info = overlap_and_idf.get(doc_id, {
            "overlap": 0, "mean_idf": 0.0, "max_idf": 0.0,
            "min_idf": 0.0, "sum_idf": 0.0,
        })
        overlap_count = info["overlap"]
        overlap_ratio = overlap_count / query_length if query_length > 0 else 0.0

        row = [
            float(bm25_scores.get(doc_id, 0.0)),
            float(embedding_similarities[rank_idx]),
            float(overlap_count),
            float(overlap_ratio),
            float(dl),
            float(norm_dl),
            float(query_length),
            float(info["mean_idf"]),
            float(info["max_idf"]),
            float(info["min_idf"]),
            float(info["sum_idf"]),
            float(bm25_ranks.get(doc_id, 999)),
        ]
        feature_matrix.append(row)

    return np.array(feature_matrix, dtype=np.float64)
```

### حساب تشابه التضمين (Embedding Similarity)
يتم تحويل الاستعلام فوراً إلى متجه تضمين عبر النموذج `all-MiniLM-L6-v2`، ثم حساب تشابه جيب التمام مع تضمينات الوثائق المرشحة المحفوظة مسبقاً:

```python
def _compute_embedding_similarities(
    self, query_text: str, doc_ids: list[str],
) -> np.ndarray:
    query_embedding = self.embedding_model.encode(
        [query_text], show_progress_bar=False, convert_to_numpy=True,
    )
    candidate_indices = []
    valid_mask = []
    for doc_id in doc_ids:
        idx = self.embedding_index_map.get(doc_id)
        if idx is not None:
            candidate_indices.append(idx)
            valid_mask.append(True)
        else:
            candidate_indices.append(0)
            valid_mask.append(False)

    candidate_embeddings = self.document_embeddings[candidate_indices]
    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
    similarities[~np.array(valid_mask)] = 0.0
    return similarities
```

### تدريب النموذج
يُدرب النموذج على بيانات التقييم المشتقة من تقارير BM25، حيث تُعامل الوثائق المسترجعة كعلامات ملاءمة ثنائية (1.0) والباقية كغير ملاءمة (0.0):

```python
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    min_samples_leaf=5,
    random_state=42,
)
model.fit(X, y)
joblib.dump(model, output_directory / "ltr_model.joblib")
```

### عملية البحث (Re-Ranking)
عند استدعاء البحث، يقوم النظام بما يلي:
1. تمعالجة الاستعلام عبر `preprocess_lotte()`.
2. توليد أفضل 100 مرشح عبر BM25 من الفهرس العكسي.
3. استخراج متجه الـ 12 ميزة لكل مرشح.
4. التنبؤ بدرجة الملاءمة عبر `model.predict(features)`.
5. ترتيب المرشحين تنازلياً حسب الدرجة الناتجة وإرجاع أفضل `top_k`.

```python
def search(self, query: str, top_k: int = 10, include_text: bool = True) -> dict:
    query_tokens = preprocess_lotte(query)

    # المرحلة الأولى: توليد المرشحين عبر BM25
    candidate_doc_ids, bm25_scores, bm25_ranks = (
        self._get_bm25_candidates(query_tokens, candidate_k=CANDIDATE_K)
    )

    # المرحلة الثانية: استخراج الميزات والتنبؤ
    features = self.feature_extractor.extract_features(
        query_text=query,
        candidate_doc_ids=candidate_doc_ids,
        bm25_scores=bm25_scores,
        bm25_ranks=bm25_ranks,
    )
    ltr_scores = self.model.predict(features)

    # ترتيب حسب الدرجة المتوقعة
    scored_candidates = list(zip(candidate_doc_ids, ltr_scores))
    top_indices = np.argsort(
        [score for _, score in scored_candidates]
    )[::-1][:top_k]

    ranked_results = []
    for idx in top_indices:
        doc_id = candidate_doc_ids[idx]
        ltr_score = max(float(ltr_scores[idx]), 0.0)
        ranked_results.append({"doc_id": doc_id, "score": ltr_score})
    # ...
```

### نتائج التقييم
تم تقييم نموذج LTR على **563** استعلام من مجموعة بيانات LoTTe وأظهر النتائج التالية:

| المقياس | القيمة |
|---------|--------|
| Precision@10 | 0.164 |
| Recall@10 | 0.604 |
| MAP@10 | 0.657 |
| nDCG@10 | 0.576 |
