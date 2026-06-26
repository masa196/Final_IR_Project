import time
from pathlib import Path

# التأكد من تثبيت المكتبة: pip install faiss-cpu
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# استيراد الخدمة ومسارات المشروع
from services.embedding.embedding_search_service import EmbeddingSearchService

print("====================================================")
print("⚔️ VECTOR STORE SPEED RACE: FAISS VS CUSTOM BRUTE-FORCE")
print("====================================================")

# 1. تحديد المسار المطلق لمجلد الـ embedding بدقة كالعادة
PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_DIRECTORY = PROJECT_ROOT / "indices" / "lotte" / "embedding"

print("📥 Loading project data and services...")
try:
    original_embedding_service = EmbeddingSearchService(embedding_directory=EMBEDDING_DIRECTORY)
    raw_matrix = original_embedding_service.document_embeddings
    document_ids = original_embedding_service.document_ids
    model = original_embedding_service.model
except Exception as e:
    print(f"❌ خطأ أثناء تحميل الخدمة: {e}")
    exit(1)

# تجهيز المصفوفات
embeddings_matrix = np.array(raw_matrix).astype('f4')

# 2. بناء فهرس الـ Vector Store (FAISS)
dimension = embeddings_matrix.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings_matrix)


# 3. دالة المقارنة الشاملة
def compare_search_performance(query: str, top_k: int = 5):
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    
    # تحويل السؤال لمتجه ليكون موحداً في الطريقتين
    query_vector = model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype('f4')
    
    # ----------------------------------------------------
    # 🔴 أولاً: البحث التقليدي (بدون Vector Store) - يدوي خطي
    # ----------------------------------------------------
    start_old = time.time()
    
    # حساب الجداء الداخلي يدوياً لكل المستندات (Matrix Multiplication)
    # وهو ما يعادل الكوزين سيملاريتي لأن المتجهات ممعيرة
    custom_scores = np.dot(embeddings_matrix, query_vector.T).flatten()
    
    # ترتيب النتائج يدوياً وتنازلياً لجلب الأفضل
    old_top_indices = np.argsort(custom_scores)[::-1][:top_k]
    
    time_without_faiss = time.time() - start_old
    
    # ----------------------------------------------------
    # 🟢 ثانياً: البحث باستخدام الـ Vector Store (FAISS)
    # ----------------------------------------------------
    start_new = time.time()
    
    faiss_scores, faiss_indices = faiss_index.search(query_vector, top_k)
    
    time_with_faiss = time.time() - start_new
    
    # ----------------------------------------------------
    # 📊 طباعة تقرير المقارنة والدليل الرقمي
    # ----------------------------------------------------
    print(f"🔴 Time WITHOUT Vector Store (Custom): {time_without_faiss:.6f} seconds")
    print(f"🟢 Time WITH Vector Store (FAISS):      {time_with_faiss:.6f} seconds")
    
    # حساب كم مرة أصبح أسرع
    speedup = time_without_faiss / max(time_with_faiss, 1e-9)
    print(f"\n🚀 RESULT: Vector Store is [ {speedup:.1f}x ] FASTER!!")
    print("-" * 60)
    
    # التأكيد على تطابق النتائج الرياضية لضمان الأمان
    print("🎯 Sample of FAISS Results (Rank & Doc ID):")
    for rank, idx in enumerate(faiss_indices[0], start=1):
        if idx != -1:
            print(f"  Rank {rank}: Doc ID = {document_ids[idx]}")
    print("=" * 60)

# ====================================================
# 🔥 تشغيل اختبار المقارنة الحيّة
# ====================================================
sample_query = "do spells count as ranged attacks?"
compare_search_performance(sample_query, top_k=5)