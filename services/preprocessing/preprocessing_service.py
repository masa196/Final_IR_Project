import re
import nltk
import unicodedata
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag


# ---------- الإعداد المبدئي ----------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------- دالة مساعدة لربط الـ POS Tags بـ WordNet ----------
def get_wordnet_pos(tag):
    tag = tag[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


# ---------- الدالة الأساسية لمعالجة النصوص ----------
def preprocess_text(text: str, keep_numbers=False):
    if not text or not isinstance(text, str):
        return []

    # 1. تحويل الحروف إلى صغيرة
    text = text.lower()

    text = text.replace("d&d", "dnd")
    text = text.replace("i.e.", " ")
    text = text.replace("e.g.", " ")
    text = text.replace("vs.", " vs ")

    # 1.5. تطبيع النصوص باستخدام Unicode Normalization لتوحيد تمثيل الحروف والأرقام
    text = normalize_unicode(text)

    # 1.6. توسيع الاختصارات (Contractions)
    text = contractions.fix(text)

    # 2. إزالة روابط الويب أولاً
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)

    # 50,000 → 50000
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    # 1.8 → 1_8
    text = re.sub(r"(?<=\d)\.(?=\d)", "_", text)

    # 3. إزالة الرموز الخاصة وعلامات الترقيم
    text = re.sub(r"[^\w\s]", " ", text)

    # 4. التقطيع إلى كلمات (Tokenization)
    tokens = word_tokenize(text)

    # 5. استخراج الـ POS Tags والـ Lemmatization قبل حذف الـ Stopwords
    pos_tags = pos_tag(tokens)

    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]

  
    final_tokens = []
    for t in lemmatized:
        if t in stop_words:
            continue        
        if t.isnumeric() and not keep_numbers:
            continue  
            
        final_tokens.append(t)

    return final_tokens

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# ---------- دالة معالجة LoTTe ----------
def preprocess_lotte(text):
    return preprocess_text(text, keep_numbers=True)


