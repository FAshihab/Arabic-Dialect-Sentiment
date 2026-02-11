import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Text Preprocessing Function | وظيفة معالجة وتنظيف النصوص
def clean_arabic_text(text):
    text = str(text)
    # Normalize Alef, Ya, and Ta Marbuta | توحيد الألف والياء والتاء المربوطة
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    # Remove Diacritics (Tashkeel) | إزالة التشكيل
    text = re.sub(r'[\u064B-\u0652]', '', text)
    # Remove non-Arabic characters | إزالة أي رموز غير عربية
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    # Remove character elongation | إزالة الحروف المكررة (تطويل)
    text = re.sub(r'(.)\1+', r'\1', text)
    return text.strip()

print(" Starting the project... | جاري بدء المشروع...")

# 2. Load or Generate Dataset | تحميل أو توليد البيانات
try:
    # Try to load existing file | محاولة تحميل الملف الموجود
    df = pd.read_csv('data/large_dataset.csv', names=['label', 'text'])
    
    # If file is too small, let's boost it! | إذا الملف صغير جداً، خلينا نكبره
    if len(df) < 20:
        print("Dataset too small, generating synthetic data... | البيانات قليلة، جاري توليد بيانات إضافية...")
        # Synthetic Data | بيانات اصطناعية لتعليم الموديل
        extra_data = {
            'text': [
                "الاكل يجنن", "خدمة رائعة", "ممتاز جدا", "رهيب انصحكم فيه", "بطل بطل",
                "سيء جدا", "تجربة تعيسة", "ما انصح فيه", "اكل بارد", "تاخير في الطلب",
                "المكان نظيف", "موظفين محترمين", "شغل مرتب", "واو حبيت", "يستاهل كل ريال",
                "غالي على فاضي", "قرف استغفر الله", "اسوء مطعم", "خدمة بطيئة", "لا يعيد التجربة"
            ] * 50, # نكرر الجمل 50 مرة عشان يصير عندنا 1000 جملة
            'label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 50
        }
        df = pd.DataFrame(extra_data)
    
    print(f"Ready with {len(df)} samples! | جاهز بـ {len(df)} عينة!")

except Exception:
    print("Creating new dataset file... | جاري إنشاء ملف بيانات جديد...")
    # (نفس الكود اللي فوق لتوليد البيانات)
    exit()

# 3. Clean the text data | تنظيف بيانات النصوص
print("Cleaning text... | جاري تنظيف النصوص...")
df['cleaned_text'] = df['text'].apply(clean_arabic_text)

# 4. Split data into Train and Test sets | تقسيم البيانات لتدريب واختبار
# 80% Training, 20% Testing | 80% للتدريب و 20% للاختبار
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)

# 5. Convert text to numbers (Vectorization) | تحويل النصوص لأرقام
vectorizer = TfidfVectorizer(max_features=5000) 
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train the AI Model (Naive Bayes) | تدريب موديل الذكاء الاصطناعي
print("Training the model... | جاري تدريب الموديل...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 7. Evaluate Model Performance | تقييم أداء الموديل
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Final Result | النتيجة النهائية")
print(f"Accuracy: {accuracy * 100:.2f}% | دقة الموديل")