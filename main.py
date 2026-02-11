import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Setup and Data Generation | إعداد وتوليد البيانات
def setup_data():
    print("Starting the project...") # بدء المشروع
    data_path = 'data/dataset.csv'
    
    # Create data folder if it doesn't exist | إنشاء مجلد البيانات إذا لم يكن موجود
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Generate synthetic data if file is missing | توليد بيانات إصطناعية إذا كان الملف مفقود
    if not os.path.exists(data_path):
        print("Dataset not found. Generating 1000 synthetic samples...")
        data = {
            'text': [
                "هذا المطعم يجنن والأكل رائع", "الخدمة ممتازة والجو جميل", 
                "تجربة سيئة جدا وتأخير في الطلب", "الأكل بارد والخدمة تعيسة",
                "ممتاز انصح به", "لا انصح بالتعامل معهم", "جيد جدا", "سيء للغاية"
            ] * 125,
            'sentiment': [1, 1, 0, 0, 1, 0, 1, 0] * 125
        }
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        print("New dataset file created successfully.")
    
    return pd.read_csv(data_path)

# 2. Text Cleaning Function | وظيفة تنظيف النصوص
def clean_arabic_text(text):
    # Normalize Alef | توحيد أشكال الألف
    text = re.sub("[إأآ]", "ا", text)
    # Remove repeated characters | إزالة تكرار الحروف (تطويل)
    text = re.sub(r'(.)\1+', r'\1', text)
    # Remove non-arabic characters | إزالة الرموز والأحرف غير العربية
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text

# 3. Main Execution Flow | مسار التنفيذ الرئيسي
def run_analysis():
    # Load the data | تحميل البيانات
    df = setup_data()
    
    print("Cleaning text...") # جاري تنظيف النصوص
    df['text'] = df['text'].apply(clean_arabic_text)
    
    # 4. Vectorization (Linear Algebra) | تحويل النصوص لمتجهات (الجبر الخطي)
    # Transforming text into a Matrix of numbers | تحويل النص إلى مصفوفة أرقام
    print("Training the model...") # جاري تدريب الموديل
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text']) # This is your Matrix | المصفوفة الخاصة بك
    y = df['sentiment']
    
    # Split data into training and testing | تقسيم البيانات للتدريب والاختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Machine Learning Model | تدريب موديل تعلم الآلة
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # 5. Final Results | النتائج النهائية
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\nFinal Result") # النتيجة النهائية
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    run_analysis()