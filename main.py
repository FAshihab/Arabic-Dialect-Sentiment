import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Text Cleaning & Validation | وظيفة تنظيف وفحص النصوص
def clean_arabic_text(text):
    """
    Cleans Arabic text by normalizing characters and removing non-Arabic symbols.
    تنظيف النص العربي من خلال توحيد الأحرف وإزالة الرموز غير العربية.
    """
    text = text.strip()
    # Ensure the input contains Arabic characters | التأكد من أن الإدخال يحتوي على أحرف عربية
    if not re.search(r'[\u0600-\u06FF]', text):
        return None
        
    text = re.sub("[إأآ]", "ا", text) # Normalize Alef | توحيد الألف
    text = re.sub(r'(.)\1+', r'\1', text) # Remove repeated chars | إزالة تكرار الحروف
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text) # Remove symbols | إزالة الرموز
    return text

# 2. Setup and Training | إعداد البيانات وتدريب المودل
def setup_and_train():
    print("Starting the project... | جاري بدء المشروع...")
    data_path = 'data/dataset.csv'
    
    # Create directory if missing | إنشاء المجلد إذا لم يكن موجود
    if not os.path.exists('data'): os.makedirs('data')
    
    # Positive regional examples | أمثلة إيجابية بمختلف اللهجات
    positive_examples = [
        "والله إنه قيد القوة والجمال", "شغل هول يجنن ما شاء الله", "يا فديت روحك على هالشغل",
        "احس براحة نفسية اليوم", "الله يجعله دوم هالزين", "مرة مستانس ومبسوط بالحيل",
        "شغل سنع بالحيل وتوب", "يا بعد حي وميتي على هالزين", "ما شاء الله بالحيل مستانس",
        "خلف ابوي والله هالشغل", "تسذا الشغل السنع ولا بلاش", "يا حبني لكم على هالترتيب",
        "احس اني ملكت الدنيا من الفرحة", "يا ملحكم وملح شغلكم",
        "يا بعد راسي والله إنك ذيب", "شغل بطل وعز الله مقامك", "ما يقصرون أهل الشيمة",
        "والله اني اليوم في قمة حماسي", "الحمدلله الخاطر طيب",
        "يا واد الشغل مرة كول وجنان", "يسلموا دياتك على هالفن", "شغل مية مية وربي",
        "احس اني طاير من الفرح", "مرة نايس الله يسعدك",
        "شغل بطل ومرتب بالحيل", "كفو والله كفيت ووفيت", "مرة حبيته الله يوفقكم",
        "احس براحة وطمأنينة", "اليوم يومي وكل شي ضابط", "رايق", "مبسوط", "فرحان"
    ]
    
    # Negative regional examples | أمثلة سلبية بمختلف اللهجات
    negative_examples = [
        "والله إنه خياس وما يسوى", "شغل هباب وضاع وقتي", "مو زين ابد والتعامل تعيس",
        "للاسف خيبتوا ظني فيكم", "اسوأ مطعم جربته بحياتي", "خدمة بطيئة واكل طعمه غريب",
        "ما يستاهل ولا ريال واحد", "والله إنه فاشل وبقوة", "تعاملهم شين مرة وما انصح",
        "احس بضيق وتعبان مرة", "اليوم مالي خلق لاي شي", "مرة محبط والوضع ما يطمن",
        "احس اني ضايع ومهموم", "الوضع سيء والخدمة ماش", "تعبان", "حزين", "ضايق", "سيء", "خايس"
    ]
    
    # Balance data and labels | موازنة البيانات والتقييمات
    text_data = positive_examples + negative_examples
    sentiments = ([1] * len(positive_examples) + [0] * len(negative_examples))
    
    # Create DataFrame and save CSV | إنشاء الجدول وحفظ الملف
    data = {'text': text_data * 100, 'sentiment': sentiments * 100}
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    
    # Load and clean for training | تحميل البيانات وتنظيفها للتدريب
    df = pd.read_csv(data_path)
    df['text'] = df['text'].apply(lambda x: clean_arabic_text(x) or "نص غير معروف")
    
    # Vectorization & Model Training | تحويل النصوص لمتجهات وتدريب الموديل
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    # Split data | تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Calculate accuracy | حساب الدقة
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, acc

# 3. Interactive Execution | التنفيذ التفاعلي
if __name__ == "__main__":
    # Train the model and receive results | تدريب المودل وإستلام النتائج
    trained_model, our_vectorizer, final_accuracy = setup_and_train()
    print(f"Model Accuracy: {final_accuracy * 100:.2f}%")
    
    print("\n--- Interactive Sentiment Analysis ---")
    print("للأيقاف أكتب (خروج)")
    
    while True:
        # Prompt user input | طلب مدخلات المستخدم
        user_text = input("كيف تحس الحين؟: \n")
        
        if user_text.lower() in ['exit', 'خروج']:
            print("Exiting program... Goodbye!")
            break
            
        # Clean and Validate Input | تنظيف وفحص المدخلات
        cleaned = clean_arabic_text(user_text)
        if not cleaned:
            print("⚠️ يرجى إدخال نص عربي صحيح.")
            continue
            
        # Transform and Predict | التحويل والتنبؤ
        vec = our_vectorizer.transform([cleaned])
        res = trained_model.predict(vec)
        
        # Output Result | عرض النتيجة
        sentiment = "Positive | إيجابي" if res[0] == 1 else "Negative | سلبي"
        print(f"Analysis Result: {sentiment}")