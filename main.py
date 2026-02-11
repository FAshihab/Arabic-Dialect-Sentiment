import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Text Cleaning and Validation | وظيفة تنظيف وفحص النصوص
def clean_arabic_text(text):
    # إزالة المسافات الزائدة في البداية والنهاية | Removing excess space at the beginning and end
    text = text.strip()
    
    # التحقق مما إذا كان النص فارغ أو يحتوي على أحرف إنجليزية/رموز فقط (الخرابيط) | Check if input is empty or just gibberish (non-Arabic characters)

    if not re.search(r'[\u0600-\u06FF]', text):
        return None  # نعيد None لنعرف أن الإدخال غير صحيح
        
    text = re.sub("[إأآ]", "ا", text)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text

# 2. Setup and Training | إعداد البيانات وتدريب المودل
def setup_and_train():
    print("Starting the project... | جاري بدء المشروع...")
    data_path = 'data/dataset.csv'
    if not os.path.exists('data'): os.makedirs('data')
    
    # قائمة البيانات | Dataset
    text_data = [
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
        "احس براحة وطمأنينة", "اليوم يومي وكل شي ضابط",
        "والله إنه خياس وما يسوى", "شغل هباب وضاع وقتي", "مو زين ابد والتعامل تعيس",
        "للاسف خيبتوا ظني فيكم", "اسوأ مطعم جربته بحياتي", "خدمة بطيئة واكل طعمه غريب",
        "ما يستاهل ولا ريال واحد", "والله إنه فاشل وبقوة", "تعاملهم شين مرة وما انصح",
        "احس بضيق وتعبان مرة", "اليوم مالي خلق لاي شي", "مرة محبط والوضع ما يطمن",
        "احس اني ضايع ومهموم", "الوضع سيء والخدمة ماش"
    ]
    
    sentiments = ([1] * 29 + [0] * 14)
    data = {'text': text_data * 100, 'sentiment': sentiments * 100}
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    
    df = pd.read_csv(data_path)
    df['text'] = df['text'].apply(lambda x: clean_arabic_text(x) or "نص غير معروف")
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, acc

# 3. Execution | التنفيذ التفاعلي
if __name__ == "__main__":
    trained_model, our_vectorizer, final_accuracy = setup_and_train()
    print(f"Model Accuracy: {final_accuracy * 100:.2f}%")
    
    print("\n--- Interactive Sentiment Analysis ---")
    print(" --- للخروج أكتب (خروج) ---")
    
    while True:
        user_text = input("\nكيف تحس الحين؟: \n")
        
        if user_text.lower() == 'exit' or user_text == 'خروج':
            print("Exiting program... Goodbye!")
            break
            

        cleaned = clean_arabic_text(user_text)
        
        if cleaned is None or len(cleaned.strip()) == 0:
            print("⚠️ عفواً، يرجى إدخال نص صحيح باللغة العربية.")
            continue
            
        vec = our_vectorizer.transform([cleaned])
        res = trained_model.predict(vec)
        
        sentiment = "Positive | إيجابي" if res[0] == 1 else "Negative | سلبي"
        print(f"Analysis Result: {sentiment}")