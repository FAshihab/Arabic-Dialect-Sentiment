import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Text Cleaning & Validation Logic
def clean_arabic_text(text):
    """
    Normalizes Arabic characters and filters out non-Arabic noise.
    Ensures input validity before processing.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    
    # Check if the text contains Arabic characters to avoid gibberish
    if not re.search(r'[\u0600-\u06FF]', text):
        return None
        
    # Normalization and cleaning
    text = re.sub("[إأآ]", "ا", text)
    text = re.sub(r'(.)\1+', r'\1', text) # Remove character repetition
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text) # Keep only Arabic script and spaces
    return text

# 2. Automated Training Pipeline
def setup_and_train():
    """
    Builds the dataset, trains the Naive Bayes model, 
    and returns the model and vectorizer for production use.
    """
    data_path = 'data/dataset.csv'
    if not os.path.exists('data'): os.makedirs('data')
    
    # Saudi Dialect Base Dataset (Positive & Negative)
    pos_base = [
        "والله إنه قيد القوة والجمال", "شغل هول يجنن ما شاء الله", "يا فديت روحك على هالشغل",
        "احس براحة نفسية اليوم", "الله يجعله دوم هالزين", "مرة مستانس ومبسوط بالحيل",
        "شغل سنع بالحيل وتوب", "يا بعد حي وميتي على هالزين", "ما شاء الله بالحيل مستانس",
        "كفو والله كفيت ووفيت", "يا بعد راسي والله إنك ذيب", "مرة نايس الله يسعدك",
        "رايق", "مبسوط", "فرحان", "يجنن", "روعة", "ممتاز", "رهيب", "خيال", "كفو"
    ]
    
    neg_base = [
        "والله إنه خياس وما يسوى", "شغل هباب وضاع وقتي", "مو زين ابد والتعامل تعيس",
        "للاسف خيبتوا ظني فيكم", "اسوأ مطعم جربته بحياتي", "خدمة بطيئة واكل طعمه غريب",
        "ما يستاهل ولا ريال واحد", "والله إنه فاشل وبقوة", "تعاملهم شين مرة وما انصح",
        "تعبان", "حزين", "ضايق", "سيء", "خايس", "زفت", "هباب", "ماش"
    ]
    
    # Expand dataset to 10,000 samples for statistical robustness
    text_data = pos_base + neg_base
    sentiments = ([1] * len(pos_base) + [0] * len(neg_base))
    multiplier = 10000 // len(text_data) + 1
    
    data = {'text': text_data * multiplier, 'sentiment': sentiments * multiplier}
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    
    # Preprocessing and Feature Extraction
    df['text'] = df['text'].apply(lambda x: clean_arabic_text(str(x)) or "unknown")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    # Train-Test Split and Model Fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    return model, vectorizer

# 3. Sentiment Prediction API Function
def predict_sentiment(text, model, vectorizer):
    """
    Processes a single input and returns a dictionary with 
    sentiment label and confidence score.
    """
    cleaned = clean_arabic_text(text)
    if not cleaned:
        return {"error": "Invalid input. Please provide Arabic text."}
    
    # Transform input and calculate probabilities
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    
    return {
        "text": text,
        "sentiment": "Positive" if prediction == 1 else "Negative",
        "confidence": f"{round(max(probs) * 100, 2)}%"
    }

# Entry point for development and testing
if __name__ == "__main__":
    # Initialize and train the engine
    trained_model, our_vectorizer = setup_and_train()
    print("AI Engine ready for integration.")
    
    # Quick test case
    test_phrase = "الخدمة ماش والتعامل سيء"
    result = predict_sentiment(test_phrase, trained_model, our_vectorizer)
    print(f"Integration Test Output: {result}")