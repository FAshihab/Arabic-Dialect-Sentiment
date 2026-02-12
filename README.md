# Saudi Dialect Sentiment Analysis

## Project Description
This project is a machine learning tool designed to analyze and classify sentiments in Saudi Arabian regional dialects. While most models focus on formal Arabic, this engine is specifically trained to understand local idioms and expressions from various regions of Saudi Arabia, such as the Najd, Hejaz, South, and North.

The project is built as a modular "AI Engine," making it easy for developers or companies to integrate it into their own systems like chatbots or customer feedback tools.

## How it Works
1. **Data Scaling**: The model trains on a balanced dataset of 10,000 samples to ensure high accuracy.
2. **Text Preprocessing**: It includes a cleaning pipeline that normalizes Arabic letters and removes noise.
3. **Machine Learning**: It uses the Multinomial Naive Bayes algorithm combined with TF-IDF vectorization to understand the weight of dialect-specific words.
4. **Confidence Scoring**: For every prediction, the model provides a "Confidence Level" percentage to show how certain it is about the result.

## Technical Stack
- **Language**: Python
- **Libraries**: Pandas, Scikit-learn
- **Model**: Multinomial Naive Bayes

## Quick Integration Example
Developers can call the prediction function directly after training:

```python
from main import setup_and_train, predict_sentiment

# Initialize engine
model, vectorizer = setup_and_train()

# Predict sentiment for a Saudi dialect phrase
text = "الخدمة ماش والتعامل سيء"
result = predict_sentiment(text, model, vectorizer)

print(result)
# Output: {'text': '...', 'sentiment': 'Negative', 'confidence': '99.99%'}