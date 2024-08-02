from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and the TfidfVectorizer
model = joblib.load('model/spam_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

# Define custom spam words
custom_spam_words = set([
    'free', 'win', 'money', 'prize', 'click', 'subscribe', 'guarantee', 'winner'
])

def contains_custom_spam_words(text):
    words = set(text.lower().split())
    return not custom_spam_words.isdisjoint(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    message_transformed = tfidf.transform([message])
    prediction = model.predict(message_transformed)
    
    # Determine if custom spam words are present
    custom_spam_detected = contains_custom_spam_words(message)

    # Adjust prediction based on custom spam words
    if custom_spam_detected:
        prediction = [1]  # Treat as spam if custom words are detected
    
    output = 'Spam' if prediction[0] == 1 else 'Ham'
    
    return render_template('index.html', prediction_text=f'This message is: {output}')

if __name__ == '__main__':
    app.run(debug=True)
