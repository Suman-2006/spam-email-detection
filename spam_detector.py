# Congratulations! You’ve won a brand new iPhone! Click here to claim your prize: [bit.ly/win-now]
#You have an urgent message from PayPal. Please update your account information here: [phishingsite.com]

# spam_detector.py


#libraries

# Data handling
import pandas as pd

# text processing
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#  web app
import streamlit as st

# Download NLTK data
# Stopwords = common words that don’t add meaning.
# We remove them to make our spam detector focus on meaningful words.
nltk.download('stopwords')

# Load and prepare dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
# renames
data.columns = ['label', 'message']

# Convert labels to binary 
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

#Removes common words and processes text
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# define function
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation]) # Removes punctuation (!, ., ?)
    words = text.split() # Splits into words
    words = [stemmer.stem(w) for w in words if w not in stop_words] # Removes stopwords
    return ' '.join(words) 

data['cleaned'] = data['message'].apply(clean_text)# Joins back into a cleaned sentence

# Split data (80% for training 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization  (Converts text into numbers)
# (TF-IDF helps the model focus on important words like "win", "free", "click") 
tfidf = TfidfVectorizer(max_features=3000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train Model
# Logistic Regression helps the computer decide if a message is spam or not based on the words it contains.
# It’s easy, fast, and gives accurate results for this kind of text classification.
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate (Predicts labels on test data)
y_pred = model.predict(X_test_vec)
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix Visualization
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Web app
st.title("📧 Spam Email Detector")
st.write("Enter a message below to check if it's spam or not:")

user_input = st.text_area("Message:")

if st.button("Predict"):
    cleaned_input = clean_text(user_input)
    input_vec = tfidf.transform([cleaned_input])
    prediction = model.predict(input_vec)[0]
    confidence = model.predict_proba(input_vec).max()

    if prediction == 1:
        st.error(f"🚫 SPAM (Confidence: {confidence:.2f})")
    else:
        st.success(f"✅ HAM (Not Spam) (Confidence: {confidence:.2f})")

# Classification report in console
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
