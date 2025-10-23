# spam_detector.py

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk
# import string
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import streamlit as st

# # Download stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords

# stop_words = set(stopwords.words('english'))

# # -----------------------
# # 1. Load Data
# # -----------------------
# @st.cache_data
# def load_data():
#     # df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
#     df = pd.read_csv('spam.csv', encoding='latin-1', sep='\t')[['v1', 'v2']]
#     Index(['label', 'message'], dtype='object')

#     df.columns = ['label', 'message']
#     df['label_num'] = df.label.map({'ham':0, 'spam':1})
#     return df

# df = load_data()

# # -----------------------
# # 2. Preprocess Text
# # -----------------------
# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'\d+', '', text) # Remove numbers
#     text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
#     words = text.split()
#     words = [word for word in words if word not in stop_words]
#     return " ".join(words)

# df['clean_message'] = df['message'].apply(preprocess_text)

# # -----------------------
# # 3. Feature Extraction
# # -----------------------
# tfidf = TfidfVectorizer(max_features=3000)
# X = tfidf.fit_transform(df['clean_message'])
# y = df['label_num']

# # -----------------------
# # 4. Train-Test Split
# # -----------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # -----------------------
# # 5. Model Training
# # -----------------------
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # -----------------------
# # 6. Model Evaluation
# # -----------------------
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# # Visualize Confusion Matrix
# plt.figure(figsize=(6,4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig('confusion_matrix.png')

# print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:\n", report)

# # -----------------------
# # 7. Streamlit Interface
# # -----------------------
# st.title("Spam Email Detection System")
# st.write("Enter your email/SMS text below to check if it's spam or not.")

# user_input = st.text_area("Email Text:")

# if st.button("Predict"):
#     if not user_input.strip():
#         st.warning("Please enter some text to classify.")
#     else:
#         clean_input = preprocess_text(user_input)
#         vector_input = tfidf.transform([clean_input])
#         prediction = model.predict(vector_input)[0]
#         proba = model.predict_proba(vector_input)[0][prediction]

#         label_text = "SPAM" if prediction == 1 else "HAM (Not Spam)"
#         st.success(f"Prediction: {label_text}")
#         st.info(f"Confidence Score: {proba:.2f}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# -----------------------
# NLTK Setup
# -----------------------
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# -----------------------
# 1. Load Dataset
# -----------------------
@st.cache_data
def load_data():
    # Make sure the file "spam.csv" exists in the same folder.
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df = df[['label', 'message']]
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# -----------------------
# 2. Preprocess Text
# -----------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_message'] = df['message'].apply(preprocess_text)

# -----------------------
# 3. Feature Extraction
# -----------------------
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_message'])
y = df['label_num']

# -----------------------
# 4. Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 5. Model Training
# -----------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------
# 6. Model Evaluation
# -----------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# -----------------------
# 7. Streamlit Interface
# -----------------------
st.title("📧 Spam Email Detection System")
st.write("Enter an email or SMS message below and find out if it's spam or not!")

st.markdown(f"**Model Accuracy:** {accuracy:.2%}")

user_input = st.text_area("✉️ Type your message:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        clean_input = preprocess_text(user_input)
        vector_input = tfidf.transform([clean_input])
        prediction = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0][prediction]

        label_text = "🚫 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
        st.success(f"**Prediction:** {label_text}")
        st.info(f"**Confidence Score:** {proba:.2f}")

# Optionally display dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk
# import string
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import streamlit as st

# # -----------------------
# # 0. NLTK stopwords
# # -----------------------
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# # -----------------------
# # 1. Load Data
# # -----------------------
# @st.cache_data
# def load_data():
#     try:
#         # Try reading with tab separator
#         df = pd.read_csv('spam.csv', encoding='latin-1', sep='\t')
#     except:
#         # fallback to comma separator
#         df = pd.read_csv('spam.csv', encoding='latin-1', sep=',')

#     # Adjust columns
#     if 'v1' in df.columns and 'v2' in df.columns:
#         df = df[['v1','v2']]
#         df.columns = ['label','message']
#     elif 'label' in df.columns and 'message' in df.columns:
#         df = df[['label','message']]
#     else:
#         # No headers fallback
#         df = pd.read_csv('spam.csv', encoding='latin-1', sep='\t', header=None)
#         df = df[[0,1]]
#         df.columns = ['label','message']

#     df['label_num'] = df['label'].map({'ham':0,'spam':1})
#     return df

# df = load_data()

# # -----------------------
# # 2. Preprocess Text
# # -----------------------
# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
#     words = text.split()
#     words = [word for word in words if word not in stop_words]
#     return " ".join(words)

# df['clean_message'] = df['message'].apply(preprocess_text)

# # -----------------------
# # 3. Feature Extraction
# # -----------------------
# tfidf = TfidfVectorizer(max_features=3000)
# X = tfidf.fit_transform(df['clean_message'])
# y = df['label_num']

# # -----------------------
# # 4. Train-Test Split
# # -----------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # -----------------------
# # 5. Model Training
# # -----------------------
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # -----------------------
# # 6. Model Evaluation
# # -----------------------
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# # Visualize Confusion Matrix
# plt.figure(figsize=(6,4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig('confusion_matrix.png')

# print(f"Accuracy: {accuracy:.4f}")
# print("Classification Report:\n", report)

# # -----------------------
# # 7. Streamlit Interface
# # -----------------------
# st.title("Spam Email/SMS Detection System")
# st.write("Enter your email or SMS text below to check if it's spam or not:")

# user_input = st.text_area("Enter Text:")

# if st.button("Predict"):
#     if not user_input.strip():
#         st.warning("Please enter some text to classify.")
#     else:
#         clean_input = preprocess_text(user_input)
#         vector_input = tfidf.transform([clean_input])
#         prediction = model.predict(vector_input)[0]
#         proba = model.predict_proba(vector_input)[0][prediction]

#         label_text = "SPAM" if prediction == 1 else "HAM (Not Spam)"
#         st.success(f"Prediction: {label_text}")
#         st.info(f"Confidence Score: {proba:.2f}")


# spam_detector.py

# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # -------------------------------
# # Step 1: Load dataset
# # -------------------------------
# def load_data(csv_file='spam.csv'):
#     """
#     Load a CSV file containing spam/ham messages.
#     Handles different encodings and column names.
#     """
#     try:
#         # Try reading with utf-8
#         df = pd.read_csv(csv_file, encoding='utf-8')
#     except UnicodeDecodeError:
#         # Fallback to latin-1 encoding
#         df = pd.read_csv(csv_file, encoding='latin-1')

#     # Rename columns if using UCI dataset format
#     if 'v1' in df.columns and 'v2' in df.columns:
#         df = df.rename(columns={'v1':'label', 'v2':'message'})

#     # Keep only label and message columns
#     if 'label' not in df.columns or 'message' not in df.columns:
#         raise ValueError("CSV must have 'label' and 'message' columns")
    
#     df = df[['label', 'message']]
    
#     # Map labels to numeric: spam=1, ham=0
#     df['label'] = df['label'].map({'spam':1, 'ham':0})
    
#     return df

# # -------------------------------
# # Step 2: Preprocess text
# # -------------------------------
# def clean_text(text):
#     """
#     Clean the message text: lowercase, remove punctuation, numbers, extra spaces
#     """
#     text = str(text).lower()
#     text = re.sub(r'\W', ' ', text)       # remove non-word characters
#     text = re.sub(r'\d', '', text)        # remove numbers
#     text = re.sub(r'\s+', ' ', text)      # remove extra spaces
#     text = text.strip()
#     return text

# # -------------------------------
# # Step 3: Train model
# # -------------------------------
# def train_model(df):
#     """
#     Train a Logistic Regression model on the preprocessed dataset
#     """
#     # Clean all messages
#     df['message'] = df['message'].apply(clean_text)
    
#     # Split dataset
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['message'], df['label'], test_size=0.2, random_state=42)
    
#     # Convert text to numeric features using TF-IDF
#     vectorizer = TfidfVectorizer()
#     X_train_features = vectorizer.fit_transform(X_train)
#     X_test_features = vectorizer.transform(X_test)
    
#     # Train Logistic Regression
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train_features, y_train)
    
#     # Evaluate
#     y_pred = model.predict(X_test_features)
#     print("Accuracy on test set:", accuracy_score(y_test, y_pred))
#     print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
#     return model, vectorizer

# # -------------------------------
# # Step 4: Predict new messages
# # -------------------------------
# def predict_messages(model, vectorizer, messages):
#     """
#     Predict spam/ham for a list of messages
#     """
#     messages_clean = [clean_text(msg) for msg in messages]
#     features = vectorizer.transform(messages_clean)
#     predictions = model.predict(features)
#     return ['Spam' if p==1 else 'Ham' for p in predictions]

# # -------------------------------
# # Step 5: Main
# # -------------------------------
# if __name__ == "__main__":
#     # Load dataset
#     df = load_data("spam.csv")  # replace with your CSV path
    
#     # Train model
#     model, vectorizer = train_model(df)
    
#     # Test with new messages
#     new_messages = [
#         "Congratulations! You have won a $1000 gift card. Click here to claim now!",
#         "Hey, are we meeting at 6 PM tonight?",
#         "Earn $5000/week from home! Sign up now!",
#         "Dinner is ready. Come to the dining room."
#     ]
    
#     results = predict_messages(model, vectorizer, new_messages)
    
#     for msg, res in zip(new_messages, results):
#         print(f"Message: {msg}\nPrediction: {res}\n")
