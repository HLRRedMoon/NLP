import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv("Ar_review10k(1).csv")

# Download necessary nltk data
nltk.download('stopwords')

# Set up Arabic stop words
stop_words = set(stopwords.words('arabic'))
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_review(review):
    # Convert to lowercase and tokenize
    tokens = tokenizer.tokenize(review.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into a string
    return ' '.join(filtered_tokens)

# Preprocess all the reviews in the dataset
data['processed_text'] = data['text'].apply(preprocess_review)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['processed_text'], data['label'], test_size=0.2, random_state=42)

# Feature extraction - BoW
vectorizer_bow = CountVectorizer()
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

# Feature extraction - TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)


# Logistic Regression for BoW
lr_bow = LogisticRegression(max_iter=1000)
lr_bow.fit(X_train_bow, y_train)

# Logistic Regression for TF-IDF
lr_tfidf = LogisticRegression(max_iter=1000)
lr_tfidf.fit(X_train_tfidf, y_train)

# Evaluation for Logistic Regression BoW
y_pred_lr_bow = lr_bow.predict(X_test_bow)
accuracy_lr_bow = accuracy_score(y_test, y_pred_lr_bow)
precision_lr_bow = precision_score(y_test, y_pred_lr_bow, average='macro')
recall_lr_bow = recall_score(y_test, y_pred_lr_bow, average='macro')
f1_lr_bow = f1_score(y_test, y_pred_lr_bow, average='macro')

# Evaluation for Logistic Regression TF-IDF
y_pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)
accuracy_lr_tfidf = accuracy_score(y_test, y_pred_lr_tfidf)
precision_lr_tfidf = precision_score(y_test, y_pred_lr_tfidf, average='macro')
recall_lr_tfidf = recall_score(y_test, y_pred_lr_tfidf, average='macro')
f1_lr_tfidf = f1_score(y_test, y_pred_lr_tfidf, average='macro')

print("Logistic Regression BoW - Accuracy:", accuracy_lr_bow, "Precision:", precision_lr_bow, "Recall:", recall_lr_bow, "F1-score:", f1_lr_bow)
print("Logistic Regression TF-IDF - Accuracy:", accuracy_lr_tfidf, "Precision:", precision_lr_tfidf, "Recall:", recall_lr_tfidf, "F1-score:", f1_lr_tfidf)