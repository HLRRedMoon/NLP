# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import RegexpTokenizer
# from sklearn.metrics import precision_score, recall_score, f1_score
#
# # Load the IMDB dataset
# # download from here
# data = pd.read_csv("Ar_review10k(1).csv")
#
# nltk.download('stopwords')
# nltk.download('wordnet')
#
# stop_words = set(stopwords.words('arabic'))
# lemmatizer = WordNetLemmatizer()
# tokenizer = RegexpTokenizer(r'\w+')
#
#
# def preprocess_review(review):
#     # Convert the review to lowercase
#     review = review.lower()
#
#     # Tokenize the review
#     tokens = tokenizer.tokenize(review)
#
#     # Remove stop words from the review
#     filtered_tokens = [token for token in tokens if token not in stop_words]
#
#     # Lemmatize the filtered tokens
#     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
#
#     # Join the lemmatized tokens back into a single string
#     preprocessed_review = ' '.join(lemmatized_tokens)
#
#     return preprocessed_review
#
#
# # Preprocess all the reviews in the dataset
# data['Positive'] = data['text'].apply(preprocess_review)
#
# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(data['Positive'], data['text'], test_size=0.2, random_state=42)
#
# # Create a CountVectorizer object to convert the text data into numerical feature vectors
# vectorizer = CountVectorizer(stop_words='english')
#
# # Fit the vectorizer on the training data and transform it into numerical feature vectors
# X_train_vectors = vectorizer.fit_transform(X_train)
# X_test_vectors = vectorizer.transform(X_test)
#
# # Create a Multinomial Naive Bayes classifier
# clf = MultinomialNB()
#
# # Train the classifier on the training data
# clf.fit(X_train_vectors, y_train)
#
# # Make predictions on the test data
# y_pred = clf.predict(X_test_vectors)
#
# # Calculate the accuracy precision, recall, and F1-score
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average=None)
# recall = recall_score(y_test, y_pred, average=None)
# f1 = f1_score(y_test, y_pred, average=None)
#
# print("Accuracy", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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

# Create and train a Multinomial Naive Bayes classifier for BoW
clf_bow = MultinomialNB()
clf_bow.fit(X_train_bow, y_train)

# Create and train a Multinomial Naive Bayes classifier for TF-IDF
clf_tfidf = MultinomialNB()
clf_tfidf.fit(X_train_tfidf, y_train)

# Evaluate the classifier using BoW
y_pred_bow = clf_bow.predict(X_test_bow)
accuracy_bow = accuracy_score(y_test, y_pred_bow)
precision_bow = precision_score(y_test, y_pred_bow, average='macro')
recall_bow = recall_score(y_test, y_pred_bow, average='macro')
f1_bow = f1_score(y_test, y_pred_bow, average='macro')

# Evaluate the classifier using TF-IDF
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
precision_tfidf = precision_score(y_test, y_pred_tfidf, average='macro')
recall_tfidf = recall_score(y_test, y_pred_tfidf, average='macro')
f1_tfidf = f1_score(y_test, y_pred_tfidf, average='macro')

print("BoW Model - Accuracy:", accuracy_bow, "Precision:", precision_bow, "Recall:", recall_bow, "F1-score:", f1_bow)
print("TF-IDF Model - Accuracy:", accuracy_tfidf, "Precision:", precision_tfidf, "Recall:", recall_tfidf, "F1-score:", f1_tfidf)
