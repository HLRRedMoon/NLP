# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#
# # Load the dataset (adjust the path to where you've uploaded your CSV in Colab)
# data = pd.read_csv("Ar_review10k(1).csv")
#
# # Define a simple preprocessing function to tokenize text
# def preprocess_text(text):
#     # Simple whitespace tokenizer
#     tokens = text.split()
#     return ' '.join(tokens)
#
# # Apply preprocessing to your data
# data['processed_text'] = data['text'].apply(preprocess_text)
#
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(data['processed_text'], data['label'], test_size=0.2, random_state=42)
#
# # Feature extraction using Bag of Words
# vectorizer = CountVectorizer()
# X_train_bow = vectorizer.fit_transform(X_train)
# X_test_bow = vectorizer.transform(X_test)
#
# # One-hot encode labels
# y_train_encoded = pd.get_dummies(y_train)
# y_test_encoded = pd.get_dummies(y_test)
#
# # Build the neural network model
# model = Sequential()
# model.add(Dense(10, activation='relu', input_dim=X_train_bow.shape[1]))
# model.add(Dense(y_train_encoded.shape[1], activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train_bow.toarray(), y_train_encoded, epochs=30, verbose=1)  # Adjust the number of epochs as needed
#
# # Predict on the test data
# y_pred_prob = model.predict(X_test_bow.toarray())
# y_pred = np.argmax(y_pred_prob, axis=1)
#
# # Convert one-hot encoded test labels back to class integers for evaluation
# y_test_classes = np.argmax(y_test_encoded.to_numpy(), axis=1)
#
# # Calculate accuracy, precision, recall, and F1-score
# accuracy = accuracy_score(y_test_classes, y_pred)
# precision = precision_score(y_test_classes, y_pred, average='macro')
# recall = recall_score(y_test_classes, y_pred, average='macro')
# f1 = f1_score(y_test_classes, y_pred, average='macro')
#
# # Print the results
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the dataset (adjust the path to where you've uploaded your CSV in Colab)
data = pd.read_csv("Ar_review10k(1).csv")

# Define a simple preprocessing function to tokenize text
def preprocess_text(text):
    # Simple whitespace tokenizer
    tokens = text.split()
    return ' '.join(tokens)

# Apply preprocessing to your data
data['processed_text'] = data['text'].apply(preprocess_text)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['processed_text'], data['label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# One-hot encode labels
y_train_encoded = pd.get_dummies(y_train)
y_test_encoded = pd.get_dummies(y_test)

# Build the neural network model
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=X_train_tfidf.shape[1]))
model.add(Dense(y_train_encoded.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tfidf.toarray(), y_train_encoded, epochs=20, verbose=1)  # Adjust the number of epochs as needed

# Predict on the test data
y_pred_prob = model.predict(X_test_tfidf.toarray())
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert one-hot encoded test labels back to class integers for evaluation
y_test_classes = np.argmax(y_test_encoded.to_numpy(), axis=1)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test_classes, y_pred)
precision = precision_score(y_test_classes, y_pred, average='macro')
recall = recall_score(y_test_classes, y_pred, average='macro')
f1 = f1_score(y_test_classes, y_pred, average='macro')

# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
