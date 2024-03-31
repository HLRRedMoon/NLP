#Quisetion 1
# import re
# import requests
# from collections import defaultdict
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import nltk
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
#
# def get_gutenberg_text(url):
#     # Fetch the text from the provided URL
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.text
#     else:
#         print("Failed to fetch data from the provided URL.")
#         return None
#
#
# def create_unigram_model(text):
#     unigram_model = defaultdict(int)
#     # Tokenize the text into words
#     words = word_tokenize(text)
#
#     # Filter out non-alphabetic words and convert to lowercase
#     words = [word.lower() for word in words if word.isalpha()]
#
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if word not in stop_words]
#
#     # Count occurrences of each word
#     for word in words:
#         unigram_model[word] += 1
#
#     return unigram_model
#
#
# def main():
#     # Example URL for Project Gutenberg text
#     url = "https://www.gutenberg.org/files/1342/1342-0.txt"
#
#     # Get text from Project Gutenberg
#     text = get_gutenberg_text(url)
#
#     if text:
#         # Create unigram model
#         unigram_model = create_unigram_model(text)
#
#         # Print the first 20 words and their counts
#         print("First 20 words and their counts in the unigram model:")
#         count = 0
#         for word, frequency in unigram_model.items():
#             print(f"{word}: {frequency}")
#             count += 1
#             if count >= 20:
#                 break
#     else:
#         print("Failed to retrieve text from Project Gutenberg.")
#
#
# main()
# if __name__ == "__main__":
#     pass


#muhaned code ********************************************

import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random
corpus = nltk.corpus.gutenberg.words('austen-emma.txt')

def unigram():
    text = nltk.corpus.gutenberg.raw("shakespeare-caesar.txt")

    # Create a unigram model
    unigram_model = nltk.FreqDist(text.split())

    # Generate a sentence
    sentence = ""
    for i in range(0, 9):
        for word in random.choices(unigram_model.most_common(10)):
            sentence += word[0] + " "
    return sentence


def create_ngram_model(n, corpus):

    """
    Create an n-gram model using a dictionary of English vocabulary
    collected from any corpus of Project Gutenberg.

    :param n: The value of n for the n-gram model.
    :param corpus: A list of words representing the corpus.
    :return: A dictionary representing the n-gram model.
    """
    ngrams = list(nltk.ngrams(corpus, n))
    model = defaultdict(lambda: defaultdict(lambda: 0))
    for ngram in ngrams:
        model[ngram[:-1]][ngram[-1]] += 1
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
    return model


def generate_sentence(model, initial_words, num_words):
    """
    Generate a sentence of a given number of words from some initial words
    using a previously learned n-gram model.

    :param initial_words: A tuple of initial words to start the sentence with.
    :param num_words: The number of words to generate in the sentence.
    :return: A list of words representing the generated sentence.
    """
    sentence = list(initial_words)
    for i in range(num_words):
        next_word = random.choices(list(model[tuple(sentence[-len(initial_words):])].keys()),
                                   list(model[tuple(sentence[-len(initial_words):])].values()))[0]
        sentence.append(next_word)
    return sentence

# Load the text corpus

# Test drive code
unigram_model = unigram()
bigram_model = create_ngram_model(2, corpus)
trigram_model = create_ngram_model(3, corpus)

initial_words = random.choices(corpus)
initial_words2 = ("the", "ladies")

s2 = generate_sentence(bigram_model, initial_words, 10)
s3 = generate_sentence(trigram_model, initial_words2, 10)
print(''.join(unigram_model))
print(' '.join(s2))
print(' '.join(s3))




#chat code ****************
# import re
# import requests
# from collections import defaultdict
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import random
# import nltk
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
#
# def get_gutenberg_text(url):
#     # Fetch the text from the provided URL
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.text
#     else:
#         print("Failed to fetch data from the provided URL.")
#         return None
#
#
# def create_unigram_model(text):
#     unigram_model = defaultdict(int)
#     # Tokenize the text into words
#     words = word_tokenize(text)
#
#     # Filter out non-alphabetic words and convert to lowercase
#     words = [word.lower() for word in words if word.isalpha()]
#
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if word not in stop_words]
#
#     # Count occurrences of each word
#     for word in words:
#         unigram_model[word] += 1
#
#     return unigram_model
#
#
# def create_bigram_model(text):
#     bigram_model = defaultdict(lambda: defaultdict(int))
#     # Tokenize the text into words
#     words = word_tokenize(text)
#
#     # Filter out non-alphabetic words and convert to lowercase
#     words = [word.lower() for word in words if word.isalpha()]
#
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if word not in stop_words]
#
#     # Create bigrams and count occurrences
#     for i in range(len(words) - 1):
#         current_word = words[i]
#         next_word = words[i + 1]
#         bigram_model[current_word][next_word] += 1
#
#     return bigram_model
#
#
# def create_trigram_model(text):
#     trigram_model = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
#     # Tokenize the text into words
#     words = word_tokenize(text)
#
#     # Filter out non-alphabetic words and convert to lowercase
#     words = [word.lower() for word in words if word.isalpha()]
#
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if word not in stop_words]
#
#     # Create trigrams and count occurrences
#     for i in range(len(words) - 2):
#         current_word = words[i]
#         next_word = words[i + 1]
#         next_next_word = words[i + 2]
#         trigram_model[current_word][next_word][next_next_word] += 1
#
#     return trigram_model
#
#
# def generate_sentence(ngram_model, length=10, initial_words=None):
#     sentence = []
#     if initial_words:
#         # Use initial words to start the sentence
#         sentence.extend(initial_words)
#     else:
#         # Choose a random starting point from the model
#         initial_words = random.choice(list(ngram_model.keys()))
#         sentence.append(initial_words)
#
#     while len(sentence) < length:
#         current_word = sentence[-1]
#         # Check if the current word exists in the model
#         if current_word in ngram_model:
#             next_word_options = list(ngram_model[current_word].keys())
#             if next_word_options:
#                 next_word = random.choice(next_word_options)
#                 sentence.append(next_word)
#             else:
#                 break
#         else:
#             break
#
#     return ' '.join(sentence)
#
#
# # Example URL for Project Gutenberg text
# url = "https://www.gutenberg.org/files/1342/1342-0.txt"
#
# # Get text from Project Gutenberg
# text = get_gutenberg_text(url)
#
# if text:
#     # Create unigram model
#     unigram_model = create_unigram_model(text)
#
#     # Create bigram model
#     bigram_model = create_bigram_model(text)
#
#     # Create trigram model
#     trigram_model = create_trigram_model(text)
#
#     # Generate sentences using the n-gram models
#     print("Unigram model sentence:")
#     print(generate_sentence(unigram_model))
#     print("\nBigram model sentence:")
#     print(generate_sentence(bigram_model))
#     print("\nTrigram model sentence:")
#     print(generate_sentence(trigram_model))
# else:
#     print("Failed to retrieve text from Project Gutenberg.")
