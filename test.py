def levenshtein_distance(word1, word2):
    m, n = len(word1), len(word2)

    # Create a 2D matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the matrix with the base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Populate the matrix using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 2

            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                          dp[i][j - 1] + 1,  # Insertion
                          dp[i - 1][j - 1] + cost)  # Substitution

    # The bottom-right cell contains the minimum edit distance
    return dp[m][n]


import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg
from nltk.corpus import stopwords


def create_vocabulary_dict(corpus_name):
    # Download the NLTK resources if not already downloaded
    # import nltk
    # nltk.download('punkt')
    # nltk.download('stopwords')

    # Get the list of words from the specified Gutenberg corpus
    raw_text = gutenberg.raw(corpus_name)

    # Tokenize the text into words
    words = word_tokenize(raw_text)

    # Remove punctuation, convert to lowercase, and filter out non-alphabetic words
    words = [word.lower() for word in words if word.isalpha()]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Count the frequency of each word
    word_counts = Counter(words)

    # Create a dictionary with words as keys and their frequencies as values
    vocabulary_dict = dict(word_counts)

    return vocabulary_dict


def spelling_checker(misspelled_word, vocabulary_dict, distance_function):
    # Calculate Levenshtein distance between the misspelled word and all words in the dictionary
    distances = [(word, distance_function(misspelled_word, word)) for word in vocabulary_dict]

    # Filter out words that are not valid English words
    valid_words = [word for word, _ in distances if word in vocabulary_dict]

    # Sort the valid words based on their Levenshtein distance (lower is better)
    sorted_distances = sorted(distances, key=lambda x: x[1])

    # Extract the five most likely corrected words among the valid words
    top_five_corrections = [word for word, _ in sorted_distances if word in valid_words][:5]

    return top_five_corrections


# Example usage:
corpus_name = 'shakespeare-caesar.txt'
vocabulary_dict = create_vocabulary_dict(corpus_name)

misspelled_word = 'entr'
corrections = spelling_checker(misspelled_word, vocabulary_dict, levenshtein_distance)
print(f"Top five corrections for '{misspelled_word}': {corrections}")

