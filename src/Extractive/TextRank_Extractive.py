# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# import networkx as nx
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity  # Add this import

# nltk.download('punkt')
# nltk.download('stopwords')

# def preprocess_text(text):
#     sentences = sent_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     preprocessed_sentences = [
#         ' '.join([word.lower() for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words])
#         for sentence in sentences
#     ]
#     return preprocessed_sentences

# def textrank_summarize(text, num_sentences=3):
#     preprocessed_sentences = preprocess_text(text)

#     # Use TF-IDF for sentence embeddings
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

#     # Calculate cosine similarity between sentences
#     similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

#     # Use the PageRank algorithm to rank sentences
#     graph = nx.from_numpy_array(similarity_matrix)
#     scores = nx.pagerank(graph)

#     # Sort sentences based on their PageRank scores
#     ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(preprocessed_sentences)), reverse=True)

#     # Extract the top-ranked sentences to form the summary
#     summary = ' '.join([sentence for score, sentence in ranked_sentences[:num_sentences]])

#     return summary

# # Example usage
# original_text = """
# Paste your long text here that you want to summarize.
# It can be an article, a blog post, or any other form of text.
# """
# summary = textrank_summarize(original_text)

# print("Original Text:")
# print(original_text)
# print("\nTextRank Summary:")
# print(summary)


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = [
        ' '.join([word.lower() for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words])
        for sentence in sentences
    ]
    return preprocessed_sentences

def textrank_summarize(text, max_words=1000):
    preprocessed_sentences = preprocess_text(text)

    # Use TF-IDF for sentence embeddings
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    # Calculate cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Use the PageRank algorithm to rank sentences
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Sort sentences based on their PageRank scores
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(preprocessed_sentences)), reverse=True)

    # Extract sentences until the word limit is reached
    current_words = 0
    selected_sentences = []
    for score, sentence in ranked_sentences:
        words_in_sentence = len(sentence.split())
        if current_words + words_in_sentence <= max_words:
            selected_sentences.append(sentence)
            current_words += words_in_sentence
        else:
            break

    # Combine selected sentences to form the summary
    summary = ' '.join(selected_sentences)

    return summary

# Example usage
original_text = """
Paste your long text here that you want to summarize.
It can be an article, a blog post, or any other form of text.
"""
summary = textrank_summarize(original_text, max_words=50)

print("Original Text:")
print(original_text)
print("\nTextRank Summary:")
print(summary)
