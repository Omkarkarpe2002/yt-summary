from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize

def extractive_summarization_bert(original_text, num_sentences=3):
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize and encode the original text
    tokenized_text = tokenizer(original_text, return_tensors='pt', max_length=512, truncation=True)
    
    # Get BERT embeddings for each token in the input text
    with torch.no_grad():
        outputs = model(**tokenized_text)

    # Use the output embeddings for the [CLS] token as sentence embeddings
    sentence_embeddings = outputs.last_hidden_state[:, 0, :]

    # Calculate pairwise cosine similarity between sentence embeddings
    similarity_matrix = cosine_similarity(sentence_embeddings, sentence_embeddings)

    # Extract top sentences based on similarity score
    sentence_scores = np.sum(similarity_matrix, axis=1)
    top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]

    # Sort the selected sentences in their original order
    top_sentence_indices.sort()

    # Tokenize and get the top sentences from the original text
    sentences = sent_tokenize(original_text)
    summary = ' '.join([sentences[i] for i in top_sentence_indices])

    return summary

# Example usage
original_text = """
Paste your long text here that you want to summarize.
It can be an article, a blog post, or any other form of text.
"""
summary = extractive_summarization_bert(original_text)

print("Original Text:")
print(original_text)
print("\nExtractive Summary:")
print(summary)



# import nltk

# # Download the 'punkt' resource
# nltk.download('punkt')
