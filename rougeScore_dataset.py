from datasets import load_dataset
import pandas as pd

def dataset():
    # Load XSum dataset
    xsum = load_dataset("xsum")
    # Access the validation split
    validation_data = xsum["validation"]
    # Convert validation_data to a Pandas DataFrame
    df = pd.DataFrame(validation_data)
    # Add a new column 'document_word_count' containing the word count of each document
    df['document_word_count'] = df['document'].apply(lambda x: len(x.split()))
    # Filter rows where 'document_word_count' is greater than or equal to 1000
    filtered_df = df[df['document_word_count'] >= 1000]
    # Optionally, you can drop the 'document_word_count' column if you no longer need it
    filtered_df = filtered_df.drop(columns=['document_word_count'])
    # Generating random subset
    random_subset = filtered_df.sample(n=10, random_state=42)
    return filtered_df

