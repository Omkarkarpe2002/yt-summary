from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def summarize_text(input_text):
    # Load PEGASUS model and tokenizer
    model_name = "google/pegasus-large"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and generate summary
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Example usage
input_text = """
Paste your long text here that you want to summarize.
It can be an article, a blog post, or any other form of text.
"""
summary = summarize_text(input_text)

print("Original Text:")
print(input_text)
print("\nSummarized Text:")
print(summary)
