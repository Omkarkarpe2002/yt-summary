from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text_t5(input_text):
    # Load T5 model and tokenizer
    model_name = "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenize and generate summary
    inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Example usage
input_text = """
Paste your long text here that you want to summarize.
It can be an article, a blog post, or any other form of text.
"""
summary = summarize_text_t5(input_text)

print("Original Text:")
print(input_text)
print("\nSummarized Text:")
print(summary)
