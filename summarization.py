from transformers import BartTokenizer, BartForConditionalGeneration
from summa import summarizer
import transcribe


def abstractiveSummarization(subtitles, max_words=50):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained(
        'facebook/bart-large-cnn')
    inputs = tokenizer(subtitles, return_tensors="pt",
                       max_length=1024, truncation=True)

    # Adjust max_length based on the desired word limit
    max_length = max_words * 10  # Assuming an average of 10 words per token

    summary_ids = model.generate(
        inputs['input_ids'], max_length=max_length, num_beams=4, length_penalty= 20, early_stopping=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=False)
    return summary

def extractiveSummarization(original_text, word_limit=1000):
    # Summarize using TextRank algorithm with a specified word limit
    summary = summarizer.summarize(original_text, words=word_limit)
    return summary

def count_words(text):
    words = text.split()
    return len(words)

def summarizeText(videoLink):
    subtitles = transcribe.download_subtitles_from_video(videoLink)
    len_subtitles = count_words(subtitles)
    if len_subtitles <= 1024:
        return abstractiveSummarization(subtitles)
    else:
        extractiveSummary = extractiveSummarization(subtitles)
        summary = abstractiveSummarization(extractiveSummary)
        summary = summary.replace("<s>", "").replace("</s>", "")
        return summary
    
def generatingSummary(subtitles):
    len_subtitles = count_words(subtitles)
    if len_subtitles <= 1024:
        return abstractiveSummarization(subtitles)
    else:
        extractiveSummary = extractiveSummarization(subtitles)
        summary = abstractiveSummarization(extractiveSummary)
        summary = summary.replace("<s>", "").replace("</s>", "")
        return summary