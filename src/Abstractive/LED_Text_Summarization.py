from transformers import LEDTokenizer, LEDForConditionalGeneration, LEDConfig
def summary(sequence):
    tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
    model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
    inputs = tokenizer([sequence], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'])
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return summary