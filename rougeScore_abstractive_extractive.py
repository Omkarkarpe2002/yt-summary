import rougeScore_dataset as dataset
from rouge import Rouge
from summarization import generatingSummary

# Assuming 'generatedSummary' contains your predicted summaries and 'referenceSummary' contains your referenceSummaryerence summaries

def abstractive_extractive_rough_score():
    randomSample = dataset.dataset()
    referenceSummaryList = []
    generatedSummaryList = []  
    for index, row in randomSample.iterrows():
        referenceSummaryList.append(row['summary'])
        generatedSummaryList.append(generatingSummary(row['document']))
    rouge = Rouge()
    scores = rouge.get_scores(generatedSummaryList, referenceSummaryList, avg=True)
    # Access individual ROUGE scores
    rouge_1_score = scores['rouge-1']['f']
    rouge_2_score = scores['rouge-2']['f']
    rouge_l_score = scores['rouge-l']['f']
    print("our-model Accuracy")
    print(f"ROUGE-1 Score: {rouge_1_score}")
    print(f"ROUGE-2 Score: {rouge_2_score}")
    print(f"ROUGE-L Score: {rouge_l_score}")

    

abstractive_extractive_rough_score()



# bart_model, led_model, pegasus_model, t5_model, bert_model, gpt2_model, textrank_algorithm, xlnet_model, referenceSummaryList, generatedSummaryList = dataset.reference_predicted_summary()

# # Initialize the Rouge object


# # Calculate ROUGE scores
# print("our-model")
# scores = rouge.get_scores(generatedSummaryList, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")


# print(bart_model)
# scores = rouge.get_scores(bart_model, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")

# print(led_model)
# scores = rouge.get_scores(led_model, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")

# print(pegasus_model)
# scores = rouge.get_scores(pegasus_model, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")

# print(t5_model)
# scores = rouge.get_scores(t5_model, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")

# print(bert_model)
# scores = rouge.get_scores(bert_model, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")

# print(gpt2_model)
# scores = rouge.get_scores(gpt2_model, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")

# print(textrank_algorithm)
# scores = rouge.get_scores(textrank_algorithm, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")

# print(xlnet_model)
# scores = rouge.get_scores(xlnet_model, referenceSummaryList, avg=True)
# # Access individual ROUGE scores
# rouge_1_score = scores['rouge-1']['f']
# rouge_2_score = scores['rouge-2']['f']
# rouge_l_score = scores['rouge-l']['f']

# # Print the scores
# print(f"ROUGE-1 Score: {rouge_1_score}")
# print(f"ROUGE-2 Score: {rouge_2_score}")
# print(f"ROUGE-L Score: {rouge_l_score}")








# # def reference_predicted_summary():
# #     filtered_df = dataset()
# #     random_subset = filtered_df.sample(n=1)
# #     bart_model = []
# #     led_model = []
# #     pegasus_model = []
# #     t5_model = []
# #     bert_model = []
# #     gpt2_model = []
# #     textrank_algorithm = []
# #     xlnet_model = []
# #     referenceSummaryList = []
# #     generatedSummaryList = []  
# #     for index, row in random_subset.iterrows():
# #         referenceSummaryList.append(row['summary'])
# #         generatedSummaryList.append(summarization.generatingSummary(row['document']))
# #         bart_model.append(BART_Text_Summarization.summarizer(row['document']))
# #         led_model.append(LED_Text_Summarization.summary(row['document']))
# #         pegasus_model.append(Pegasus_Text_Summarization.summarize_text(row['document']))
# #         t5_model.append(T5_Text_Summarization.summarize_text_t5(row['document']))
# #         bert_model.append(BERT_Extractive_Text_Summarization.extractive_summarization_bert(row['document']))
# #         gpt2_model.append(GPT2_Extractive_Text_Summarization.extractive_summarization_gpt2(row['document']))
# #         textrank_algorithm.append(TextRank_Extractive.textrank_summarize(row['document']))
# #         xlnet_model.append(XLNet_Extractive_Text_Summarization.extractive_summarization_xlnet(row['document']))
# #     return (bart_model , led_model , pegasus_model , t5_model , bert_model , gpt2_model , textrank_algorithm , xlnet_model , referenceSummaryList , generatedSummaryList)

