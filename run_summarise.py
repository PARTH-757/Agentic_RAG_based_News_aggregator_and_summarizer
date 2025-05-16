from summarize import load_bart_model_and_tokenizer, summarize_and_store_in_batches
# Load BART model and tokenizer
#bart_tokenizer, bart_model = load_bart_model_and_tokenizer()
bart_tokenizer, bart_model, bart_timestamp = load_bart_model_and_tokenizer()

# Then call the function with the required arguments:
summarize_and_store_in_batches(df, bart_tokenizer, bart_model, batch_size=4, output_file="summarized_news_batch.csv")
