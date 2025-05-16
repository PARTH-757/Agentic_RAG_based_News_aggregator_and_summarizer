from transformers import BartTokenizer, BartForConditionalGeneration
import spacy
import pandas as pd
import os
import time  # ✅ Import time for timestamps

# ✅ Load models once to avoid reloading
def load_bart_model_and_tokenizer(model_name="facebook/bart-large-cnn"):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model, int(time.time())  # ✅ Return load timestamp

def load_spacy_model(model_name="en_core_web_sm"):
    nlp = spacy.load(model_name)
    return nlp, int(time.time())  # ✅ Return load timestamp

# ✅ Summary generation with empty text handling
def generate_summary(text, bart_tokenizer, bart_model, max_input_length=1024, max_output_length=150):
    if not text.strip():
        return ""  # ✅ Return empty summary if text is empty
    
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True)
    summary_ids = bart_model.generate(
        inputs.input_ids,
        max_length=max_output_length,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ Extract keywords (passing `nlp` as an argument)
def extract_keywords(text, nlp, num_keywords=5):
    if not text.strip():
        return ""  # ✅ Handle empty text case
    
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return ", ".join(list(dict.fromkeys(keywords))[:num_keywords])  # ✅ Preserve order, remove duplicates

# ✅ Summarize and store in batches
def summarize_and_store_in_batches(df, bart_tokenizer, bart_model, nlp, batch_size=4, output_file="summarized_news_batch.csv"):
    """
    Summarize articles in batches using BART and save the results with a timestamp.
    """
    print("Columns in DataFrame:", df.columns.tolist())  # Debugging step

    if "cleaned_text" not in df.columns:
        print("⚠️ 'cleaned_text' column not found! Available columns:", df.columns.tolist())
        return  # Stop execution if required column is missing

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_data = []  # ✅ Reset batch data inside loop

        first_batch = not os.path.exists(output_file) if i == 0 else False  # ✅ Check for first batch inside loop

        for _, row in batch.iterrows():
            cleaned_text = row.get("cleaned_text", "")  # ✅ Use .get() to avoid KeyError
            summary = generate_summary(cleaned_text, bart_tokenizer, bart_model)
            keywords = extract_keywords(summary, nlp)  # ✅ Pass `nlp`

            batch_data.append({
                "title": row["title"],
                "source": row["source"],
                "url": row["url"],
                "summary": summary,
                "keywords": keywords,
                "timestamp": int(time.time())
            })

        batch_df = pd.DataFrame(batch_data)
        mode = 'w' if first_batch else 'a'
        batch_df.to_csv(output_file, mode=mode, header=first_batch, index=False)

        print(f"✅ Batch {i//batch_size + 1} saved to {output_file}")

    print("✅ Summarization complete!")
