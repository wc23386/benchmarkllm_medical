import pandas as pd
import json

import pandas as pd
import json
import numpy as np

# Read the dataset
df = pd.read_json('dataset3_llama_qa_pairs.jsonl', lines=True)

# Function to check if response has multiple answers
def has_multiple_answers(response):
    if pd.isna(response):
        return True
    return ',' in str(response) or '[' in str(response) or ']' in str(response)

# Filter out rows where llama_response is null or has multiple answers
df_filtered = df[~df['Prediction'].apply(has_multiple_answers)]

# Convert to records and save
filtered_data = df_filtered.to_dict('records')

# Save filtered data
with open('filtered_dataset.jsonl', 'w') as f:
    for item in filtered_data:
        json.dump(item, f)
        f.write('\n')

print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size: {len(df_filtered)}")
print(f"Removed {len(df) - len(df_filtered)} rows")

# Read both datasets
gpt_df = pd.read_json('dataset3_gpt_qa_pairs.jsonl', lines=True)
llama_df = pd.read_json('dataset3_llama_qa_pairs.jsonl', lines=True)

# Print column names to debug
print("GPT DataFrame columns:", gpt_df.columns.tolist())
print("LLaMA DataFrame columns:", llama_df.columns.tolist())

# Print first few rows of each DataFrame
print("\nFirst few rows of GPT DataFrame:")
print(gpt_df.head())
print("\nFirst few rows of LLaMA DataFrame:")
print(llama_df.head())

# Create a set of questions from GPT dataset for faster lookup
gpt_questions = set(gpt_df['Question'].tolist())

# Filter llama_df to keep only rows where question matches in gpt_df
matched_df = llama_df[llama_df['Question'].isin(gpt_questions)]

# Convert to records and save
matched_data = matched_df.to_dict('records')

# Save matched data
with open('matched_llama_dataset.jsonl', 'w') as f:
    for item in matched_data:
        json.dump(item, f)
        f.write('\n')

print(f"GPT dataset size: {len(gpt_df)}")
print(f"Original LLaMA dataset size: {len(llama_df)}")
print(f"Matched LLaMA dataset size: {len(matched_df)}")
print(f"Number of matching questions: {len(matched_df)}") 