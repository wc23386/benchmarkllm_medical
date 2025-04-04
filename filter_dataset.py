import pandas as pd
import json
import numpy as np

# Read the dataset
df = pd.read_json('dataset3_4omini_qa_pairs.jsonl', lines=True)

# Function to check if response has multiple answers
def has_multiple_answers(response):
    if pd.isna(response):
        return True
    return ',' in str(response) or '[' in str(response) or ']' in str(response)

# Filter out rows where llama_response is null or has multiple answers
df_filtered = df[~df['llama_response'].apply(has_multiple_answers)]

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