import torch
import pandas as pd
import json

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device:{device}")

# Load json file into dataframe
def json_to_df(json_file_path):
    sample_id_list = []
    question_list = []
    response_list =[]
    llm_model = json_file_path.split('/')[-1].split('_')[1]

    with open(json_file_path, 'rb') as file:
        for line in file:
            json_object = json.loads(line)
            # print(json_object, type(json_object)) # <class 'dict'> reminder: dict_values() object is not subscriptable
            sample_id_list.append(list(json_object.values())[0])
            question_list.append(list(json_object.values())[1])
            response_list.append(list(json_object.values())[2])

    df = pd.DataFrame({'question': question_list, f'{llm_model}_response': response_list})
    return df
gpt_df = json_to_df('/Users/wcchang/Documents/llm_distillation/dataset3_gpt_qa_pairs.jsonl')
llama_df = json_to_df('/Users/wcchang/Documents/llm_distillation/matched_llama_dataset.jsonl')
# print(gpt_df.columns, gpt_df.shape, '\n', llama_df.columns, llama_df.shape)
# Merge two dataframes, ['question', 'gpt_response', 'llama_response']
combined_df = pd.merge(gpt_df, llama_df, on='question', how='left')
# print(combined_df.columns, combined_df.shape)

# Split into train and test set
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
# print(train_df.shape, test_df.shape)

# Only focus on gpt train set
gpt_inf_df = train_df[['question', 'gpt_response']]
# print(gpt_inf_df.shape)

# Fine tuning the model
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)

# First load the tokenizer and set padding token
model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Then load the model with the padding token configuration and use_cache=False
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    pad_token_id=tokenizer.eos_token_id,
    use_cache=False
).to(device)

# Configure LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Prepare the model for training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=64,  # rank
    lora_alpha=64,  # scaling factor
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj", 
        "up_proj", 
        "down_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Print trainable parameters info
model.print_trainable_parameters()

# Configure training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='steps',
    eval_steps=1000,
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=1000,
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=2,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    gradient_accumulation_steps=8,
    max_grad_norm=0.3,
    gradient_checkpointing=True,
)

# Split gpt_inf_df into train and validation sets
train_df, eval_df = train_test_split(gpt_inf_df, test_size=0.1, random_state=42)

# Format chat template (reusing your existing function)
def format_chat_template(example):
    # Input prompt is just the question
    input_messages = [
        {"role": "system", "content": "You are a medical knowledge assistant trained to answer multiple choice questions. Please only provide the unique correct option's letter (e.g. A, B, C, etc) with no additional text or punctuation."},
        {"role": "user", "content": example['question']}
    ]
    
    # Target response is the GPT response
    target_messages = [
        {"role": "assistant", "content": example['gpt_response']}
    ]
    
    # Get the prompt and target text
    input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)
    target_text = tokenizer.apply_chat_template(target_messages, tokenize=False)
    
    # Tokenize input and target
    model_inputs = tokenizer(input_text, truncation=True, padding='max_length', max_length=512)
    labels = tokenizer(target_text, truncation=True, padding='max_length', max_length=128)
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs
from datasets import Dataset, load_dataset
# Convert DataFrames to Datasets and apply formatting
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Process datasets
train_dataset = train_dataset.map(
    format_chat_template,
    remove_columns=train_dataset.column_names,
    desc="Processing train dataset"
)

eval_dataset = eval_dataset.map(
    format_chat_template,
    remove_columns=eval_dataset.column_names,
    desc="Processing eval dataset"
)

# Verify dataset format
print("Train dataset features:", train_dataset.features)
print("Sample sequence length:", len(train_dataset[0]["input_ids"]))

# Initialize DataCollatorForLanguageModeling with explicit padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8  # Optional: for efficient tensor operations
)

# Initialize Trainer (without the incorrect label_names parameter)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./models/llama-3.2-3b-inf-phase2')
tokenizer.save_pretrained('./models/llama-3.2-3b-inf-phase2')