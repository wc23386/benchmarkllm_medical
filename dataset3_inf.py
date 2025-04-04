# Evaluate the model
from datasets import load_dataset


# Load the dataset
dataset = load_dataset('json', data_files={'test': './llm_distillation/matched_llama_dataset.jsonl'})
print(dataset)

# Inference with batching and progress bar
from tqdm.auto import tqdm
import numpy as np

def batch_generate_responses(dataset, batch_size=8):
    """
    Generate responses for the dataset in batches with progress bar
    """
    all_responses = []
    
    # Create batches
    num_examples = len(dataset['test'])
    num_batches = int(np.ceil(num_examples / batch_size))
    
    # Progress bar
    pbar = tqdm(total=num_examples, desc="Generating responses")
    
    for i in range(0, num_examples, batch_size):
        batch_questions = dataset['test']['question'][i:i + batch_size]
        batch_responses = []
        
        # Process each question in the batch
        for question in batch_questions:
            messages = [
                {"role": "system", "content": "You are a medical knowledge assistant trained to provide information and guidance on various health-related topics."},
                {"role": "user", "content": question}
            ]
            
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                batch_responses.append(response.split("assistant")[-1].strip())
        
        all_responses.extend(batch_responses)
        pbar.update(len(batch_questions))
    
    pbar.close()
    return all_responses

# Function to evaluate model performance
def evaluate_model(dataset, batch_size=8):
    """
    Evaluate model performance on the test set
    """
    print("Starting model evaluation...")
    
    # Generate predictions
    predictions = batch_generate_responses(dataset, batch_size)
    
    # Calculate metrics (you can add more metrics as needed)
    results = {
        "total_examples": len(predictions),
        "average_length": np.mean([len(p.split()) for p in predictions]),
    }
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")
    
    return results, predictions

# Save predictions to file
def save_predictions(predictions, output_file='predictions.jsonl'):
    """
    Save predictions to a JSONL file
    """
    with open(output_file, 'w') as f:
        for idx, pred in enumerate(predictions):
            json.dump({
                'question': dataset['test']['question'][idx],
                'prediction': pred,
                'ground_truth': dataset['test']['llama_response'][idx]
            }, f)
            f.write('\n')

# Run evaluation
print("Running evaluation on test set...")
results, predictions = evaluate_model(dataset, batch_size=8)

# Save predictions
save_predictions(predictions, 'test_predictions.jsonl')


