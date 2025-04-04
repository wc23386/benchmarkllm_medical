from comet_ml import Experiment
from datasets import load_dataset
import requests
import json
from tqdm import tqdm
import warnings
import time
import os
from opik import track
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Ensure this is called before accessing the API key
warnings.filterwarnings("ignore", category=Warning)

# Initialize Comet experiment
experiment = Experiment(api_key=os.getenv("OPIK_API_KEY"), project_name='llama_dataset3')

# (alternative) Set up Opik environment variables
# os.environ["OPIK_API_KEY"] = "OPIK_API_KEY"  # Replace with your API key
# os.environ["OPIK_WORKSPACE"] = "YOUR_WORKSPACE"  # Replace with your workspace
print("Using project name:", experiment.project_name)

@track(project_name="llama_dataset3")
def get_llama_response(question, max_retries=3):
    """Get response from locally running Llama model via Ollama"""
    messages = [
        {
            "role": "system",
            "content": "You are a medical knowledge assistant trained to answer multiple choice questions. Please only provide the correct option's letter (e.g. A, B, C, etc) without any additional text or punctuation."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = requests.post('http://localhost:11434/api/generate',
                                   json={
                                       'model': 'llama3.2:3b',
                                       'prompt': json.dumps(messages),
                                       'stream': False,
                                       'options': {
                                           'mps': True  # Use Apple Silicon GPU
                                       }
                                   },
                                   timeout=30)
            response.raise_for_status()
            # Strip any trailing punctuation (like periods)
            return response.json()['response'].strip().rstrip('.')
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to get response after {max_retries} attempts: {str(e)}")
            time.sleep(2 * (attempt + 1))

@track(project_name="llama_dataset3")
def calculate_accuracy(predictions, references):
    """Calculate accuracy for a batch"""
    correct = sum(p == r[0] for p, r in zip(predictions, references))
    return correct / len(references) if references else 0

@track(project_name='llama_dataset3')
def main():
    # Load medical dataset
    try:
        # Login using e.g. `huggingface-cli login` to access this dataset
        medical_dataset = load_dataset("OpenMedical/medical-verifiable-dedup")
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        return

    total_entries = len(medical_dataset['train'])
    # total_entries = 10
    batch_size = 1000
    
    # Initialize lists for storing responses
    llama_responses = []
    answer_list = []
    results_to_save = []
    error_count = 0
    
    print(f"\nProcessing {total_entries} entries in batches of {batch_size}:")
    print("=" * 80)
    
    output_file = "dataset3_llama3_qa_pairs.jsonl"
    # Process all examples with progress bar
    for i, example in enumerate(tqdm(medical_dataset['train'], total=total_entries)):
        if error_count >= 10:
            print("\nToo many errors encountered. Stopping processing.")
            break
            
        # Filter out entries without a correct option
        if not example.get('correct_option'):
            continue
            
        try:
            question = example['question']
            ground_truth = example['correct_option']
            
            prediction = get_llama_response(question=question)
            llama_responses.append(prediction)
            answer_list.append([ground_truth])
            
            # Store result in JSONL format
            result = {
                "custom_id": str(i),
                "Question": question,
                "Prediction": prediction,
                "Reference": ground_truth
            }
            results_to_save.append(result)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
            
            # Print first 5 Q&As
            if i < 5:
                print(f"\nQ{i+1}: {question}")
                print(f"A{i+1}: {prediction}")
                print(f"Ground Truth: {ground_truth}\n")
            
        except Exception as e:
            error_count += 1
            print(f"\nError processing entry {i}: {str(e)}")
            continue

        
        # Calculate metrics every batch_size entries
        if (i + 1) % batch_size == 0 and llama_responses:
            accuracy = calculate_accuracy(llama_responses, answer_list)
            experiment.log_metric("accuracy", accuracy)
            
            print(f"\nMetrics after {i + 1} entries:")
            print("Accuracy:", accuracy)
            print(f"Errors encountered: {error_count}")
            print("-" * 80)
    
    # Calculate final accuracy
    if llama_responses:
        print("\nFinal Metrics:")
        final_accuracy = calculate_accuracy(llama_responses, answer_list)
        print("Final Accuracy:", final_accuracy)
        # Log final accuracy to Comet
        experiment.log_metric("final_accuracy", final_accuracy)

    # Save results to JSONL file
    output_file = "dataset3_llama3_qa_pairs.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results_to_save:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()