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
import evaluate

# Load environment variables
load_dotenv()  # Ensure this is called before accessing the API key
warnings.filterwarnings("ignore", category=Warning)

# Initialize Comet experiment
experiment = Experiment(api_key=os.getenv("OPIK_API_KEY"), project_name='llama_dataset2')
print("Using project name:", experiment.project_name)

# (alternative) Set up Opik environment variables
# os.environ["OPIK_API_KEY"] = "OPIK_API_KEYC"  # Replace with your API key
# os.environ["OPIK_WORKSPACE"] = "YOUR_WORKSPACE"  # Replace with your workspace

@track(project_name="llama_dataset2")
def get_llama_response(question, max_retries=3):
    """Get response from locally running Llama model via Ollama"""
    messages = [
        {
            "role": "system",
            "content": "You are a concise, highly knowledgeable medical assistant. Provide medically accurate and direct answers to health-related questions without any disclaimers or explanations."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = requests.post('http://localhost:11434/api/chat',
                                   json={
                                       'model': 'llama3.2:3b',
                                       'messages': messages,
                                       'stream': False,
                                       'options': {
                                           'mps': True  # Use Apple Silicon GPU
                                       }
                                   },
                                   timeout=30)
            response.raise_for_status()
            return response.json()['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to get response after {max_retries} attempts: {str(e)}")
            time.sleep(2 * (attempt + 1))

@track(project_name="llama_dataset2")
def calculate_metrics(predictions, references):
    """Calculate BLEU and ROUGE scores for a batch"""
    bleu_eval = evaluate.load("bleu")
    bleu_results = bleu_eval.compute(predictions=predictions, references=references)
    
    rouge_eval = evaluate.load("rouge")
    rouge_results = rouge_eval.compute(predictions=predictions, references=[r[0] for r in references])
    
    return {**bleu_results, **rouge_results}

@track(project_name="llama_dataset2")
def main():
    # Load medical dataset
    try:
        medical_dataset = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem")
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        return

    # total_entries = min(len(medical_dataset['train']), 10)  # Limit to first 10 entries
    total_entries = len(medical_dataset['train'])
    batch_size = 1000
    
    # Initialize lists for storing responses
    llama_responses = []
    answer_list = []
    error_count = 0
    
    print(f"\nProcessing {total_entries} entries in batches of {batch_size}:")
    print("=" * 80)
    
    # Process all examples with progress bar
    for i, example in enumerate(tqdm(medical_dataset['train'], total=total_entries)):
        if i >= total_entries:
            break  # Stop processing after reaching total_entries
        
        if error_count >= 10:
            print("\nToo many errors encountered. Stopping processing.")
            break
            
        try:
            question = example['Open-ended Verifiable Question']
            ground_truth = example['Ground-True Answer']
            
            prediction = get_llama_response(question=question)
            llama_responses.append(prediction)
            answer_list.append([ground_truth])
            
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
            metrics = calculate_metrics(llama_responses, answer_list)
            
            # Log metrics to OPIK
            experiment.log_metrics(metrics)  # Log metrics to OPIK
            
            print(f"\nMetrics after {i + 1} entries:")
            print("BLEU Score:", metrics['bleu'])
            print("ROUGE Scores:")
            print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
            print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
            print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
            print(f"Errors encountered: {error_count}")
            print("-" * 80)
    
    # Calculate final metrics
    if llama_responses:
        print("\nFinal Metrics:")
        final_metrics = calculate_metrics(llama_responses, answer_list)
        final_metrics['total_errors'] = error_count
        print(json.dumps(final_metrics, indent=2))

        # Log final metrics to Comet
        experiment.log_metrics(final_metrics)  # Log final metrics to OPIK

    # Ensure proper flushing of logs
    experiment.flush()  # Flush the logs to ensure they are sent to OPIK

if __name__ == "__main__":
    main()