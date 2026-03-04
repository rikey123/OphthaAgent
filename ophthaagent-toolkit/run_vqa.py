import json
import os
import sys
import re
import importlib.util
from datetime import datetime
from tqdm import tqdm
import argparse

def import_module_from_path(module_name, file_path):
    """
    Imports a module from a given file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{module_name}' at '{file_path}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def parse_answer(result_message):
    """
    Parses the answer from the agent's output message.
    It robustly handles two formats:
    1. The expected format: <answer>X</answer> where X is the letter choice.
    2. A direct answer: where the content is just the letter choice (e.g., 'B').
    """
    content = None
    if hasattr(result_message, 'content'):
        content = result_message.content
    elif isinstance(result_message, str):
        content = result_message

    if content:
        # First, try to find the answer within <answer> tags
        match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE | re.DOTALL)
        if match:
            # Extract and return the answer from the tags
            return match.group(1).strip()
        
        # If no tags are found, check if the content itself is a valid answer.
        # A valid answer is a single uppercase letter from A-D.
        stripped_content = content.strip()
        if len(stripped_content) == 1 and 'A' <= stripped_content.upper() <= 'D':
            return stripped_content.upper()
            
    return None

def run_vqa_test(vqa_file, results_output_path, agent_file_path):
    """
    Main function to run the VQA test.
    """
    # Path to the agent script to be tested
    # agent_file_path = '<ANON_ABS_PATH>'
    
    # Import the agent script as a module
    try:
        fundus_agent = import_module_from_path('fundus_agent', agent_file_path)
    except ImportError as e:
        print(f"Error importing agent script: {e}")
        return

    # VQA data file and image directory prefix
    # vqa_file = '<ANON_ABS_PATH>'
    image_prefix = '<ANON_ABS_PATH>'
    
    # Define output path for JSONL results and clear it initially
    # results_output_path = '<ANON_ABS_PATH>'
    with open(results_output_path, 'w') as f:
        pass  # This clears the file at the beginning of the run

    results_for_summary = []
    
    # Load the agent module from the specified file path
    spec = importlib.util.spec_from_file_location("fundus_agent", agent_file_path)
    fundus_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fundus_agent)
    app = fundus_agent.app

    # Load VQA data
    with open(vqa_file, 'r') as f:
        vqa_data = json.load(f)

    # Prepare to save results
    # results_output_path = '<ANON_ABS_PATH>'
    if os.path.exists(results_output_path):
        os.remove(results_output_path)

    correct_predictions = 0
    total_predictions = 0
    error_count = 0

    # Process each entry with a progress bar
    for data in tqdm(vqa_data, desc="Processing VQA entries"):
        original_stdout = sys.stdout
        log_dir = None
        try:
            # Construct the full image path
            image_path = os.path.join(image_prefix, data['image'])
            
            # Create a unique output directory for this run
            base_name = os.path.basename(image_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            date_str = datetime.now().strftime("%Y%m%d")
            # The directory for logs, without the data['id']
            output_dir_name = f"{file_name_without_ext}_{date_str}"
            log_dir = os.path.join("<ANON_ABS_PATH>", output_dir_name)
            os.makedirs(log_dir, exist_ok=True)

            # Set up logging with unique filenames to avoid overwriting
            log_file_path = os.path.join(log_dir, f"reasoning_log_{data['id']}.txt")
            run_log_path = os.path.join(log_dir, f"run_log_{data['id']}.txt")

            # Redirect stdout to a log file for detailed analysis
            with open(run_log_path, 'w') as f_log:
                sys.stdout = f_log

                # Extract the question and the correct answer
                user_query = data['conversations'][0]['value']
                correct_answer = data['conversations'][1]['value']
                
                # Ensure the image file exists before proceeding
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found at {image_path}. Skipping.")
                    continue
                
                initial_state = {
                    "input": user_query,
                    "image_path": image_path,
                    "log_file_path": log_file_path, # Add log_file_path to state
                }
                
                # Get the agent app and thread config
                app = fundus_agent.app
                # Create a unique thread_id for each VQA entry to ensure isolation
                thread_config = {"configurable": {"thread_id": f"vqa_{data['id']}"}}
                
                # Invoke the agent
                final_state = app.invoke(initial_state, config=thread_config)
                
                # Parse the answer from the final state
                predicted_answer = parse_answer(final_state.get("response"))

            # Restore stdout
            sys.stdout = original_stdout

            # Store results
            result_entry = {
                "id": data["id"],
                "image": data["image"],
                "question": user_query,
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "is_correct": predicted_answer == correct_answer if predicted_answer else False
            }
            results_for_summary.append(result_entry)
            
            # Append result to the output JSONL file immediately
            with open(results_output_path, 'a') as f_out:
                f_out.write(json.dumps(result_entry) + '\n')

        except Exception as e:
            # Restore stdout in case of an error
            sys.stdout = original_stdout
            print(f"An error occurred while processing id {data.get('id', 'N/A')}: {e}")
            # Optionally log the error to a separate error log file
            if log_dir:
                with open(os.path.join(log_dir, 'error_log.txt'), 'w') as f_err:
                    f_err.write(f"Error processing id {data.get('id', 'N/A')}:\n{e}\n")
                    import traceback
                    traceback.print_exc(file=f_err)
            continue # Continue to the next item

    # Final summary calculation
    if not results_for_summary:
        print("No results were processed to generate a summary.")
        return

    correct_predictions = sum(1 for r in results_for_summary if r['is_correct'])
    total_count = len(results_for_summary)
    accuracy = (correct_predictions / total_count) * 100 if total_count > 0 else 0
    
    print("\n--- VQA Test Summary ---")
    print(f"Total questions processed: {total_count}")
    print(f"Correct answers: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")    
    print(f"Detailed results saved to {results_output_path}")

    print(f"Final VQA Test Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions}) with {error_count} errors.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run VQA test on a specified agent and dataset.")
    parser.add_argument("--vqa_file", type=str, required=True, help="Path to the VQA JSON file.")
    parser.add_argument("--results_output_path", type=str, required=True, help="Path to save the results JSONL file.")
    parser.add_argument("--agent_file_path", type=str, required=True, help="Path to the agent Python file.")
    args = parser.parse_args()

    run_vqa_test(args.vqa_file, args.results_output_path, args.agent_file_path)