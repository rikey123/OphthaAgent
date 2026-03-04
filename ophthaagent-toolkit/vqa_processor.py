import re
from tqdm import tqdm

# --- Configuration ---
import json
import os
import requests
import base64

# --- API Configuration ---
VQA_API_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "<ANON_VALUE>"
API_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_image_mime_type(image_path):
    """Determine the MIME type of an image based on its extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.jpg' or ext == '.jpeg':
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    else:
        return 'application/octet-stream'

def call_vqa_api(image_path, question):
    """
    Calls the SiliconFlow VQA API with the specified image and question.
    """
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    
    mime_type = get_image_mime_type(image_path)
    image_url = f"data:{mime_type};base64,{base64_image}"

    system_prompt = "You are an expert ophthalmologist. Your task is to analyze the provided fundus image and identify all matching diagnoses or descriptions from the list of options. You MUST provide an answer. If you are uncertain, choose the single most likely option. There might be one or multiple correct answers. List all correct options you find, separated by commas. Your final answer must be enclosed in <answer></answer> tags. For example: <answer>A,C</answer> or <answer>B</answer>."

    payload = {
        "model": "Qwen/Qwen2.5-VL-32B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 1.0
    }

    # 3. Make the API call
    print(f"Sending request to API. Payload size: {len(json.dumps(payload).encode('utf-8')) / 1024:.2f} KB")
    try:
        response = requests.post(VQA_API_ENDPOINT, headers=API_HEADERS, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        if e.response:
            print(f"API Response: {e.response.text}")
        return None

def process_vqa_data(input_json_path, output_json_path, image_base_dir):
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {input_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {input_json_path}")
        print(f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno} (char {e.pos})")
        return

    # Print the input file path for debugging
    print(f"\n--- Processing input file: {input_json_path} ---\n")

    results = []
    for item in tqdm(data, desc="Processing VQA data"):
        try:
            image_path = os.path.join(image_base_dir, item['image'])
            
            # Find the question and answer in conversations
            question = None
            correct_answer = None
            for conv in item['conversations']:
                if conv['from'] == 'human':
                    question = conv['value']
                elif conv['from'] == 'gpt':
                    correct_answer = conv['value']

            if not os.path.exists(image_path):
                print(f"Warning: Image file not found at {image_path}, skipping item ID {item['id']}")
                continue
            
            if question:
                api_response = call_vqa_api(image_path, question)
                
                model_answer_text = None
                if api_response and 'choices' in api_response and api_response['choices']:
                    message = api_response['choices'][0].get('message', {})
                    if message:
                        model_answer_text = message.get('content')

                # Add a print statement to debug the raw model output
                print(f"\n--- Raw Model Output for item ID {item['id']} ---\n{model_answer_text}\n-------------------------------------\n")

                predicted_answer = None
                if model_answer_text:
                    # Use re.DOTALL to handle cases where the answer might span newlines
                    match = re.search(r'<answer>(.*?)</answer>', model_answer_text, re.DOTALL)
                    if match:
                        predicted_answer = match.group(1).strip()

                        # Normalize answers for comparison. This handles order differences (e.g., "A,C" vs "C,A")
                        # and whitespace differences.
                        predicted_labels = sorted([label.strip() for label in predicted_answer.split(',')])

                        if correct_answer:
                            # Ensure correct_answer is a string before splitting
                            correct_labels = sorted([label.strip() for label in str(correct_answer).split(',')])
                        else:
                            correct_labels = []

                        is_correct = (predicted_labels == correct_labels)

                        result_item = {
                            "id": item['id'],
                            "image_path": item['image'],
                            "question": question,
                            "correct_answer": correct_answer,
                            "predicted_answer": predicted_answer,
                            "is_correct": is_correct
                        }
                        results.append(result_item)

                        # Incrementally save the results to the file
                        with open(output_json_path, 'w') as f:
                            json.dump(results, f, indent=4)
        except Exception as e:
            print(f"\n--- An error occurred while processing item ID {item.get('id', 'N/A')} ---")
            print(f"Error: {e}")
            print("----------------------------------------------------\n")
            continue

    if results:
        print(f"Successfully processed {len(results)} items and saved results to {output_json_path}")
    else:
        print("Processing complete, but no results were generated.")


if __name__ == '__main__':
    JSON_FILE = "<ANON_ABS_PATH>"
    IMAGE_BASE_DIR = "<ANON_ABS_PATH>"
    OUTPUT_JSON_FILE = "<ANON_ABS_PATH>"
    
    process_vqa_data(JSON_FILE, OUTPUT_JSON_FILE, IMAGE_BASE_DIR)