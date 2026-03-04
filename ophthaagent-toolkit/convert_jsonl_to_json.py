import json
import os

def convert_jsonl_to_json(jsonl_file_path):
    """
    Converts a .jsonl file to a .json file.

    Args:
        jsonl_file_path (str): The absolute path to the .jsonl file.
    """
    json_file_path = os.path.splitext(jsonl_file_path)[0] + ".json"
    
    data = []
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully converted {jsonl_file_path} to {json_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {jsonl_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {jsonl_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    files_to_convert = [
        "<ANON_ABS_PATH>",
        "<ANON_ABS_PATH>",
        "<ANON_ABS_PATH>",
        "<ANON_ABS_PATH>",
        "<ANON_ABS_PATH>"
    ]
    
    for file_path in files_to_convert:
        convert_jsonl_to_json(file_path)
