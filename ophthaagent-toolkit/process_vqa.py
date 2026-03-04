import json
import os
from fundus_decision2_v11_03 import app, decision_maker, thread_config

def process_vqa_dataset(input_jsonl_path, output_jsonl_path):
    """
    Processes a VQA dataset in JSONL format, runs the OphthaAgent on each entry,
    and saves the results to a new JSONL file.
    """
    with open(input_jsonl_path, 'r') as infile, open(output_jsonl_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            
            # Construct the full image path
            image_path = os.path.join("<ANON_ABS_PATH>", data["image"])
            
            # Extract the question
            question = data["conversations"][0]["value"]
            
            # Check if the image file exists
            if not os.path.exists(image_path):
                print(f"Image not found, skipping: {image_path}")
                continue

            # Set up the initial state for the agent
            initial_state = {
                "input": question,
                "image_path": image_path,
            }

            try:
                # Run the agent
                final_state = app.invoke(initial_state, config=thread_config)
                
                # Get the final result from the decision_maker
                result = decision_maker(final_state)

                # Prepare the output data
                output_data = {
                    "image_path": data["image"],
                    "question": question,
                    "result": result.content  # Assuming result is a message object with content
                }
                
                # Write the result to the output file
                outfile.write(json.dumps(output_data) + '\n')
                print(f"Processed: {data['id']}")

            except Exception as e:
                print(f"Error processing entry {data.get('id', 'N/A')}: {e}")


if __name__ == "__main__":
    input_file = "<ANON_ABS_PATH>"
    output_file = "<ANON_ABS_PATH>"
    process_vqa_dataset(input_file, output_file)
    print(f"Processing complete. Results saved to {output_file}")
