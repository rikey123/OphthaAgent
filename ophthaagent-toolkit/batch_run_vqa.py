
import subprocess
import os

# Define the base path for the project to ensure all paths are constructed correctly
base_path = '<ANON_ABS_PATH>'
run_script_path = os.path.join(base_path, 'run_vqa.py')

# A list of dictionaries, where each dictionary represents a single VQA test run configuration.
configurations = [
    # {
    #     "vqa_file": os.path.join(base_path, 'benchmark/filtered_AMD_RDMid/filtered_rfmid.json'),
    #     "agent_file_path": os.path.join(base_path, 'fundus_decision2_v11_18_amd.py'),
    #     "dataset_name": "AMD_RDMid"
    # },
    {
        "vqa_file": os.path.join(base_path, 'benchmark/filtered_GlAUCOMA_ACRIMA/filtered_acrima.json'),
        "agent_file_path": os.path.join(base_path, 'fundus_decision2_parallel_v12_02.py'),
        "dataset_name": "GlAUCOMA_ACRIMA"
    },
    # {
    #     "vqa_file": os.path.join(base_path, 'benchmark/filtered_DR_E-ophta/filtered_by_id_E-ophta.json'),
    #     "agent_file_path": os.path.join(base_path, 'fundus_decision2_v11_18_dr.py'),
    #     "dataset_name": "DR_E-ophta"
    # },
    # {
    #     "vqa_file": os.path.join(base_path, 'benchmark/filtered_DR_grading/filtered_by_id_APTOS.json'),
    #     "agent_file_path": os.path.join(base_path, 'fundus_decision2_v11_18_lession.py'),
    #     "dataset_name": "DR_grading_APTOS"
    # }
]

# Iterate over each configuration and execute the VQA test script
for config in configurations:
    vqa_file = config["vqa_file"]
    agent_file_path = config["agent_file_path"]
    dataset_name = config["dataset_name"]
    
    # Dynamically create the output path for the results based on the dataset name
    results_output_path = os.path.join(base_path, f'vqa_results_{dataset_name}.jsonl')
    
    print(f"--- Running VQA test for dataset: {dataset_name} ---")
    print(f"  VQA File: {vqa_file}")
    print(f"  Agent File: {agent_file_path}")
    print(f"  Results will be saved to: {results_output_path}")
    
    # Construct the command-line arguments for the run_vqa.py script
    command = [
        'python',
        run_script_path,
        '--vqa_file', vqa_file,
        '--results_output_path', results_output_path,
        '--agent_file_path', agent_file_path
    ]
    
    try:
        # Execute the command and wait for it to complete
        process = subprocess.run(command, check=True)
        print(f"--- Successfully completed test for dataset: {dataset_name} ---\n")
    except subprocess.CalledProcessError as e:
        # If the script returns a non-zero exit code, it indicates an error
        print(f"--- An error occurred while running test for dataset: {dataset_name} ---")
        print(f"--- Test failed for dataset: {dataset_name} ---\n")

print("All VQA tests have been executed.")
