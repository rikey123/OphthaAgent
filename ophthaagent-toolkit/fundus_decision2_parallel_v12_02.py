"""
Ophthalmic Diagnosis Agent based on LangGraph.
"""

import json
from config_fundus import Config
import concurrent.futures
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from typing import Tuple
import operator
import base64
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from agents.fundus_tools_agent.tools_interface import enhance_fundus_image
from agents.fundus_tools_agent.tools_interface import fundus_lesion_segmentation
from agents.fundus_tools_agent.tools_interface import crop_by_fit
from agents.fundus_tools_agent.tools_interface import enhance_by_fit
from agents.fundus_tools_agent.tools_interface import quality_assess_by_fit
from agents.fundus_tools_agent.tools_interface import fov_od_localization_by_fit
from agents.fundus_tools_agent.tools_interface import vessel_segment_by_fit

from agents.fundus_tools_agent.tools.tools_interface1 import (
    run_automorph_m0_m1,
    run_automorph_m2,
    run_quantitative_measurement
)

from agents.fundus_tools_agent.tool_interface3 import(
    segment_by_ddcs,
    DR_Grading
)
from agents.fundus_tools_agent.tool_interface4 import(
    AMD_predict_fundus_by_deepseenet,
    Lesion_predict_OCT_by_opticnet,
    segment_by_AutoMorphalyzer,
    dme_risk_assessment,
)

from rag_query import rag_query
from tool_call_test import autoMorphProcess
from langgraph.checkpoint.memory import MemorySaver
import atexit

# Global list to store the log of each step
execution_log = []

def save_log():
    """Saves the execution log to a JSON file."""
    # The log will be saved in the same directory where the script is run
    with open('execution_log.json', 'w', encoding='utf-8') as f:
        json.dump(execution_log, f, indent=4, ensure_ascii=False)

# Register the save_log function to be called on script exit
atexit.register(save_log)

import numpy as np


def invoke_llm_with_json_retry(chain, input_data, max_retries=3, pydantic_model=None):
    """
    Invoke LLM chain with automatic retry on JSON parsing errors.

    Args:
        chain: The LangChain chain to invoke
        input_data: Input dictionary for the chain
        max_retries: Maximum number of retry attempts
        pydantic_model: Optional Pydantic model for validation

    Returns:
        Parsed result from the chain

    Raises:
        Exception: If all retries fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            result = chain.invoke(input_data)

            # If we got a result, validate it's proper JSON/dict
            if isinstance(result, dict):
                # Additional validation for required fields if pydantic model provided
                if pydantic_model:
                    try:
                        # Validate against pydantic model with lenient checking
                        # Only validate the structure exists, not strict field matching
                        pydantic_model.model_validate(result)
                    except Exception as validation_error:
                        # Check if this is a minor validation error we can ignore
                        error_str = str(validation_error)

                        # If it's about tool_call structure, and we have thought field, it's probably okay
                        if 'tool_call' in error_str and 'thought' in result:
                            print(f"⚠️  Attempt {attempt + 1}/{max_retries}: Minor validation warning (ignoring): {validation_error}")
                            # Return the result anyway since it has the essential fields
                            return result

                        print(f"⚠️  Attempt {attempt + 1}/{max_retries}: Validation error - {validation_error}")
                        last_error = validation_error

                        # Add more specific hint about the validation error
                        if attempt < max_retries - 1:
                            input_data['json_error_hint'] = f"IMPORTANT: Your previous response had validation errors: {validation_error}. Please ensure all required fields are present with correct structure."
                        continue

                return result 
            else:
                print(f"⚠️  Attempt {attempt + 1}/{max_retries}: Result is not a dict, got {type(result)}")
                last_error = ValueError(f"Expected dict, got {type(result)}")
                continue

        except json.JSONDecodeError as e:
            print(f"⚠️  Attempt {attempt + 1}/{max_retries}: JSON decode error - {e}")
            print(f"    Error location: line {e.lineno}, column {e.colno}")
            last_error = e

            # Add a hint to the input for next attempt
            if attempt < max_retries - 1:
                input_data['json_error_hint'] = "IMPORTANT: Your previous response had invalid JSON format. Please ensure your response is valid JSON with proper syntax (correct commas, quotes, brackets)."

        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1}/{max_retries}: Unexpected error - {type(e).__name__}: {e}")
            last_error = e

            if attempt < max_retries - 1:
                input_data['error_hint'] = f"IMPORTANT: Your previous response caused an error: {type(e).__name__}. Please fix the format."

    # All retries failed
    print(f"❌ All {max_retries} attempts failed. Last error: {last_error}")
    raise last_error


load_dotenv()

# Load application configuration settings
config = Config()

# Initialize memory for saving and restoring graph state
memory = MemorySaver()

# Specify a unique ID for the current execution thread for state management
thread_config = {"configurable": {"thread_id": "1"}}

# 明确最多工具调用次数
MAX_TOOL_CALLS = 7


class ToolCall(BaseModel):
    """Represents a single tool call in the plan."""
    tool_name: str = Field(description="The name of the tool to call.")
    tool_args: dict = Field(description="The arguments for the tool.")

class Act(BaseModel):
    """The action to take next."""
    thought: str = Field(description="A brief thought process for the action (less than 150 words). You MUST think first before each function call, and reflect extensively on the outcomes of the past steps. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully. ")
    tool_call: Optional[ToolCall] = Field(default=None, description="The tool call to execute.")
    response: Optional[str] = Field(default=None, description="The final diagnosis or answer to the user.")

    @model_validator(mode='before')
    def check_one_action(cls, values):
        """Ensure that either tool_call or response is set, but not both."""
        if isinstance(values, dict):
            action_fields = ['tool_call', 'response']
            set_actions = [field for field in action_fields if values.get(field) is not None]
            if len(set_actions) != 1:
                raise ValueError(f"Exactly one of {action_fields} must be set. Found: {set_actions}")
        return values

class OphthaAgentState(TypedDict):
    """State tracker for the ophthalmic diagnosis agent."""
    input: str
    image_path: str
    past_steps: Annotated[List[Tuple[ToolCall, Dict]], operator.add]
    observations: Annotated[List[str], operator.add]
    response: Optional[str]
    next_tool_call: Optional[ToolCall]
    next_tool_calls_batch: Optional[List[ToolCall]]  # For parallel execution
    thought: Optional[str]  # Add thought to the state


def amd_predict_wrapper(**kwargs):
    """
    Wrapper for AMD_predict_fundus_by_deepseenet to simplify its output.
    It processes the raw JSON output to extract key findings for each lesion,
    including its label and confidence score, as well as the overall AREDS simplified severity score.
    """
    raw_output = AMD_predict_fundus_by_deepseenet(**kwargs)

    if not isinstance(raw_output, dict):
        return raw_output

    # Extract simplified score, which is a key diagnostic conclusion
    simplified_score_data = raw_output.get('simplified_score', {})

    # Extract and simplify lesion details
    simplified_lesions = {}
    for lesion, details in raw_output.get('results', {}).items():
        if isinstance(details, dict):
            simplified_lesions[lesion] = {
                "label": details.get("label"),
                "confidence": details.get("confidence"),
            }
    
    # Combine into a new structure for the summarizer
    final_output = {
        "simplified_score": simplified_score_data,
        "lesions": simplified_lesions
    }
    
    return final_output


# 定义可用眼底工具列表
tools = {
    "assess_image_quality": quality_assess_by_fit,
    "preprocess_image": enhance_fundus_image,
    "localize_fov_and_od": fov_od_localization_by_fit,
    "segment_vessel":vessel_segment_by_fit,
    "detect_lesions": fundus_lesion_segmentation,
    "autoMorphProcess":segment_by_AutoMorphalyzer,
    "dme_risk_assessment":dme_risk_assessment,
    "rag_query":rag_query,
    "segment_by_ddcs":segment_by_ddcs,
    "DR_Grading":DR_Grading,
    "AMD_predict_fundus_by_deepseenet":amd_predict_wrapper,
    "Lesion_predict_OCT_by_opticnet":Lesion_predict_OCT_by_opticnet,
}

# 获得工具的描述
def get_tool_descriptions():
    """Generate a string describing the available tools with their parameters."""
    return """""assess_image_quality": Assesses the quality of a fundus image.
  tool_args:
    image_path (str): The path to the image file.
    output_path (str): The path to the output result. 

"preprocess_image": Preprocesses a fundus image to improve its quality.
  tool_args:
    image_path (str): The path to the image file.
    output_path (str): The path to the output result.

"localize_fov_and_od": locate fov and od.
  tool_args:
    image_path (str): The path to the image file.
    output_path (str): The path to the output result.

"segment_vessel": segment blood vessel.
  tool_args:
    image_path (str): The path to the image file.
    output_path (str): The path to the output result.

"detect_lesions": Detects various types of lesions in a fundus image and give quantitative metrics.
  tool_args:
    image_path (str): The path to the image file.
    output_path (str): The path to the output result.

 "autoMorphProcess": Invoke the AutoMorph tool to perform fully automated analysis of fundus images, automatically assess image quality, segment the optic cup and optic disc, and compute quantitative metrics.
    tool_args:
    image_path (str): The path to the image file.
    output_path (str): The path to the output result.
 "dme_risk_assessment": Assess the risk of diabetic macular edema (DME) in a fundus image based on the optic disc diameter and the distance to the fovea.
    tool_args:
    image_path (str): The path to the image file.
    output_path (str): The path to the output result.
 "rag_query": Invoke the RAG tool to retrieve relevant medical knowledge and diagnostic quantitative metrics from the knowledge base. 
    tool_args:
    query (str): your query question.

 "segment_by_ddcs": Performs semantic segmentation on fundus images to identify and outline anatomical structures and lesions.
    tool_args:
    image_path (str): The path to the image file.
    output_path (str): The directory where the segmentation results will be saved.

 "DR_Grading": Analyzes a fundus image to determine the severity of diabetic retinopathy and the confidence of the grading prediction.
    tool_args:
    image_path (str): The path to the image file.

 "AMD_predict_fundus_by_deepseenet": Analyzes a fundus image to predict the likelihood of Age-related Macular Degeneration (AMD). The tool provides a detailed breakdown of predictions for various AMD-related features and calculates an overall AREDS simplified severity score (0-5), which serves as a key diagnostic indicator.
    tool_args:
    image_path (str): The path to the fundus image file.

 "Lesion_predict_OCT_by_opticnet": Analyzes an Optical Coherence Tomography (OCT) image to detect and classify various retinal lesions, providing predictions for conditions such as CNV, DME, Drusen, and Normal.
    tool_args:
    image_path (str): The path to the OCT image file.
"""""

# --- Node implementations ---

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def planner(state: OphthaAgentState):
    """
    This node deterministically decides the next action.
    It supports parallel execution: preprocessing tools run first sequentially,
    then all other tools can run in parallel.
    """
    # Define preprocessing tools that must run sequentially first
    preprocessing_tools = []

    # Define all other tools that can run in parallel after preprocessing
    # parallel_tools = ["detect_lesions", "segment_by_ddcs", "AMD_predict_fundus_by_deepseenet", "DR_Grading", "autoMorphProcess"]
    # parallel_tools = ["detect_lesions","AMD_predict_fundus_by_deepseenet","DR_Grading","dme_risk_assessment","autoMorphProcess"]
    parallel_tools = ["autoMorphProcess"]
    # For testing, you can modify these lists
    # tool_sequence = ["preprocess_image", "detect_lesions", "segment_by_ddcs","AMD_predict_fundus_by_deepseenet" ,"DR_Grading","autoMorphProcess"]
    # preprocessing_tools = ["preprocess_image"]
    # parallel_tools = ["DR_Grading"]  # For single tool testing

    past_steps = state.get("past_steps", [])
    num_past_steps = len(past_steps)

    llm = config.agent_decision.llm
    observations_str = "\n".join(state.get('observations', []))

    output_dir = "<ANON_ABS_PATH>"
    image_path = state.get("image_path")
    base_name = os.path.basename(image_path)
    file_name_without_ext = os.path.splitext(base_name)[0]

    # Determine the current input image path based on past steps
    if num_past_steps > 0:
        last_tool_call, last_output = past_steps[-1]
        tool_result = last_output.get('output', {}).get('result', {})
        if isinstance(tool_result, dict) and tool_result.get('save_path'):
            input_image_path = tool_result['save_path']
        else:
            input_image_path = last_tool_call.get('tool_args', {}).get('image_path', image_path)
    else:
        input_image_path = image_path

    # Phase 1: Execute preprocessing tools sequentially
    if num_past_steps < len(preprocessing_tools):
        next_tool_name = preprocessing_tools[num_past_steps]

        # Determine output path
        if next_tool_name == "preprocess_image":
            output_path = os.path.join(output_dir, f"{file_name_without_ext}_preprocessed.jpg")
        else:
            output_path = None

        # Generate thought for this preprocessing step
        system_prompt = """You are an expert ophthalmologist AI in a diagnostic workflow.
Your original goal is to answer: {input}

Observations from past steps:
{observations}

You are in a sequence of operations and must call the next preprocessing tool. The next tool is: {next_tool_name}.
Provide a brief thought process for this step.

CRITICAL JSON FORMAT REQUIREMENTS:
- Your response MUST be valid JSON
- Use double quotes (") for strings, not single quotes (')
- Ensure all commas are properly placed
- Do not include trailing commas
- Properly escape special characters in strings

Respond in JSON format that adheres to the `Act` schema. Your response MUST contain a `thought` and a `tool_call` field. The `tool_call` field can be an empty object as it will be ignored.
STRICT OUTPUT RULES:
- Respond in JSON format that adheres to the `Act` schema. Your response MUST contain a `thought` and a `tool_call` field. The `tool_call` field can be an empty object as it will be ignored.
- If uncertain, choose the most probable option.

{json_error_hint}
{error_hint}
"""
        base64_image = encode_image(input_image_path)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", [
                {"type": "text", "text": "Here is the image for the current step. Please proceed with the tool call based on my instructions and the image provided."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ])
        ])
        chain = prompt | llm | JsonOutputParser(pydantic_object=Act)

        # Use retry mechanism for robust JSON parsing
        input_data = {
            "input": state.get("input"),
            "observations": observations_str,
            "next_tool_name": next_tool_name,
            "json_error_hint": "",  # Will be filled by retry mechanism if needed
            "error_hint": ""  # Will be filled by retry mechanism if needed
        }
        act_obj = invoke_llm_with_json_retry(
            chain,
            input_data,
            max_retries=3,
            pydantic_model=Act
        )

        # Construct the tool call dictionary
        tool_call_dict = {
            "tool_name": next_tool_name,
            "tool_args": {"image_path": input_image_path},
        }
        if output_path:
            tool_call_dict["tool_args"]["output_path"] = output_path
        return {"next_tool_call": tool_call_dict, "next_tool_calls_batch": None, "response": None, "thought": act_obj['thought']}

    # Phase 2: After preprocessing, execute all parallel tools at once
    elif num_past_steps == len(preprocessing_tools):
        # Prepare batch of parallel tool calls
        batch_tool_calls = []

        for tool_name in parallel_tools:
            # Determine output path for each tool
            if tool_name == "detect_lesions":
                output_path = os.path.join(output_dir, f"{file_name_without_ext}_lesions.jpg")
            elif tool_name == "segment_by_ddcs":
                output_path = os.path.join(output_dir, f"{file_name_without_ext}_ddcs_segmentation")
            else:
                output_path = None

            tool_call_dict = {
                "tool_name": tool_name,
                "tool_args": {"image_path": input_image_path},
            }
            if output_path:
                tool_call_dict["tool_args"]["output_path"] = output_path

            batch_tool_calls.append(tool_call_dict)

        # Generate thought for parallel execution phase
        system_prompt = """You are an expert ophthalmologist AI in a diagnostic workflow.
Your original goal is to answer: {input}

Observations from past steps:
{observations}

Preprocessing is complete. Now you will execute multiple diagnostic tools in parallel: {parallel_tools}.
Provide a brief thought process for this parallel execution phase.

CRITICAL JSON FORMAT REQUIREMENTS:
- Your response MUST be valid JSON
- Use double quotes (") for strings, not single quotes (')
- Ensure all commas are properly placed
- Do not include trailing commas
- Properly escape special characters in strings

Respond in JSON format that adheres to the `Act` schema. Your response MUST contain a `thought` and a `tool_call` field. The `tool_call` field can be an empty object as it will be ignored.

{json_error_hint}
{error_hint}
"""
        base64_image = encode_image(input_image_path)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", [
                {"type": "text", "text": "Here is the preprocessed image. Multiple diagnostic tools will now analyze it in parallel."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ])
        ])
        chain = prompt | llm | JsonOutputParser(pydantic_object=Act)

        # Use retry mechanism for robust JSON parsing
        input_data = {
            "input": state.get("input"),
            "observations": observations_str,
            "parallel_tools": ", ".join(parallel_tools),
            "json_error_hint": "",  # Will be filled by retry mechanism if needed
            "error_hint": ""  # Will be filled by retry mechanism if needed
        }
        act_obj = invoke_llm_with_json_retry(
            chain,
            input_data,
            max_retries=3,
            pydantic_model=Act
        )

        return {
            "next_tool_call": None,
            "next_tool_calls_batch": batch_tool_calls,
            "response": None,
            "thought": act_obj['thought']
        }

    # Phase 3: Generate final response after all tools complete
    else:
        system_prompt = """You are an expert ophthalmologist AI.
    Your original goal was to answer: {input}

    You have completed all the planned tool calls and have the following observations:
    {observations}
    Based on all the key findings of the all above tool calls gathered, provide your final comprehensive diagnosis.
    Be extremely careful!!! Some questions may have multiple correct options — they are multiple-choice questions.
    Please comprehensively consider all the key evidence and clues discovered by the tools mentioned above, without omitting any. Any conflicts should be discussed in conjunction with the confidence level.

    CRITICAL JSON FORMAT REQUIREMENTS:
    - Your response MUST be valid JSON
    - Use double quotes (") for strings, not single quotes (')
    - Ensure all commas are properly placed
    - Do not include trailing commas
    - Properly escape special characters in strings

    Respond in JSON format that adheres to the `Act` schema. Your JSON output MUST contain a `thought` field and a `response` field. The `tool_call` field must not be present. The thought process must be based on the key findings from all the above observations, and a conclusion must be reached after step-by-step thinking.

    STRICT OUTPUT RULES:
    - Respond in JSON format that adheres to the `Act` schema. Your JSON output MUST contain a `thought` field and a `response` field. The `tool_call` field must not be present. The thought process must be based on the key findings from all the above observations, and a conclusion must be reached after step-by-step thinking.
    - You MUST place the final answer tagged<answer>...</answer>
    - Do NOT include chain-of-thought, explanations, or rationale in <answer>...</answer>.
    - You are doing the VQA task. The answer must be options in user questions(e.g. <answer>B</answer>).
    - If uncertain, choose the most probable option.
    - Be extremely careful!!! Some questions may have multiple correct options — they are multiple-choice questions.

    {json_error_hint}
    {error_hint}
    """
        image_path = state.get("image_path")
        base64_image = encode_image(image_path)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", [
                {"type": "text", "text": "Here is the image for the final diagnosis. Please provide your comprehensive diagnosis based on all the observations and the image provided."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ])
        ])
        chain = prompt | llm | JsonOutputParser(pydantic_object=Act)

        # Use retry mechanism for robust JSON parsing
        input_data = {
            "input": state.get("input"),
            "observations": observations_str,
            "json_error_hint": "",  # Will be filled by retry mechanism if needed
            "error_hint": ""  # Will be filled by retry mechanism if needed
        }
        act_obj = invoke_llm_with_json_retry(
            chain,
            input_data,
            max_retries=3,
            pydantic_model=Act
        )

        return {"response": act_obj.get("response"), "next_tool_call": None, "next_tool_calls_batch": None}

def execute_tool(state: OphthaAgentState) -> Dict:
    """
    Executes the tool(s) decided by the planner.
    Supports both single tool execution and parallel batch execution.
    """
    single_tool_call = state.get("next_tool_call")
    batch_tool_calls = state.get("next_tool_calls_batch")

    # Case 1: Single tool execution (sequential mode)
    if single_tool_call is not None:
        name = single_tool_call.get("tool_name")
        args = single_tool_call.get("tool_args", {})
        tool = tools.get(name)

        if tool is None:
            error_msg = f"Tool '{name}' not found."
            return {"past_steps": [(single_tool_call, {"error": error_msg})]}

        try:
            output = tool(**args)
            return {"past_steps": [(single_tool_call, {"output": output})]}
        except Exception as e:
            return {"past_steps": [(single_tool_call, {"error": f"{type(e).__name__}: {e}"})]}

    # Case 2: Batch parallel execution
    elif batch_tool_calls is not None and len(batch_tool_calls) > 0:
        import time
        from datetime import datetime

        def execute_single_tool(tool_call_dict):
            """Helper function to execute a single tool."""
            name = tool_call_dict.get("tool_name")
            args = tool_call_dict.get("tool_args", {})
            tool = tools.get(name)

            if tool is None:
                error_msg = f"Tool '{name}' not found."
                return (tool_call_dict, {"error": error_msg})

            try:
                start_time = time.perf_counter()
                start_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"  [{start_timestamp}] ⚡ Started: {name}")

                output = tool(**args)

                end_time = time.perf_counter()
                end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                elapsed = end_time - start_time
                print(f"  [{end_timestamp}] ✓ Completed: {name} ({elapsed:.2f}s)")

                # Add timing info to output
                if isinstance(output, dict):
                    output['_execution_time'] = elapsed
                    output['_start_timestamp'] = start_timestamp
                    output['_end_timestamp'] = end_timestamp

                return (tool_call_dict, {"output": output})
            except Exception as e:
                end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"  [{end_timestamp}] ✗ Failed: {name} - {e}")
                return (tool_call_dict, {"error": f"{type(e).__name__}: {e}"})

        # Execute all tools in parallel using ThreadPoolExecutor
        results = []
        batch_start = time.perf_counter()
        batch_start_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        print(f"\n{'='*70}")
        print(f"🚀 PARALLEL EXECUTION BATCH")
        print(f"{'='*70}")
        print(f"Start time: {batch_start_timestamp}")
        print(f"Number of tools: {len(batch_tool_calls)}")
        print(f"Max workers: {len(batch_tool_calls)}")
        print(f"Tools: {', '.join([tc.get('tool_name') for tc in batch_tool_calls])}")
        print(f"{'='*70}\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_tool_calls)) as executor:
            # Submit all tool executions
            future_to_tool = {
                executor.submit(execute_single_tool, tool_call): tool_call
                for tool_call in batch_tool_calls
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_tool):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    tool_call = future_to_tool[future]
                    results.append((tool_call, {"error": f"Parallel execution error: {type(e).__name__}: {e}"}))

        batch_end = time.perf_counter()
        batch_end_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        batch_duration = batch_end - batch_start

        print(f"\n{'='*70}")
        print(f"✅ PARALLEL EXECUTION COMPLETED")
        print(f"{'='*70}")
        print(f"End time: {batch_end_timestamp}")
        print(f"Total batch duration: {batch_duration:.2f}s")
        print(f"Tools completed: {len(results)}/{len(batch_tool_calls)}")

        # Calculate if execution was truly parallel
        individual_times = []
        for tool_call, output_dict in results:
            if 'output' in output_dict and isinstance(output_dict['output'], dict):
                exec_time = output_dict['output'].get('_execution_time', 0)
                if exec_time > 0:
                    individual_times.append(exec_time)

        if individual_times:
            sum_sequential = sum(individual_times)
            speedup = sum_sequential / batch_duration if batch_duration > 0 else 1
            efficiency = (speedup / len(batch_tool_calls)) * 100 if len(batch_tool_calls) > 0 else 0

            print(f"Sequential estimate: {sum_sequential:.2f}s")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Parallel efficiency: {efficiency:.1f}%")

        print(f"{'='*70}\n")

        return {"past_steps": results}

    # Case 3: No tool to execute
    else:
        raise ValueError("Controller decided to execute a tool, but no tool or batch was found.")

def summarizer(state: OphthaAgentState):
    """
    Summarizes the output of the last tool call(s) into textual observation(s).
    Supports both single tool execution and parallel batch execution.
    """
    past_steps = state.get("past_steps", [])

    # Determine how many new steps were added
    # For single execution: 1 step
    # For batch execution: multiple steps
    batch_tool_calls = state.get("next_tool_calls_batch")

    if batch_tool_calls is not None and len(batch_tool_calls) > 0:
        # Batch mode: summarize all tools from the last batch
        num_new_steps = len(batch_tool_calls)
        steps_to_summarize = past_steps[-num_new_steps:]
    else:
        # Single mode: summarize only the last step
        steps_to_summarize = [past_steps[-1]]

    observations = []

    for tool_call, tool_output in steps_to_summarize:
        tool_call_str = json.dumps(tool_call, indent=2)
        tool_output_str = json.dumps(tool_output, indent=2)

        system_prompt = """You are a professional ophthalmology assistant. Your task is to interpret and summarize the results from a diagnostic tool.
Extract key information from the tool's output and format your summary according to the examples below. Only output the summary for the corresponding tool.

Tool Call: {tool_call}
Tool Output: {tool_output}

**Formatting Examples:**
1.  **For image preprocess tools (e.g., preprocess_image):**
    *(e.g., The preprocess_image tool successfully/unsuccessfully processed the image)*

1.  **For image quality tools (e.g., assess_image_quality):**
    `good quality / bad quality Confidence: <confidence_score>`
    *(e.g., good quality Confidence: 0.95)*

2.  **For "detect_lesions" tool:**
    `Lesion segmentation detected <count> Cotton Wool Spots, <count> Exudates, <count> Hemorrhages, <count> Microaneurysms`
    *(e.g., Lesion segmentation detected 1 Cotton Wool Spots, 0 Exudates, 8 Hemorrhages, 0 Microaneurysms)*

3.  **For "segment_by_ddcs" tool:**
    Provide a descriptive summary of the quantitative analysis.
    *(e.g., The analyzed fundus image shows a vascular density of 0.014378, indicating a relatively low proportion of visible vasculature within the retinal field. The optic disc area ratio is 0.006458, consistent with typical proportions for standard fundus fields.Hemorrhagic changes are minimal, with a hemorrhage area ratio of 0.00526 and a total hemorrhage size of 1,379 pixels, suggesting only small localized bleeding.Hard exudates are more pronounced, with a hard exudate area ratio of 0.032852, a distribution index of 0.333966, and an average lesion size of 36.57 pixels, indicating moderate but spatially clustered lipid deposits. Microaneurysm activity is very limited, with 1 detected lesion and a microaneurysm area ratio of 0.000221.No soft exudates were detected (area ratio, distribution index, and mean area = 0), suggesting an absence of acute ischemic or cotton-wool spots.)*

4.  **For "DR_Grading" tool:**
    `The DR_grading tool predicted <disease_severity> (confidence: <confidence_score>). The image quality: <quality_level> (confidence: <confidence_score>)`
    *(e.g., The DR_grading tool predicted Moderate retinal disease (confidence: 0.379126). The image quality: high quality (confidence: 0.752380))*

5.  **For "autoMorphProcess" or other quantitative analysis tools:**
    Please extract and analyze the following indicators: image quality, optic disc-to-cup ratio, overall vascularity, arterial features, venous features, and arteriovenous ratio.
    *(e.g., optic disc-to-cup ratio: CDR_vertical ≈ 0.38, CDR_horizontal ≈ 0.41 — within normal range, no indication of glaucoma).
    Overall vascularity: Fractal_dimension 1.45, Vessel_density 0.077 — normal vascular morphological complexity and density.
    Arterial features: Artery_Vessel_density 0.0227, Tortuosity_density 0.766 — mild tortuosity, commonly seen in hypertension/age-related changes.
    Venous features: Vein_Vessel_density 0.0424, Tortuosity_density 0.740 — moderate vein width and tortuosity, no significant dilation.
    Arteriovenous ratio (AVR): zone_b 0.622, zone_c 0.693 — close to the lower limit of normal; observation in conjunction with blood pressure is recommended.
    AutoMorph: Fundus structure and vascular parameters are within the physiological or mildly altered range; no obvious pathological abnormalities were observed.)*

6.  **For "AMD_predict_fundus_by_deepseenet" tool:**
    `AREDS严重程度评分 (0-5分)：<score>分. Prediction details: drusen - <label> (confidence: <confidence_score>), pigment - <label> (confidence: <confidence_score>), amd - <label> (confidence: <confidence_score>), ga - <label> (confidence: <confidence_score>), cga - <label> (confidence: <confidence_score>)`
    *(e.g., AREDS严重程度评分 (0-5分)：0分. Prediction details: drusen - Large (confidence: 0.50), pigment - Yes (confidence: 0.88), amd - Yes (confidence: 0.56), ga - No (confidence: 0.91), cga - No (confidence: 1.00))*

Based on the provided Tool Call and Tool Output, generate a concise observation following the format of the corresponding example above.
"""
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        llm = config.agent_decision.llm
        chain = prompt | llm

        observation = chain.invoke({
            "tool_call": tool_call_str,
            "tool_output": tool_output_str
        }).content

        observations.append(observation)

        # Log the details of this step
        log_entry = {
            "thought": state.get("thought"),
            "tool_call": tool_call,
            "tool_output": tool_output,
            "observation": observation
        }
        execution_log.append(log_entry)

    return {"observations": observations}

# Termination Condition
def should_continue(state: OphthaAgentState) -> Literal["executor", "__end__"]:
    """
    Determines whether to continue the loop or terminate.
    The loop terminates when there is a response or no next tool call.
    Supports both single and batch execution modes.
    """
    if state.get("response"):
        return "__end__"

    # Check if there's any work to do (single or batch)
    if state.get("next_tool_call") or state.get("next_tool_calls_batch"):
        return "executor"

    return "__end__"


# Define the graph
workflow = StateGraph(OphthaAgentState)

workflow.add_node("planner", planner)
workflow.add_node("executor", execute_tool)
workflow.add_node("summarizer", summarizer)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    should_continue,
    {
        "executor": "executor",
        "__end__": END
    }
)
workflow.add_edge("executor", "summarizer")
workflow.add_edge("summarizer", "planner") # Loop back to the planner to think again

# Compile the graph
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    image_path = "<ANON_ABS_PATH>"
    user_query = """What does the image depict?\nA: Glaucoma\nB: Central serous retinopathy\nC: Moderate diabetic retinopathy\nD: Mild diabetic retinopathy"""
    initial_state = {
        "input": user_query,
        "image_path": image_path,
        "past_steps": [],
        "observations": [],
    }

    final_state = app.invoke(initial_state, config=thread_config)

    final_response = final_state.get("response")

    print("--- Final Diagnosis ---")
    print(final_response)