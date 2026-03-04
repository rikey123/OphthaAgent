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
from datetime import datetime
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
    Lesion_predict_OCT_by_opticnet
)

from rag_query import rag_query
from tool_call_test import autoMorphProcess
from langgraph.checkpoint.memory import MemorySaver
import numpy as np



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
    log_file_path: str  # Add this line
    past_steps: Annotated[List[Tuple[ToolCall, Dict]], operator.add]
    observations: Annotated[List[str], operator.add]
    response: Optional[str]
    next_tool_call: Optional[ToolCall]


def amd_predict_wrapper(**kwargs):
    """
    Wrapper for AMD_predict_fundus_by_deepseenet to simplify its output.
    It processes the raw JSON output to extract key findings for each lesion,
    including its label and confidence score, as well as the overall AREDS simplified severity score.
    """
    raw_output = AMD_predict_fundus_by_deepseenet(**kwargs)

    if not isinstance(raw_output, dict):
        return raw_output

    result_data = raw_output.get('result', {})
    if not isinstance(result_data, dict):
        return {
            "simplified_score": {},
            "lesions": {"error": "Invalid format, 'result' key not found or not a dict"}
        }

    # Extract simplified score, which is a key diagnostic conclusion
    simplified_score_data = result_data.get('simplified_score', {})

    # Extract and simplify lesion details
    simplified_lesions = {}
    for lesion, details in result_data.get('analysis_results', {}).items():
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
    "autoMorphProcess":autoMorphProcess,
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
    It forces the agent to follow a predefined sequence of tool calls.
    Only after the sequence is complete does it generate a final response.
    """
    tool_sequence = ["detect_lesions", "segment_by_ddcs","AMD_predict_fundus_by_deepseenet" ,"DR_Grading","autoMorphProcess"]
    # tool_sequence = ["DR_Grading"]
    past_steps = state.get("past_steps", [])
    num_past_steps = len(past_steps)

    llm = config.agent_decision.llm
    
    observations_str = "\n".join(state.get('observations', []))

    if num_past_steps < len(tool_sequence):
        next_tool_name = tool_sequence[num_past_steps]
        
        image_path = state.get("image_path")
        base_name = os.path.basename(image_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        
        # Create a unique output directory for this run
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir_name = f"{file_name_without_ext}_{date_str}"
        output_dir = os.path.join("<ANON_ABS_PATH>", output_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        # Copy the original image to the new output directory
        if num_past_steps == 0:
            import shutil
            shutil.copy(image_path, os.path.join(output_dir, base_name))

        # --- Logic to determine the input image path for the current tool ---

        original_image_path = state.get("image_path")

        # Case 1: If 'preprocess_image' is NOT in the tool sequence, all tools use the original image.
        if "preprocess_image" not in tool_sequence:
            input_image_path = original_image_path
        
        # Case 2: If 'preprocess_image' is in the tool sequence, apply the special logic.
        else:
            # Check if the preprocessed image has been created yet.
            preprocessed_image_path = None
            for tool_call, output in past_steps:
                if tool_call.get("tool_name") == "preprocess_image":
                    tool_result = output.get('output', {}).get('result', {})
                    if isinstance(tool_result, dict) and tool_result.get('save_path'):
                        preprocessed_image_path = tool_result['save_path']
                        break
            
            # If the preprocessed image exists, all subsequent tools must use it.
            if preprocessed_image_path:
                input_image_path = preprocessed_image_path
            # If the preprocessed image does not exist yet (i.e., we are in steps before it),
            # use the chained output-to-input logic.
            elif num_past_steps > 0:
                last_tool_call, last_output = past_steps[-1]
                tool_result = last_output.get('output', {}).get('result', {})
                if isinstance(tool_result, dict) and tool_result.get('save_path'):
                    input_image_path = tool_result['save_path']
                else:
                    input_image_path = last_tool_call.get('tool_args', {}).get('image_path', original_image_path)
            # For the very first step (when num_past_steps is 0).
            else:
                input_image_path = original_image_path

        # Determine output path
        if next_tool_name == "preprocess_image":
            output_path = os.path.join(output_dir, f"{file_name_without_ext}_preprocessed.jpg")
        elif next_tool_name == "detect_lesions":
            output_path = os.path.join(output_dir, f"{file_name_without_ext}_lesions.jpg")
        elif next_tool_name == "segment_by_ddcs":
            output_path = os.path.join(output_dir, f"{file_name_without_ext}_ddcs_segmentation")
        else:
            output_path = None

        # This prompt forces the LLM to generate a thought for the mandatory tool call
        system_prompt = """You are an expert ophthalmologist AI in a diagnostic workflow.
Your original goal is to answer: {input}

Observations from past steps:
{observations}

You are in a sequence of operations and must call the next tool. The next tool is: {next_tool_name}.
Provide a brief thought process for this step.
Respond in JSON format that adheres to the `Act` schema. Your response MUST contain a `thought` and a `tool_call` field. The `tool_call` field can be an empty object as it will be ignored.
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
        act_obj = chain.invoke({
            "input": state.get("input"),
            "observations": observations_str,
            "next_tool_name": next_tool_name,
        })

        # Construct the tool call dictionary deterministically
        tool_call_dict = {
            "tool_name": next_tool_name,
            "tool_args": {"image_path": input_image_path},
        }
        if output_path:
            tool_call_dict["tool_args"]["output_path"] = output_path
        
        # Log the thought and planned tool call
        log_file_path = state.get("log_file_path")
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"--- Step {num_past_steps + 1} ---\n")
            f.write(f"Thought: {act_obj.get('thought')}\n")
            f.write(f"Tool Call: {json.dumps(tool_call_dict, indent=2, ensure_ascii=False)}\n")

        return {"next_tool_call": tool_call_dict, "response": None}


    else:

        system_prompt = """
        You are an expert ophthalmologist AI.

Original VQA-style question (single- or multiple-choice, with options):
{input}

You have already run all diagnostic tools. Below are their summarized observations, including key quantitative values and detected lesions:
{observations}

YOUR TASK:
Integrate ALL observations above (including numerical values, severity scores, and confidence levels), think step by step, and choose ONE OR MORE options among the given choices that are supported by the evidence.

REASONING GUIDELINES (for the `thought` field ONLY, not visible to the user):
1. Briefly comment on overall image quality (e.g. good/poor, tool confidence).
2. List the main POSITIVE findings with numbers and severity, such as:
   - DR_Grading result and its confidence
   - Lesion counts and area ratios (microaneurysms, hemorrhages, exudates, etc.)
   - AMD AREDS simplified severity score and lesion predictions
   - Optic disc / cup ratio and glaucomatous changes
   - Vascular parameters (vessel density, tortuosity, AVR, etc.)
3. List key NEGATIVE / ABSENT findings that help rule out diagnoses
   (e.g. “no glaucomatous CDR enlargement”, “no AMD-related lesions detected”).
4. Reconcile any CONFLICTS between tools by referring to:
   - image quality,
   - confidence scores,
   - and the clinical plausibility of each result.
5. Map the combined evidence to EACH answer option given in the question
   (e.g. A, B, C, D, E,etc.):
   - briefly state whether the evidence supports or contradicts each option.
6. Decide for EACH option whether it should be SELECTED or NOT SELECTED.
   - Select ALL options that are well supported by the integrated evidence.
   - If none is clearly correct, choose the SINGLE most probable option.
   - At the end of your `thought`, clearly list all selected option labels.

OUTPUT FORMAT (must strictly follow the `Act` schema):
- Your entire output MUST be a single valid JSON object.
- Fields:
  - "thought": your internal step-by-step reasoning (≤ 200 words, in English).
  - "response": the final answer tag ONLY.

Example JSON when only one option is chosen:
{{
  "thought": "<your internal reasoning here>",
  "response": "<answer>C</answer>"
}}

Example JSON when multiple options are chosen:
{{
  "thought": "<your internal reasoning here>",
  "response": "<answer>A,C,F</answer>"
}}

STRICT RULES FOR `response`:
- "response" MUST contain ONLY one final answer tag in the form: <answer>...</answer>.
- Inside the tag, list ALL chosen option labels separated by commas and WITHOUT spaces,
  e.g. <answer>A,C,F</answer>.
- The option labels MUST EXACTLY match the identifiers used in the question
  (e.g. A,B,C,D etc.).
- Do NOT include any explanations, chain-of-thought, or extra text in "response".
- Do NOT include a "tool_call" field in the JSON.
- Even if the evidence is imperfect or partially conflicting, you MUST output at least
  one option (the most plausible single option or a small set of plausible options).
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
        act_obj = chain.invoke({
            "input": state.get("input"),
            "observations": observations_str
        })
        
        # Log the final thought process
        log_file_path = state.get("log_file_path")
        if log_file_path:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write("--- Final Thought ---\n")
                f.write(f"{act_obj.get('thought')}\n\n")

        # Serialize the final response object to a JSON string
        final_response_json = json.dumps({
            "thought": act_obj.get("thought"),
            "response": act_obj.get("response")
        })

        return {"response": final_response_json, "next_tool_call": None}

def execute_tool(state: OphthaAgentState) -> Dict:
    """Executes the tool decided by the planner."""
    tool_call = state.get("next_tool_call")
    if tool_call is None:
        raise ValueError("Controller decided to execute a tool, but no tool was found.")
    
    name = tool_call.get("tool_name")
    args = tool_call.get("tool_args", {})
    tool = tools.get(name)
    
    if tool is None:
        error_msg = f"Tool '{name}' not found."
        return {"past_steps": [(tool_call, {"error": error_msg})]}
    
    try:
        output = tool(**args)
        return {"past_steps": [(tool_call, {"output": output})]}
    except Exception as e:
        return {"past_steps": [(tool_call, {"error": f"{type(e).__name__}: {e}"})]}

def summarizer(state: OphthaAgentState):
    """Summarizes the output of the last tool call into a textual observation."""
    last_step = state.get("past_steps", [])[-1]
    tool_call, tool_output = last_step

    tool_call_str = json.dumps(tool_call, indent=2, ensure_ascii=False)
    tool_output_str = json.dumps(tool_output, indent=2, ensure_ascii=False)

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

    Your output MUST follow a two-part structure.

    First, summarize the output of each sub-model (e.g., DRAC, DEEPDR, EYEQ) in a structured list.
    Use this exact template for each item in the list:
    - DR_Grading (<model_name>):predicted <dr_severity_label>(confidence: <probability_value>). Image quality: <quality_label>(confidence: <probability_value>).

    Part 2: Integrated Summary
    After the list, provide a single, short summary line that integrates the results from all models.
    Use this exact template for the summary:
    `Integrated DR_Grading summary: majority prediction — <combined_disease_severity>, with average confidence ≈ <mean_confidence>. Overall image quality — <quality_consensus>.`

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

    # Log the tool output and observation
    log_file_path = state.get("log_file_path")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"Tool Output: {tool_output_str}\n")
        f.write(f"Observation: {observation}\n\n")

    return {"observations": [observation]}

# Termination Condition
def should_continue(state: OphthaAgentState) -> Literal["executor", "__end__"]:
    """
    Determines whether to continue the loop or terminate.
    The loop terminates when there is a response or no next tool call.
    """
    if state.get("response") or not state.get("next_tool_call"):
        return "__end__"
    return "executor"


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

    # Create a unique output directory for this run
    base_name = os.path.basename(image_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir_name = f"{file_name_without_ext}_{date_str}"
    output_dir = os.path.join("<ANON_ABS_PATH>", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file_path = os.path.join(output_dir, "reasoning_log.txt")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Image Path: {image_path}\n")
        f.write(f"Query: {user_query}\n\n")

    initial_state = {
        "input": user_query,
        "image_path": image_path,
        "log_file_path": log_file_path,
        "past_steps": [],
        "observations": [],
    }

    final_state = app.invoke(initial_state, config=thread_config)

    final_response = final_state.get("response")

    # Log the final answer
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write("--- Final Thought ---\n")
        f.write(f"{final_response}\n")

    print("--- Final Diagnosis ---")
    print(final_response)
