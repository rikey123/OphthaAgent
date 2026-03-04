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
MAX_TOOL_CALLS = 6


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
    tool_sequence = ["preprocess_image", "detect_lesions", "segment_by_ddcs", "DR_Grading","autoMorphProcess"]
    past_steps = state.get("past_steps", [])
    num_past_steps = len(past_steps)

    llm = config.agent_decision.llm
    
    observations_str = "\n".join(state.get('observations', []))

    if num_past_steps < len(tool_sequence):
        next_tool_name = tool_sequence[num_past_steps]
        
        output_dir = "<ANON_ABS_PATH>"
        image_path = state.get("image_path")
        base_name = os.path.basename(image_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        if num_past_steps > 0:
            last_tool_call, last_output = past_steps[-1]
            tool_result = last_output.get('output', {}).get('result', {})
            if isinstance(tool_result, dict) and tool_result.get('save_path'):
                input_image_path = tool_result['save_path']
            else:
                # If the last tool didn't produce a new image, reuse its input image.
                input_image_path = last_tool_call.get('tool_args', {}).get('image_path', image_path)
        else:
            input_image_path = image_path

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
        
        return {"next_tool_call": tool_call_dict, "response": None}


    else:

        system_prompt = """You are an expert ophthalmologist AI.
    Your original goal was to answer: {input}

    You have completed all the planned tool calls and have the following observations:
    {observations}

    Based on all the key findings of the all above tool calls gathered, provide your final comprehensive diagnosis. 
    Please comprehensively consider all the key evidence and clues discovered by the tools mentioned above, without omitting any. Any conflicts should be discussed in conjunction with the confidence level.
    Respond in JSON format that adheres to the `Act` schema. Your JSON output MUST contain a `thought` field and a `response` field. The `tool_call` field must not be present. The thought process must be based on the key findings from all the above observations, and a conclusion must be reached after step-by-step thinking.

    STRICT OUTPUT RULES:
    - You MUST place the final answer tagged<answer>...</answer>
    - Do NOT include chain-of-thought, explanations, or rationale.
    - You are doing the VQA task. The answer must be options in user questions(e.g. <answer>B</answer>).
    - If uncertain, choose the most probable option.
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
        
        return {"response": act_obj.get("response"), "next_tool_call": None}

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

Based on the provided Tool Call and Tool Output, generate a concise observation following the format of the corresponding example above.
"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    llm = config.agent_decision.llm
    chain = prompt | llm
    
    observation = chain.invoke({
        "tool_call": tool_call_str,
        "tool_output": tool_output_str
    }).content
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