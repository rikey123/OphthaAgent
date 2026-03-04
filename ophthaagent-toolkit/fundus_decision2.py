"""
Ophthalmic Diagnosis Agent based on LangGraph.
"""

import json
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
import sys
from agents.fundus_tools_agent.tools_interface import enhance_fundus_image
from agents.fundus_tools_agent.tools_interface import fundus_lesion_segmentation
from agents.fundus_tools_agent.tools_interface import crop_by_fit
from agents.fundus_tools_agent.tools_interface import enhance_by_fit
from agents.fundus_tools_agent.tools_interface import quality_assess_by_fit
from agents.fundus_tools_agent.tools_interface import fov_od_localization_by_fit
from agents.fundus_tools_agent.tools_interface import vessel_segment_by_fit
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from config_fundus import Config


from agents.fundus_tools_agent.Lesion_detection.lesion_detection import FundusLesionDetection
from agents.fundus_tools_agent.anatomical_segmentation.anatomical_segmentation import FundusAnatomicalSegmentation
from agents.fundus_tools_agent.image_quality.image_quality_assessment import FundusImageQuality
# Note the typo in the directory name 'measuremeng'
from agents.fundus_tools_agent.preprocessing.image_preprocessing import FundusImagePreprocessing

from agents.fundus_tools_agent.tools.tools_interface1 import (
    run_automorph_m0_m1,
    run_automorph_m2,
    run_quantitative_measurement
)
from rag_query import rag_query
from tool_call_test import autoMorphProcess

from langgraph.checkpoint.memory import MemorySaver


#import cv2
import numpy as np
from config_fundus import Config

import os


load_dotenv()

# Load application configuration settings
config = Config()

# Initialize memory for saving and restoring graph state
memory = MemorySaver()

# Specify a unique ID for the current execution thread for state management
thread_config = {"configurable": {"thread_id": "1"}}

# 明确最多工具调用次数
MAX_TOOL_CALLS = 5


class ToolCall(BaseModel):
    """Represents a single tool call in the plan."""
    tool_name: str = Field(description="The name of the tool to call.")
    tool_args: dict = Field(description="The arguments for the tool.")
    reason: str = Field(description="Brief explanation of why this tool call is necessary.")

class Plan(BaseModel):
    """A plan of tool calls to execute."""
    steps: List[ToolCall] = Field(
        description="A list of tool calls to execute in sequence."
    )

class Act(BaseModel):
    """The action to take next, which can be a reflection, a plan, or a final response."""
    reflection: Optional[str] = Field(default=None, description="A brief reflection on the progress so far.")
    plan: Optional[Plan] = Field(default=None, description="The next sequence of tool calls to execute.")
    response: Optional[str] = Field(default=None, description="The final diagnosis or answer to the user.")

    @model_validator(mode='before')
    def check_one_action(cls, values):
        """Ensure that exactly one action (reflection, plan, or response) is set."""
        if isinstance(values, dict):
            action_fields = ['reflection', 'plan', 'response']
            set_actions = [field for field in action_fields if values.get(field) is not None]
            if len(set_actions) != 1:
                raise ValueError(f"Exactly one of {action_fields} must be set. Found: {set_actions}")
        return values

class OphthaAgentState(TypedDict):
    """State tracker for the ophthalmic diagnosis agent."""
    input: str
    image_path: str
    plan: List[ToolCall]
    past_steps: Annotated[List[Tuple[ToolCall, Dict]], operator.add]
    reflection: str


# 定义可用眼底工具列表
tools = {
    "assess_image_quality": quality_assess_by_fit,
    "preprocess_image": enhance_fundus_image,
    "localize_fov_and_od": fov_od_localization_by_fit,
    "segment_vessel":vessel_segment_by_fit,
    "detect_lesions": fundus_lesion_segmentation,
    "autoMorphProcess":autoMorphProcess,
    "rag_query":rag_query
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
    
 "rag_query": Invoke the RAG tool to retrieve relevant medical knowledge and diagnostic quantitative metrics from the knowledge base. Suggested query: "根据眼底图像进行诊断，比如糖尿病需要那些量化指标?"
     tool_args:
    query (str): your query question.
"""""

# --- Node implementations ---

def planner(state: OphthaAgentState):
    """Generates the initial plan."""
    system_prompt = f"""You are an expert ophthalmologist AI. Your goal is to provide a comprehensive diagnosis for a given fundus image.
Based on the user's query, create a step-by-step plan to provide a diagnosis. Only select the necessary tools.

User query: "{state.get("input")}"
Image is located at: "{state.get("image_path")}"

Available tools:
{get_tool_descriptions()}

Your plan should be a sequence of tool calls. For any tool that needs an image path (like `assess_image_quality` or `preprocess_image`), you MUST use the `image_path` provided above in the `tool_args`.
For tools that run after preprocessing, you must reason about the correct path to use as input, which is usually the output from the `preprocess_image` tool.

Now, create the plan as a JSON object that adheres to the `Plan` schema. For each step, you must provide a `tool_name`, the `tool_args`, and a `reason` explaining why this step is necessary Please strictly generate JSON using these field names, and do not modify the field names on your own.

Important rules:
1. The top-level output MUST be a JSON object with a key "steps".
2. DO NOT output a raw list like [..., ...]. That is invalid.
3. DO NOT include any extra text, markdown, or explanation outside the JSON.
5. Output ONLY the JSON and nothing else.

Now generate the plan as a valid JSON object adhering exactly to the above format.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ])
    llm = config.agent_decision.llm
    chain = prompt | llm | JsonOutputParser(pydantic_object=Plan)
    plan_obj = chain.invoke({"input": state["input"], "image_path": state["image_path"]})
    return {"plan": plan_obj.get('steps', []), "past_steps": [], "reflection": ""}

def execute_tool(tool_call: dict) -> Dict:
    """Executes a single tool call."""
    tool_name = tool_call['tool_name']
    tool_args = tool_call['tool_args']
    
    if tool_name not in tools:
        return {"error": f"Tool '{tool_name}' not found."}
    
    tool = tools[tool_name]
    try:
        # Explicitly call the correct method for each tool to avoid ambiguity.
        if tool_name == "assess_image_quality":
            result = tool(**tool_args)
        elif tool_name == "preprocess_image":
            result = tool(**tool_args)
        elif tool_name == "localize_fov_and_od":
            result = tool(**tool_args)
        elif tool_name == "segment_vessel":
            result = tool(**tool_args)
        elif tool_name == "detect_lesions":
            result = tool(**tool_args)
        elif tool_name == "measure_quantitative_metrics":
            result = tool(**tool_args)
        elif tool_name == "autoMorphProcess":
            result = tool(**tool_args)
        elif tool_name == "rag_query":
            result = tool(**tool_args)
        else:
            return {"error": f"Tool '{tool_name}' has no defined execution logic."}
        print(result)
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}

def executor(state: OphthaAgentState):
    """Executes the next step in the plan."""
    if not state["plan"]:
        # This can happen if the replanner returns an empty plan, signaling completion.
        return {}
        
    next_step = state["plan"][0]
    remaining_plan = state["plan"][1:]
    
    # Execute the tool
    tool_output = execute_tool(next_step)
    
    return {
        "plan": remaining_plan,
        "past_steps": [(next_step, tool_output)] # operator.add will append this to the existing list
    }

def decision_maker(state: OphthaAgentState):
    used_calls = len(state.get("past_steps", []))

    system_prompt = f"""You are a highly experienced clinical ophthalmologist. Please provide as professional a diagnosis as possible based solely on the historical analysis of the fundus image.

    Your original goal was to answer: {{input}}
    The image is located at: {{image_path}}

    You have already completed the following steps (tool call and output):
    {{past_steps}}

    """
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    llm = config.agent_decision.llm
    #chain = prompt | llm | JsonOutputParser(pydantic_object=Act)
    chain = prompt | llm

    past_steps_str = "\n".join([f"Tool Call: {step[0]}\nOutput: {step[1]}" for step in state["past_steps"]])
    act_obj = chain.invoke({
        "input": state["input"],
        "image_path": state["image_path"],
        "past_steps": past_steps_str,
    })
    return act_obj


def replanner(state: OphthaAgentState):
    """Reviews the past steps and decides the next action."""
    used_calls = len(state.get("past_steps", []))

    system_prompt = f"""You are an expert ophthalmologist AI, acting as a replanner and reflector.
Your task is to review the diagnostic process for a fundus image, reflect on the findings, and decide the next course of action.

Your original goal was to answer: {{input}}
The image is located at: {{image_path}}

You have already completed the following steps (tool call and output):
{{past_steps}}

Available tools:
{get_tool_descriptions()}

**VERY IMPORTANT**:
- For any tool that needs an image path, you MUST use the `image_path` provided above.
- For tools that run after preprocessing, you must use the output path from the `preprocess_image` tool.
- When calling `measure_quantitative_metrics`, you MUST use the `segmentation_results` from the `segment_anatomical_structures` tool.

Now, decide on the next action by responding in JSON format that adheres to the `Act` schema.
Your response must contain exactly one of the following fields: `reflection`, `plan`, or `response`.

1.  If you need to reflect on the progress, provide a `reflection` string.
2.  If you have sufficient information for a final diagnosis, provide a `response` string.
3.  If you need more information, create a `plan` with the next tool calls. You have already used {used_calls} out of {MAX_TOOL_CALLS} tool calls.
"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    llm = config.agent_decision.llm
    chain = prompt | llm | JsonOutputParser(pydantic_object=Act)

    past_steps_str = "\n".join([f"Tool Call: {step[0]}\nOutput: {step[1]}" for step in state["past_steps"]])
    act_obj = chain.invoke({
        "input": state["input"],
        "image_path": state["image_path"],
        "past_steps": past_steps_str,
    })

    if act_obj.get("reflection"):
        return {"reflection": act_obj.get("reflection")}

    if act_obj.get("response"):
        return {"response": act_obj.get("response"), "plan": []} # End of process

    if used_calls >= MAX_TOOL_CALLS and act_obj.get("plan"):
        finalizer_system_prompt = f"""You have reached the maximum tool-call limit ({MAX_TOOL_CALLS}).
Based on the executed steps, provide the final diagnosis NOW as a `response`.
Respond strictly with a JSON object adhering to the `Act` schema, with only the `response` field populated.
"""
        finalizer_prompt = ChatPromptTemplate.from_messages([("system", finalizer_system_prompt)])
        finalizer_chain = finalizer_prompt | llm | JsonOutputParser(pydantic_object=Act)
        final_act = finalizer_chain.invoke({"input": state["input"], "past_steps": past_steps_str})
        return {"response": final_act.get("response"), "plan": []}

    plan_from_llm = act_obj.get("plan")
    steps = []
    if isinstance(plan_from_llm, dict):
        steps = plan_from_llm.get("steps", [])
    elif isinstance(plan_from_llm, list):
        steps = plan_from_llm
    return {"plan": steps}

# Termination Condition
def should_continue(state: OphthaAgentState) -> Literal["executor", "__end__"]:
    """
    Determines whether to continue the loop or terminate.
    The loop terminates when the plan is empty.
    """
    if not state.get("plan"):
        return "__end__"
    return "executor"


# Define the graph
workflow = StateGraph(OphthaAgentState)

workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("replanner", replanner)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "replanner") # After executing, we always replan

workflow.add_conditional_edges(
    "replanner",
    should_continue,
    {
        "executor": "executor",
        "__end__": END
    }
)

# Compile the graph
app = workflow.compile(checkpointer=memory)
#app.get_graph().draw_png()

if __name__ == "__main__":
    # This is an example of how to run the agent.
    # In a real application, you would replace the image path and query.
    image_path = "<ANON_ABS_PATH>"  # Replace with a real image path
    user_query = "Does this fundus image show any signs of diabetic retinopathy? A: Yes B: No."

    # Initial state
    initial_state = {
        "input": user_query,
        "image_path": image_path,
    }

    # Run the agent
    final_state = app.invoke(initial_state, config=thread_config)

    # The final response is now in the 'response' field of the final state
    final_response = final_state.get("response", final_state.get('response'))

    result = decision_maker(final_state)
    # Print the final diagnosis
    print("--- Final Diagnosis ---")
    print(result)

    print("\n--- Agent Reflections ---")
    print(final_state.get('reflection'))


    # You can also inspect the full state
    # print("\n--- Full Final State ---")
    # print(final_state)