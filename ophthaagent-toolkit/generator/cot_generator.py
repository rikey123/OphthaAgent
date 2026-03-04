"""
CoT数据生成器 - 通过大模型API批量生成高质量推理轨迹数据
"""
import json
import os
import time
from typing import List, Dict, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # 也尝试从项目根目录加载
    load_dotenv()

from api_client import APIClient
from dataset_loader import DatasetLoader, CoTExample

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoTGenerator:
    """CoT数据生成器"""
    
    SYSTEM_PROMPT = """### System Prompt For Generating Deep-Reasoning Medical Trajectories

**Role Definition**

You are an advanced Ophthalmic Agent Simulator. Your goal is to generate a **multi-step, visually-grounded reasoning trajectory** that solves a medical `Question` based on a known `Ground Truth Answer`. Act as a meticulous retinal specialist. Never reveal that you know the ground truth in advance—behave as if you infer everything from the image, tools, and knowledge.

**Core Requirements for "Thinking" (The Brain)**

1. **Visual Hallucination:** You MUST start by simulating a visual inspection. Describe specific features (e.g., "I observe scattered dot-blot hemorrhages," "The optic cup looks vertically elongated") that justify your hypothesis. **Do not just repeat the question.**
2. **Multi-Tool Strategy:** Complex diseases often require multiple tools.
   * *Example:* For Diabetic Retinopathy (DR), you often need `DR_Grading` (for severity) AND `detect_lesions` (for quantification).
   * *Example:* For Glaucoma, you might need `autoMorphProcess` (structure) AND `rag_query` (guidelines).
3. **RAG Integration:** If the question involves treatment advice, prognosis, or medical definitions, you MUST plan a `rag_query` step after getting image analysis results.
4. **Evidence-Based:** Every decision to call a tool must be backed by a visual observation or a previous tool's result.
5. **No Explicit Ground-Truth References:** Never mention that you are reverse-engineering from a provided answer. Present the reasoning as if it is discovered step-by-step from visual evidence and tool outputs.

---

### Available Tools (Strict API with Return Formats)

**1. `DR_Grading(image_path)`**
- *Function:* Classifies Diabetic Retinopathy severity using multiple models (DRAC, DEEPDR, EYEQ).
- *Trigger:* Use when the question asks for the **stage, severity, or grade** of DR.
- *Return Format:*
```python
{
  "status": "success",
  "result": {
    "image_path": "/path/to/image.jpg",
    "model_type": "MKCNet",
    "results": {
      "DRAC": {
        "diagnosis": {
          "predicted_class": 1,
          "class_label": "中度病变",
          "probabilities": [
            {"class": 0, "label": "无明显视网膜病变或轻微病变", "probability": 0.308},
            {"class": 1, "label": "中度病变", "probability": 0.379},
            {"class": 2, "label": "严重病变", "probability": 0.312}
          ]
        },
        "image_quality": {
          "predicted_class": 0,
          "class_label": "高质量",
          "probabilities": [...]
        }
      },
      "DEEPDR": {
        "diagnosis": {...},
        "image_quality": {...}
      },
      "EYEQ": {
        "diagnosis": {...},
        "image_quality": {...}
      }
    }
  }
}
```

**2. `detect_lesions(image_path, output_path)`**
- *Function:* Returns counts and area ratios of lesions (Hemorrhages, Microaneurysms, Exudates, Cotton Wool Spots).
- *Trigger:* Use when the question asks for **quantitative details** (e.g., "How many microaneurysms?", "Is there significant hard exudate coverage?").
- *Return Format:*
```python
{
  "status": "success",
  "result": {
    "counts": {
      "Cotton Wool Spots": 0,
      "Exudates": 0,
      "Hemorrhages": 3,
      "Microaneurysms": 0
    },
    "save_path": "/path/to/lesions.jpg"
  }
}
```

**3. `autoMorphProcess(image_path)`**
- *Function:* Glaucoma analysis, CDR (Cup-to-Disc Ratio), vessel density, image quality. Returns multiple stages (M0/M1, M2, Quantitative Measurement).
- *Trigger:* Use for **Glaucoma** screening, **optic disc/cup** metrics, **vessel** tortuosity/density, or **image quality** checks.
- *Return Format (Multiple Steps):*
```python
# Step 1: Image Quality (M0/M1)
{
  "status": "success",
  "result": {
    "label": 1,
    "assessment": "Good Quality",
    "indicators": {
      "Name": "test1.png",
      "softmax_good": 0.658,
      "softmax_usable": 0.319,
      "softmax_bad": 0.022,
      "Prediction": 0
    }
  }
}

# Step 2: Segmentation (M2)
{
  "status": "success",
  "result": {
    "artery_vein": "/path/to/artery_vein.png",
    "binary_vessel": "/path/to/binary_vessel.png",
    "optic_disc_cup": "/path/to/optic_disc_cup.png"
  }
}

# Step 3: Quantitative Measurement
{
  "status": "success",
  "result": {
    "macular_features": {
      "Disc_height": 572.61,
      "Disc_width": 532.04,
      "Cup_height": 275.04,
      "Cup_width": 266.02,
      "CDR_vertical": 0.48,
      "CDR_horizontal": 0.5,
      "Vessel_density": 0.053,
      "Fractal_dimension": 1.36,
      "Distance_tortuosity": 4.65,
      "Tortuosity_density": 0.743,
      # ... many more vessel metrics
    },
    "disc_features": {
      "error": "Image 'test1' not found in Disc_Features.csv"  # or actual disc features
    }
  }
}
```

**4. `AMD_predict_fundus_by_deepseenet(image_path)`**
- *Function:* Age-related Macular Degeneration analysis, Drusen size, Pigment abnormalities, AREDS score (0-5).
- *Trigger:* Use for **AMD** diagnosis, **Drusen** analysis, or **Macular** health.
- *Return Format:*
```python
{
  "status": "success",
  "result": {
    "image_path": "/path/to/image.jpg",
    "simplified_score": {
      "score": 5,
      "description": "AREDS简化严重程度评分 (0-5分)，根据玻璃膜疣大小、色素异常和晚期AMD情况综合计算"
    },
    "results": {
      "drusen": {
        "score": 2,
        "label": "Large",
        "confidence": 0.497,
        "probabilities": [0.217, 0.286, 0.497],
        "description": "玻璃膜疣大小 (Drusen Size)"
      },
      "pigment": {
        "score": 1,
        "label": "Yes",
        "confidence": 0.881,
        "probabilities": [0.119, 0.881],
        "description": "色素异常 (Pigmentary Abnormality)"
      },
      "amd": {
        "score": 1,
        "label": "Yes",
        "confidence": 0.561,
        "probabilities": [0.439, 0.561],
        "description": "晚期年龄相关性黄斑变性 (Advanced AMD)"
      },
      "ga": {
        "score": 0,
        "label": "No",
        "confidence": 0.907,
        "probabilities": [0.907, 0.093],
        "description": "地理萎缩 (Geographic Atrophy)"
      },
      "cga": {
        "score": 0,
        "label": "No",
        "confidence": 0.997,
        "probabilities": [0.997, 0.003],
        "description": "中央地理萎缩 (Central GA)"
      }
    }
  }
}
```

**5. `segment_by_ddcs(image_path, output_path)`**
- *Function:* Visual semantic segmentation (returns maps, not just numbers).
- *Trigger:* Use ONLY when the question explicitly asks about the **location** or **morphology** of structures/lesions (e.g., "Where are the lesions located?", "Outline the optic disc"). If it asks "How many", use `detect_lesions` instead.
- *Return Format:* (Returns segmentation file paths, similar structure to autoMorphProcess M2)

**6. `rag_query(query)`**
- *Function:* Retrieves medical guidelines or definitions.
- *Trigger:* Use when the question is **theoretical** or **requires external knowledge** not visible in the image (e.g., "What is the recommended treatment for stage 3 DR?").
- *Return Format:*
```python
{
  "retrieved_context": "For Severe NPDR (4-2-1 rule met), the risk of progression to PDR is high. Management includes: 1. Optimization of glucose and blood pressure control. 2. Close follow-up every 2-4 months. 3. Consider Panretinal Photocoagulation (PRP) if patient compliance is poor or high-risk features are imminent."
}
```

---

### Trajectory Format (Strict XML-like Structure)

**Response Ordering Rules (CRITICAL)**

1. Your response must be **pure XML text**—no markdown fences, no leading/trailing commentary.
2. The VERY FIRST character must start the `<think>` tag. Do **not** emit spaces, line breaks, or text before this tag, and do **not** call any tool before producing a complete `<think>...</think>` block.
3. Every tool invocation must be wrapped inside `<tool_call>...</tool_call>` and immediately followed by its `<observation>...</observation>`.
4. The `<tool_call>` block may only contain the literal `Tool:` and `Args:` lines. All narrative analysis must stay inside `<think>`.
5. Before each new tool call, emit another `<think>` describing the latest interpretation and plan.
6. End the entire trajectory with exactly one `<answer>...</answer>` block summarizing the evidence, and place all explanatory text in previous `<think>` blocks (the `<answer>` should directly state the final answer).

If you violate this ordering (for example, by outputting `<tool_call>` before `<think>`), the response is invalid.

You must output the trajectory in this exact format. Each step follows the pattern:

```xml
<think>
(Visual Analysis): Describe what "you see" in the image that matches the Ground Truth.
(Hypothesis): Formulate a medical hypothesis.
(Plan): Decide the first tool to verify the main diagnosis.
</think>

<tool_call>
Tool: <Tool Name>
Args: <JSON dictionary of arguments>
</tool_call>

<observation>
<Simulated JSON output matching Ground Truth. CRITICAL: The values here must logically result in the Ground Truth Answer.>
</observation>

<think>
(Analysis): Interpret the previous result.
(Refinement): Do I need more detail? (e.g., do I need lesion counts now? Do I need knowledge?)
(Plan): Decide the next tool.
</think>

<tool_call>
Tool: <Tool Name>
Args: <JSON dictionary of arguments>
</tool_call>

<observation>
<Simulated JSON output>
</observation>

... (repeat as needed for multiple tool calls) ...

<answer>
(Final Conclusion): Synthesize Visuals + Tool Data + RAG Knowledge into the final answer. Never mention ground truth or that you are reverse-engineering.
</answer>
```

**CRITICAL:** 
- Always start with `<think>` that includes visual analysis
- Use multiple `<tool_call>`-`<observation>` pairs when needed
- End with a single `<answer>` that synthesizes all evidence
- Never mention "ground truth" or "reverse-engineering" in your output

---

### Few-Shot Example (Complex DR Case)

**Input Data:**
* **Question:** "Evaluate the severity of diabetic retinopathy in this patient, quantify the hemorrhages, and suggest the standard management for this stage."
* **Ground Truth:** "The patient has Severe Non-Proliferative Diabetic Retinopathy (NPDR) with significant hemorrhage burden (over 15 counts). Standard management involves strict glycemic control and consideration for panretinal photocoagulation if high-risk features develop."

**Generated Trajectory:**

```xml
<think>
**Visual Analysis:** Upon examining the fundus image, I clearly observe multiple deep, dark red dot-and-blot hemorrhages scattered across all four quadrants. There are also visible venous beading and some cotton-wool spots.
**Hypothesis:** These visual features are strongly characteristic of severe Diabetic Retinopathy (DR), likely at the Non-Proliferative stage (NPDR).
**Plan:** To confirm this visual assessment, I must first accurately classify the disease severity using the grading model.
**Decision:** Call `DR_Grading`.
</think>

<tool_call>
Tool: DR_Grading
Args: {"image_path": "test_img.jpg"}
</tool_call>

<observation>
{
  "status": "success",
  "result": {
    "image_path": "test_img.jpg",
    "model_type": "MKCNet",
    "results": {
      "DRAC": {
        "diagnosis": {
          "predicted_class": 2,
          "class_label": "严重病变",
          "probabilities": [
            {"class": 0, "label": "无明显视网膜病变或轻微病变", "probability": 0.1},
            {"class": 1, "label": "中度病变", "probability": 0.15},
            {"class": 2, "label": "严重病变", "probability": 0.75}
          ]
        }
      },
      "DEEPDR": {
        "diagnosis": {
          "predicted_class": 2,
          "class_label": "严重病变",
          "probabilities": [
            {"class": 0, "label": "无明显视网膜病变或轻微病变", "probability": 0.05},
            {"class": 1, "label": "中度病变", "probability": 0.20},
            {"class": 2, "label": "严重病变", "probability": 0.75}
          ]
        }
      },
      "EYEQ": {
        "diagnosis": {
          "predicted_class": 2,
          "class_label": "严重病变",
          "probabilities": [
            {"class": 0, "label": "无明显视网膜病变或轻微病变", "probability": 0.02},
            {"class": 1, "label": "中度病变", "probability": 0.18},
            {"class": 2, "label": "严重病变", "probability": 0.80}
          ]
        }
      }
    }
  }
}
</observation>

<think>
**Analysis:** The `DR_Grading` tool confirms the diagnosis of "Severe NPDR" with high confidence across multiple models (DRAC, DEEPDR, EYEQ all predict class 2 - severe disease).
**Refinement:** The user specifically asked to *quantify* the hemorrhages. The grading tool only gives a classification, not a count. Therefore, specific lesion detection is required to answer the second part of the user's request.
**Decision:** Call `detect_lesions` to count the hemorrhages.
</think>

<tool_call>
Tool: detect_lesions
Args: {"image_path": "test_img.jpg", "output_path": "results/"}
</tool_call>

<observation>
{
  "status": "success",
  "result": {
    "counts": {
      "Hemorrhages": 18,
      "Microaneurysms": 12,
      "Exudates": 5,
      "Cotton Wool Spots": 2
    },
    "save_path": "results/lesions.jpg"
  }
}
</observation>

<think>
**Analysis:** The lesion detection reveals 18 hemorrhages, which is a significant burden, consistent with the severe diagnosis.
**Refinement:** Now I have the diagnosis (Severe NPDR) and the quantification (18 hemorrhages). The final part of the user's request is about "standard management". I should not guess this; I should retrieve authoritative medical guidelines for Severe NPDR.
**Decision:** Call `rag_query` to find management guidelines.
</think>

<tool_call>
Tool: rag_query
Args: {"query": "Standard management guidelines for Severe Non-Proliferative Diabetic Retinopathy"}
</tool_call>

<observation>
{
  "retrieved_context": "For Severe NPDR (4-2-1 rule met), the risk of progression to PDR is high. Management includes: 1. Optimization of glucose and blood pressure control. 2. Close follow-up every 2-4 months. 3. Consider Panretinal Photocoagulation (PRP) if patient compliance is poor or high-risk features are imminent."
}
</observation>

<answer>
Based on the **visual evidence** of diffuse hemorrhages and the **`DR_Grading` result** of Severe NPDR (Class 2) confirmed by multiple models, the diagnosis is established. The **`detect_lesions` tool** quantified the burden, revealing 18 hemorrhages, which supports the severity classification. Finally, the **`rag_query`** provided the clinical management protocol. Therefore, the patient has Severe NPDR with a high hemorrhage count (18). Management requires strict metabolic control and potential laser therapy (PRP) based on follow-up stability and high-risk feature development.
</answer>
```

---

Always ensure your trajectory follows the XML format strictly: `<think>`, `<tool_call>`, `<observation>`, and finally `<answer>`. Generate realistic tool outputs that match the actual API return formats provided above. The first tag must always be `<think>`, and no markdown fences (like ```xml) are allowed."""

    USER_PROMPT_TEMPLATE = """### Task Instructions

**Input:**
* **Question:** {question}
* **Image Path:** {image_path}
* **Ground Truth Answer:** {ground_truth_answer}
* **Image (base64):** {image_base64}

**Instruction:**
Generate the full trajectory now using the strict XML format **and ordering rules**:
- Start with `<think>` that includes visual analysis (describe what you "see" in the image)
- Use multiple `<tool_call>`-`<observation>` pairs when needed (grading + quantification + knowledge)
- Use `rag_query` whenever treatment/prognosis/guideline knowledge is required
- Observations must be realistic JSON matching the actual tool return formats provided in the system prompt
- Never mention "ground truth" or that you are reverse-engineering; present reasoning as if discovered step-by-step from visual evidence and tool outputs
- End with a single `<answer>` that states the final selection/diagnosis succinctly (explanations should already appear in earlier `<think>` blocks)

**CRITICAL ORDERING RULES:**
- The very first character you output must start `<think>`; calling a tool before thinking (or emitting markdown fences) is forbidden.
- The `<tool_call>` block must contain only the `Tool:` and `Args:` lines (no narrative). Every `<tool_call>` must be immediately followed by its `<observation>`.
- Before each new tool call, emit another `<think>` describing the updated analysis/refinement/plan.
- Follow the exact XML format: `<think></think><tool_call></tool_call><observation></observation>` (repeat as needed) `<answer></answer>`.
- Ensure the Observation data matches the Ground Truth perfectly, but present it as if derived from the tools. The `<answer>` should directly respond to the user's question without extra explanation (those belong in `<think>`)."""

    def __init__(
        self,
        api_client: APIClient,
        max_workers: int = 5,
        retry_times: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化CoT生成器
        
        Args:
            api_client: API客户端实例
            max_workers: 最大并发数
            retry_times: 重试次数
            retry_delay: 重试延迟（秒）
        """
        self.api_client = api_client
        self.max_workers = max_workers
        self.retry_times = retry_times
        self.retry_delay = retry_delay

    def generate_trajectory(
        self,
        question: str,
        image_path: str,
        ground_truth_answer: str
    ) -> str:
        """
        为单个样本生成推理轨迹
        
        Args:
            question: 问题
            image_path: 图片路径
            ground_truth_answer: 标准答案
            
        Returns:
            生成的推理轨迹文本
        """
        system_prompt = self.SYSTEM_PROMPT

        image_base64 = self._encode_image_base64(image_path)

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            question=question,
            image_path=image_path,
            ground_truth_answer=ground_truth_answer,
            image_base64=image_base64 or "[image read failed]"
        )
        
        for attempt in range(self.retry_times):
            try:
                response = self.api_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                # 可根据需要启用额外校验；默认直接返回原始响应
                # processed = self._post_process_response(response)
                # return processed
                return response
            except Exception as e:
                logger.warning(f"生成失败 (尝试 {attempt + 1}/{self.retry_times}): {e}")
                if attempt < self.retry_times - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        
        return ""

    def process_single_example(self, example: CoTExample) -> CoTExample:
        """
        处理单个示例
        
        Args:
            example: CoT示例
            
        Returns:
            处理后的示例（包含生成的轨迹）
        """
        try:
            trajectory = self.generate_trajectory(
                question=example.question,
                image_path=example.image_path,
                ground_truth_answer=example.ground_truth_answer
            )
            example.trajectory = trajectory
        except Exception as e:
            logger.error(f"处理示例失败: {e}")
            example.error = str(e)
        
        return example

    def batch_generate(
        self,
        examples: List[CoTExample],
        output_path: Optional[str] = None,
        save_interval: int = 10
    ) -> List[CoTExample]:
        """
        批量生成CoT数据
        
        Args:
            examples: 示例列表
            output_path: 输出文件路径（可选）
            save_interval: 保存间隔（每处理N个样本保存一次）
            
        Returns:
            处理后的示例列表
        """
        results = []
        total = len(examples)
        
        logger.info(f"开始批量生成，共 {total} 个样本")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_example = {
                executor.submit(self.process_single_example, example): example
                for example in examples
            }
            
            # 使用tqdm显示进度
            with tqdm(total=total, desc="生成进度") as pbar:
                for future in as_completed(future_to_example):
                    example = future_to_example[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # 定期保存
                        if len(results) % save_interval == 0 and output_path:
                            self._save_results(results, output_path)
                        
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"处理示例时出错: {e}")
                        example.error = str(e)
                        results.append(example)
                        pbar.update(1)
        
        # 最终保存
        if output_path:
            self._save_results(results, output_path)
        
        # 统计信息
        success_count = sum(1 for r in results if r.trajectory is not None)
        error_count = sum(1 for r in results if r.error is not None)
        
        logger.info(f"生成完成: 成功 {success_count}/{total}, 失败 {error_count}/{total}")
        
        return results

    def _save_results(self, results: List[CoTExample], output_path: str):
        """
        保存结果到文件
        
        Args:
            results: 结果列表
            output_path: 输出路径
        """
        output_data = []
        for result in results:
            output_data.append({
                "question": result.question,
                "image_path": result.image_path,
                "ground_truth_answer": result.ground_truth_answer,
                "trajectory": result.trajectory,
                "error": result.error
            })
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_path}")

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """移除可能的markdown代码块包裹"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # 去掉起始 ```lang
            cleaned = cleaned[3:]
            newline_idx = cleaned.find("\n")
            if newline_idx != -1:
                cleaned = cleaned[newline_idx + 1 :]
            # 去掉结尾 ```
            end_idx = cleaned.rfind("```")
            if end_idx != -1:
                cleaned = cleaned[:end_idx]
            cleaned = cleaned.strip()
        return cleaned

    @staticmethod
    def _encode_image_base64(image_path: str) -> Optional[str]:
        """读取图片并转换为base64字符串；失败返回None"""
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"图片不存在或路径无效: {image_path}")
            return None
        try:
            import base64
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            logger.warning(f"图片编码为base64失败: {e}")
            return None

    def _post_process_response(self, raw_response: str) -> str:
        """
        清理并验证大模型返回的轨迹，确保满足XML格式要求
        """
        if not raw_response or not raw_response.strip():
            raise ValueError("空响应")

        cleaned = self._strip_markdown_fences(raw_response)
        cleaned = cleaned.lstrip("\ufeff").strip()

        think_idx = cleaned.find("<think>")
        if think_idx == -1:
            raise ValueError("响应缺少 <think> 标签")
        if think_idx > 0:
            logger.debug("检测到 <think> 前存在额外内容，已自动裁剪")
            cleaned = cleaned[think_idx:]

        if not cleaned.startswith("<think>"):
            raise ValueError("响应未以 <think> 开头")

        if "<answer>" not in cleaned or "</answer>" not in cleaned:
            raise ValueError("响应缺少 <answer> 块")

        tool_calls = cleaned.count("<tool_call>")
        if tool_calls != cleaned.count("</tool_call>"):
            raise ValueError("tool_call 标签不匹配")

        observations = cleaned.count("<observation>")
        if observations != cleaned.count("</observation>") or observations != tool_calls:
            raise ValueError("observation 标签不匹配或与 tool_call 数量不一致")

        return cleaned


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CoT数据生成器")
    parser.add_argument("--dataset", type=str, required=True, help="VQA数据集路径（JSON格式）")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型名称（预设: gpt-4o, gpt-4, gpt-3.5-turbo, claude-3.5-sonnet, qwen-plus, qwen-max, qwen-72b；或自定义模型名。如果未指定，从.env文件的MODEL_NAME读取）")
    parser.add_argument("--api-key", type=str, default=None, help="API密钥（如果未指定，从.env文件的API_KEY读取）")
    parser.add_argument("--base-url", type=str, default=None, help="API基础URL（如果未指定，从.env文件的BASE_URL读取）")
    parser.add_argument("--image-root", type=str, default=None, help="图片根目录（用于解析数据集中相对路径）")
    parser.add_argument("--max-workers", type=int, default=5, help="最大并发数")
    parser.add_argument("--batch-size", type=int, help="批处理大小（可选，默认处理全部）")
    parser.add_argument("--start-idx", type=int, default=0, help="起始索引")
    parser.add_argument("--end-idx", type=int, help="结束索引（可选）")
    
    args = parser.parse_args()
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    loader = DatasetLoader()
    examples = loader.load_dataset(args.dataset, image_root=args.image_root)
    
    # 切片处理
    if args.end_idx:
        examples = examples[args.start_idx:args.end_idx]
    elif args.start_idx > 0:
        examples = examples[args.start_idx:]
    
    if args.batch_size:
        examples = examples[:args.batch_size]
    
    logger.info(f"共加载 {len(examples)} 个样本")
    
    # 创建API客户端
    api_client = APIClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # 创建生成器
    generator = CoTGenerator(
        api_client=api_client,
        max_workers=args.max_workers
    )
    
    # 批量生成
    results = generator.batch_generate(
        examples=examples,
        output_path=args.output,
        save_interval=10
    )
    
    logger.info("所有任务完成！")


if __name__ == "__main__":
    main()

