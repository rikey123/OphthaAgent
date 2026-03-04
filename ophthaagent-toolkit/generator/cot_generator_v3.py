"""
CoT生成器 V2 - 真实工具集成版本
支持5类推理场景，调用真实工具获取结果
"""
import json
import os
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import asdict

from api_client import APIClient
from dataset_loader import DatasetLoader, CoTExample
from real_tool_executor import RealToolExecutor
from image_processor import ImageProcessor
from reasoning_classifier import ReasoningTypeClassifier, ReasoningType
from reasoning_strategy import ReasoningStrategy
from ssh_image_fetcher import SSHImageFetcher

logger = logging.getLogger(__name__)


class CoTGeneratorV2:
    """CoT生成器 V2 - 真实工具集成版"""

    BASE_SYSTEM_PROMPT = """### System Prompt For Generating Deep-Reasoning Medical Trajectories with REAL Tool Results

**Role Definition**
You are an advanced Ophthalmic Agent Simulator. Your goal is to generate a **multi-step, visually-grounded reasoning trajectory** that solves a medical Question based on REAL tool execution results and a known Ground Truth Answer.

**CRITICAL REQUIREMENT:**
- You are provided with **PRE-EXECUTED REAL TOOL RESULTS** for reference
- You do NOT need to copy the full tool results into your response
- Your role is to construct the REASONING PROCESS that naturally leads to calling these tools

**Core Requirements for "Thinking" (The Brain)**

1. **Visual Hallucination:** Start by describing specific visual features matching the ground truth (e.g., "I observe scattered dot-blot hemorrhages")

2. **Tool Selection Strategy:** Choose tools based on:
   - Visual observations
   - Disease suspected
   - Need for quantification
   - Need for verification

3. **Observation Placeholder:** When you call a tool, you MUST:
   - Write `<observation>[TOOL_RESULT_PLACEHOLDER:tool_name]</observation>`
   - Do NOT copy the full JSON result (we will insert it automatically)
   - The placeholder will be replaced with the real tool execution result

4. **Evidence-Based Reasoning:** Every tool call must be justified by:
   - Visual observation
   - Previous tool results
   - Medical knowledge

---

### Trajectory Format (Strict XML Structure) - MANDATORY

**IMPORTANT: You MUST generate pure XML without any markdown code fences (no ```xml or ```)**

**Response Ordering Rules (CRITICAL)**

1. Your response must be **pure XML text**—no markdown fences, no code blocks
2. Start with `<think>` (visual analysis + hypothesis + plan)
3. Every `<tool_call>` must be immediately followed by `<observation>`
4. Before each new tool call, emit another `<think>` describing analysis/refinement/plan
5. **END WITH EXACTLY ONE `<answer>` BLOCK - THIS IS MANDATORY**

**COMPLETION REQUIREMENT:**
- Your response is INVALID and will be REJECTED if it does not contain a closing `<answer>` tag
- Do NOT stop generating until you have written `</answer>`
- Even if the JSON in `<observation>` is very long, you MUST continue to the final `<answer>` block

**XML Structure Example:**

<think>
(Visual Analysis): I observe the fundus image showing multiple microaneurysms and dot-blot hemorrhages scattered in the posterior pole, particularly around the macula. The retinal vessels appear mildly tortuous.
(Hypothesis): These findings are consistent with diabetic retinopathy. The presence of microaneurysms and hemorrhages without neovascularization suggests non-proliferative DR.
(Plan): I will first use DR_Grading to quantify the severity level.
</think>

<tool_call>
Tool: DR_Grading
Args: {{"image_path": "path/to/image.jpg"}}
</tool_call>

<observation>[TOOL_RESULT_PLACEHOLDER:DR_Grading]</observation>

<think>
(Analysis): The DR_Grading tool confirms Moderate NPDR with multiple microaneurysms and hemorrhages. This validates my visual assessment.
(Refinement): To complete the diagnosis, I should check for macular edema risk.
(Plan): Use dme_risk_assess to evaluate DME risk based on lesion proximity to the fovea.
</think>

<tool_call>
Tool: dme_risk_assess
Args: {{"image_path": "path/to/image.jpg"}}
</tool_call>

<observation>[TOOL_RESULT_PLACEHOLDER:dme_risk_assess]</observation>

<answer>
Based on the comprehensive analysis: The fundus image shows Moderate Non-Proliferative Diabetic Retinopathy (NPDR) with multiple microaneurysms and hemorrhages confirmed by DR grading. Additionally, the DME risk assessment indicates moderate risk with hard exudates detected near the fovea. The final diagnosis is Moderate NPDR with moderate DME risk.
</answer>

**Available Tools:**
- DR_Grading: Diabetic retinopathy severity classification (DINO-based model with 5-class grading: 正常/轻度/中度/重度/增生性)
- detect_lesions: Count lesions (hemorrhages, microaneurysms, exudates, cotton wool spots)
- DR_segmentation_anchor_ETDRS: Uses a pre-existing lesion segmentation map to automatically generate a standard ETDRS grid, anchored to the fovea/optic disc, for quantifying lesion spatial distribution.
- AMD_predict: Age-related macular degeneration analysis
- segment_by_AutoMorphalyzer: Glaucoma analysis (CDR, vessel density)
- dme_risk_assess: Diabetic macular edema risk assessment
- quality_assess: Image quality evaluation
- enhance_image: Image enhancement
- crop_roi: Crop region of interest with optional super-resolution
- fovea_od_localize: Locate fovea and optic disc
- vessel_segment: Vessel segmentation
- rag_query: Medical knowledge retrieval

---

### CRITICAL: Never Mention Ground Truth
- Present reasoning as if discovered step-by-step from evidence
- Never say "to match the ground truth" or "reverse-engineering"
- Act as if you are making a genuine diagnosis

---

### FINAL REMINDER: ALWAYS COMPLETE THE <answer> TAG
Your trajectory is WORTHLESS and will be REJECTED if you do not end with:

<answer>
(Your final medical conclusion based on all observations)
</answer>

DO NOT STOP GENERATION until you have written the complete <answer> block with closing tag.
This is a HARD REQUIREMENT. Incomplete trajectories will be discarded.

---
"""

    def __init__(
        self,
        api_client: APIClient,
        tool_executor: RealToolExecutor,
        image_processor: ImageProcessor = None,
        ssh_fetcher: SSHImageFetcher = None,
        max_workers: int = 3,
        retry_times: int = 3,
        retry_delay: float = 1.0,
        allowed_types: Optional[List[str]] = None,
        disable_ssh: bool = False,
        skip_healthy: bool = False
    ):
        """
        初始化生成器

        Args:
            api_client: API客户端
            tool_executor: 真实工具执行器
            image_processor: 图像处理器
            ssh_fetcher: SSH远程图片获取器
            max_workers: 最大并发数
            retry_times: 重试次数
            retry_delay: 重试延迟
            allowed_types: 允许生成的推理类型列表（如 ["A", "B"]），None表示允许所有类型
            disable_ssh: 禁用SSH，使用本地路径模式（默认: False）
            skip_healthy: 跳过A-healthy类型的API调用，节省token（默认: False）
        """
        self.api_client = api_client
        self.tool_executor = tool_executor
        self.tools_interface = tool_executor.tools if hasattr(tool_executor, 'tools') else None  # 获取工具接口
        self.image_processor = image_processor or ImageProcessor()
        self.ssh_fetcher = ssh_fetcher or SSHImageFetcher()  # 默认创建SSH获取器
        self.max_workers = max_workers
        self.retry_times = retry_times
        self.disable_ssh = disable_ssh  # 保存SSH禁用标志
        self.retry_delay = retry_delay
        self.allowed_types = allowed_types  # 新增：类型过滤器
        self.skip_healthy = skip_healthy  # 新增：跳过healthy类型

        self.classifier = ReasoningTypeClassifier()

    def generate_trajectory_with_real_tools(
        self,
        question: str,
        image_path: str,
        ground_truth_answer: str
    ) -> Dict:
        """
        使用真实工具生成推理轨迹

        Returns:
            dict: {
                "trajectory": str,
                "metadata": dict,
                "tool_execution_log": list,
                "generation_attempts": list,  # 新增：所有生成尝试记录
                "error": str or None
            }
        """
        try:
            # Step 1: 分类推理类型（基于标签的初步分类）
            logger.info(f"开始生成轨迹: {image_path}")
            reasoning_type = self.classifier.classify(question, ground_truth_answer)
            ground_truth_answer = ReasoningTypeClassifier._extract_answer_from_question(question, ground_truth_answer)

            # 检查是否成功分类
            if reasoning_type is None:
                logger.info(f"⚠️ 样本无法分类，跳过生成: {image_path}")
                return {
                    "trajectory": None,
                    "metadata": {
                        "reasoning_type": None,
                        "reasoning_subtype": None,
                        "confidence": 0.0,
                        "skipped": True,
                        "skip_reason": "分类器无法识别该样本的推理类型"
                    },
                    "tool_execution_log": None,
                    "generation_attempts": [],
                    "error": None
                }

            logger.info(f"初步推理类型: {reasoning_type.type}-{reasoning_type.subtype} (置信度: {reasoning_type.confidence:.2f})")

            # 保存初始推理类型，用于后续记录调整信息
            initial_reasoning_type = reasoning_type

            # 类型过滤（前置）：对于非动态类型（A/C/D），在初始分类后就可以过滤
            # 注意：Type B和Type E是动态类型，只有运行工具后才能确定，因此不在此处过滤
            if self.allowed_types is not None and reasoning_type.type not in self.allowed_types:
                # 特殊处理：如果过滤器包含动态类型（B或E），且当前样本可能动态调整，则不跳过
                might_become_type_b = self._might_become_type_b(reasoning_type, ground_truth_answer)
                print(f"might_become_type_b: {might_become_type_b}")
                might_become_type_e = self._might_become_type_e()

                if "B" in self.allowed_types and might_become_type_b:
                    logger.info(f"初步类型为 {reasoning_type.type}，但可能动态调整为Type B，继续执行")
                elif "E" in self.allowed_types and might_become_type_e:
                    logger.info(f"初步类型为 {reasoning_type.type}，但可能动态调整为Type E，继续执行")
                else:
                    logger.info(f"跳过非目标类型: {reasoning_type.type} (允许类型: {self.allowed_types})")
                    return {
                        "trajectory": None,
                        "metadata": {
                            "reasoning_type": reasoning_type.type,
                            "reasoning_subtype": reasoning_type.subtype,
                            "confidence": reasoning_type.confidence,
                            "skipped": True,
                            "skip_reason": f"类型过滤：仅生成 {self.allowed_types} 类型"
                        },
                        "tool_execution_log": None,
                        "generation_attempts": [],
                        "error": None
                    }

            # # 跳过纯DR案例的生成
            # if reasoning_type.type == "A" and reasoning_type.subtype == "pure_dr":
            #     logger.info(f"跳过纯DR案例生成: {image_path}")
            #     return {
            #         "trajectory": None,
            #         "metadata": {
            #             "reasoning_type": "A",
            #             "reasoning_subtype": "pure_dr",
            #             "confidence": reasoning_type.confidence,
            #             "skipped": True,
            #             "skip_reason": "纯DR案例，按用户要求跳过生成"
            #         },
            #         "tool_execution_log": None,
            #         "generation_attempts": [],
            #         "error": None
            #     }

            # Step 2: 获取推理策略
            strategy = ReasoningStrategy.get_strategy(reasoning_type)

            # Step 2.1: Type B专用策略覆盖
            # 如果过滤器为Type B，且样本可能成为Type B，使用包含纠错检测工具的特殊策略
            if self.allowed_types is not None and "B" in self.allowed_types:
                might_become_type_b = self._might_become_type_b(reasoning_type, ground_truth_answer)
                if might_become_type_b:
                    logger.info("Type B过滤模式：使用纠错检测策略")
                    strategy = self._get_type_b_detection_strategy(reasoning_type, ground_truth_answer, strategy)

            logger.info(f"工具序列: {strategy['tool_sequence']}")

            # Step 2.5: 质量检测优先 - 先运行质量评估
            logger.info("Step 2.5: 优先运行质量评估...")
            quality_result, quality_exec_log = self._run_quality_check(image_path)

            # 初始化工具结果和执行日志
            tool_results = {}
            execution_log = []

            if quality_result:
                tool_results["quality_assess"] = quality_result
                execution_log.append(quality_exec_log)
                logger.info(f"质量评估完成: {quality_result.get('result', {})}")

            # Step 2.6: 基于质量评估结果可能提前调整为 Type E
            # ⚠️ 已注释：批量模式下不主动生成 Type E，只有在 --type-filter E 时才生成
            # 注意：只有在允许Type E时才进行质量调整（避免非Type E过滤时意外转换）
            # quality_adjusted_type = None
            # quality_score = self._extract_quality_score(tool_results)

            # if quality_score is not None:
            #     logger.info(f"质量评分: {quality_score:.2f}")

            #     # 检查是否允许Type E调整
            #     allow_type_e = (self.allowed_types is None or "E" in self.allowed_types)

            #     if allow_type_e:
            #         # 质量极差 → Type E: low_quality
            #         if quality_score < 0.3:
            #             logger.info(f"质量极差 ({quality_score:.2f} < 0.3) → 调整为 E-low_quality (直接拒诊)")
            #             quality_adjusted_type = ReasoningType(
            #                 type="E",
            #                 subtype="low_quality",
            #                 confidence=0.95,
            #                 description="拒诊/异常: low_quality (质量极差)"
            #             )
            #             reasoning_type = quality_adjusted_type
            #             strategy = ReasoningStrategy.get_strategy(reasoning_type)
            #             logger.info(f"已调整为拒诊模式，工具序列: {strategy['tool_sequence']}")

            #         # 质量较差 → Type E: enhance_then_diagnose
            #         elif quality_score < 0.5:
            #             logger.info(f"质量较差 ({quality_score:.2f} < 0.5) → 调整为 E-enhance_then_diagnose (增强后诊断)")
            #             quality_adjusted_type = ReasoningType(
            #                 type="E",
            #                 subtype="enhance_then_diagnose",
            #                 confidence=0.90,
            #                 description="拒诊/异常: enhance_then_diagnose (质量较差需增强)"
            #             )
            #             reasoning_type = quality_adjusted_type
            #             strategy = ReasoningStrategy.get_strategy(reasoning_type)
            #             logger.info(f"已调整为增强模式，工具序列: {strategy['tool_sequence']}")

            #         else:
            #             logger.info(f"质量良好 ({quality_score:.2f} ≥ 0.5) → 继续正常诊断流程")
            #     else:
            #         # 不允许Type E调整，但记录质量信息
            #         if quality_score < 0.5:
            #             logger.info(f"质量较差 ({quality_score:.2f} < 0.5)，但未启用Type E过滤，继续正常诊断流程")
            #         else:
            #             logger.info(f"质量良好 ({quality_score:.2f} ≥ 0.5) → 继续正常诊断流程")

            # 记录质量评分到日志（供参考，不影响类型）
            quality_score = self._extract_quality_score(tool_results)
            if quality_score is not None:
                logger.info(f"质量评分: {quality_score:.2f} (仅供参考，不影响类型分类)")

            # Step 3: 预执行剩余工具获取真实结果
            additional_results, additional_log = self._pre_execute_tools(image_path, strategy)
            tool_results.update(additional_results)
            execution_log.extend(additional_log)

            # Step 4: 基于工具结果动态调整推理类型（Type B/C/D等）
            try:
                adjusted_type, adjustment_info = self._analyze_tool_results_and_adjust_strategy(
                    reasoning_type,
                    tool_results,
                    ground_truth_answer
                )
            except Exception as e:
                logger.warning(f"动态调整失败，跳过调整: {e}", exc_info=True)
                adjusted_type = None
                adjustment_info = {}

            # 如果发生了调整，更新策略并补充执行缺失工具
            if adjusted_type:
                logger.info(f"动态调整推理类型: {reasoning_type.type}-{reasoning_type.subtype} → {adjusted_type.type}-{adjusted_type.subtype}")
                logger.info(f"调整原因: {adjustment_info.get('reason', 'N/A')}")
                reasoning_type = adjusted_type
                strategy = ReasoningStrategy.get_strategy(reasoning_type)
                logger.info(f"更新后工具序列: {strategy['tool_sequence']}")

                # Step 4.5: 补充执行新策略需要但尚未执行的工具
                new_tool_sequence = strategy.get("tool_sequence", [])
                already_executed = set(tool_results.keys())
                missing_tools = [tool for tool in new_tool_sequence if tool not in already_executed]

                if missing_tools:
                    logger.info(f"补充执行缺失工具: {missing_tools}")
                    additional_results, additional_log = self._pre_execute_tools(image_path, {"tool_sequence": missing_tools})
                    tool_results.update(additional_results)
                    execution_log.extend(additional_log)
                    logger.info(f"✓ 补充执行完成，总工具数: {len(tool_results)}")
                else:
                    logger.info(f"⊙ 所有需要的工具已执行，无需补充")

            # 类型过滤（后置）：在动态调整后，检查最终类型是否匹配过滤器
            if self.allowed_types is not None and reasoning_type.type not in self.allowed_types:
                logger.info(f"动态调整后类型不匹配: {reasoning_type.type} (允许类型: {self.allowed_types})")
                return {
                    "trajectory": None,
                    "metadata": {
                        "reasoning_type": reasoning_type.type,
                        "reasoning_subtype": reasoning_type.subtype,
                        "confidence": reasoning_type.confidence,
                        "skipped": True,
                        "skip_reason": f"动态调整后类型过滤：仅生成 {self.allowed_types} 类型"
                    },
                    "tool_execution_log": execution_log,  # 保留工具执行日志，记录已执行的工具
                    "generation_attempts": [],
                    "error": None
                }

            # Step 4.9: 跳过 A-healthy 类型的API调用（如果启用）
            # 注意：只有在动态调整后仍然是 A-healthy 才跳过，确保不会误跳已升级为 Type B 的样本
            if self.skip_healthy and reasoning_type.type == "A" and reasoning_type.subtype == "healthy":
                logger.info(f"✓ 检测到 A-healthy 类型，已启用 skip_healthy，跳过API调用")
                return {
                    "trajectory": None,
                    "metadata": {
                        "reasoning_type": reasoning_type.type,
                        "reasoning_subtype": reasoning_type.subtype,
                        "confidence": reasoning_type.confidence,
                        "skipped": True,
                        "skip_reason": "A-healthy类型，已生成足够数据，跳过以节省token",
                        "tools_executed": list(tool_results.keys())
                    },
                    "tool_execution_log": execution_log,
                    "generation_attempts": [],
                    "error": None
                }

            # Step 5: 编码图片
            image_data = self._prepare_image(image_path, tool_results)

            # Step 6: 构建Prompts
            system_prompt = self._build_system_prompt(reasoning_type, strategy)
            user_prompt = self._build_user_prompt(
                question,
                image_path,
                ground_truth_answer,
                image_data,
                tool_results,
                strategy
            )

            # Step 7: 准备图像列表（用于多模态输入）
            images = []
            
            # 添加原始图像的缩略图
            if "thumbnail" in image_data and image_data["thumbnail"]:
                images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data['thumbnail']}"
                    }
                })
                logger.info("✓ 添加原始图像缩略图到多模态输入")
            
            # 添加标注图
            if "annotated_image" in image_data and "thumbnail" in image_data["annotated_image"]:
                annotated_thumbnail = image_data["annotated_image"]["thumbnail"]
                if annotated_thumbnail:
                    images.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{annotated_thumbnail}"
                        }
                    })
                    logger.info("✓ 添加标注图到多模态输入")

            # Step 8: 调用LVLM生成轨迹（带占位符）
            logger.info("调用LVLM生成轨迹...")
            generation_attempts = []
            try:
                trajectory_with_placeholders = self._generate_with_retry(
                    system_prompt, 
                    user_prompt,
                    tool_sequence=strategy.get("tool_sequence", []),
                    images=images
                )
            except RuntimeError as e:
                # 捕获包含所有尝试记录的异常
                if hasattr(e, 'all_attempts'):
                    generation_attempts = e.all_attempts
                raise

            # Step 9: 替换占位符为真实工具结果
            logger.info("替换工具占位符为真实结果...")
            trajectory = self._replace_tool_placeholders(trajectory_with_placeholders, tool_results)

            # Step 9: 返回结果（包含调整信息）
            metadata = {
                "reasoning_type": reasoning_type.type,
                "reasoning_subtype": reasoning_type.subtype,
                "confidence": reasoning_type.confidence,
                "strategy": strategy,
                "tools_executed": list(tool_results.keys())
            }

            # 如果发生了调整，记录调整信息
            if adjusted_type:
                metadata["adjustment"] = {
                    "initial_type": f"{initial_reasoning_type.type}-{initial_reasoning_type.subtype}",
                    "adjusted_type": f"{adjusted_type.type}-{adjusted_type.subtype}",
                    "reason": adjustment_info.get("reason", ""),
                    "evidence": adjustment_info.get("evidence", {})
                }

            return {
                "trajectory": trajectory,
                "trajectory_with_placeholders": trajectory_with_placeholders,  # 保留原始带占位符版本用于调试
                "metadata": metadata,
                "tool_execution_log": execution_log,
                "generation_attempts": generation_attempts,  # 记录所有生成尝试
                "error": None
            }

        except Exception as e:
            logger.error(f"生成轨迹失败: {e}")
            # 提取尝试记录（如果有）
            generation_attempts = []
            if hasattr(e, 'all_attempts'):
                generation_attempts = e.all_attempts

            return {
                "trajectory": None,
                "metadata": None,
                "tool_execution_log": None,
                "generation_attempts": generation_attempts,  # 即使失败也记录尝试
                "error": str(e)
            }

    def _analyze_tool_results_and_adjust_strategy(
        self,
        initial_type: ReasoningType,
        tool_results: Dict,
        ground_truth_answer: str
    ) -> tuple:
        """
        基于工具执行结果动态调整推理类型
        """
        
        # 1. 检查工具结果与GT的冲突 → Type B (纠错工具错误)
        gt_lower = ground_truth_answer.lower()

        # pure_dr这个分类，DR分级不准确 → Type B (DR分级错误)
        if initial_type.type == "A" and initial_type.subtype == "pure_dr":
            # 场景1: GT提到DR整体诊断，但工具分级与GT不符 → B-dr_grading_mismatch
            if any(kw in gt_lower for kw in ["diabetic retinopathy"]):
                has_dr_grade = self._has_dr_grade(tool_results)
                print(f"工具DR分级结果：{has_dr_grade}")
                if has_dr_grade and has_dr_grade not in gt_lower:
                    logger.info("工具DR分级 → 调整为 B-dr_grading_mismatch")
                    return (
                        ReasoningType(
                            type="B",
                            subtype="dr_grading_mismatch",
                            confidence=0.92,
                            description="纠错: DR分级工具误判，需要进一步深入分析"
                        ),
                        {
                            "reason": "DR分级工具结果不可靠，需要进一步收集病灶量化和空间分布证据",
                            "evidence": {
                                "ground_truth": ground_truth_answer,
                                "tool_grade": has_dr_grade
                            }
                        }
                    )

        # 1.1 GT显示AMD，但工具误判为DR → 需要纠错 (False DR is AMD), 应该是从pure_seg转过来的
        if any(kw in gt_lower for kw in ["amd","drusen", "age-related macular", "age related macular", "macular degeneration"]):
            has_dr_features = self._has_dr_features(tool_results)
            if has_dr_features:
                logger.info("检测到工具误判DR，实际GT=AMD → 调整为 B-false_dr_is_amd")
                return (
                    ReasoningType(
                        type="B",
                        subtype="false_dr_is_amd",
                        confidence=0.88,
                        description="纠错: 工具误判DR硬性渗出，实为AMD玻璃膜疣"
                    ),
                    {
                        "reason": "GT标签为AMD，但DR病灶检测工具误报硬性渗出，需要纠正为玻璃膜疣",
                        "evidence": {
                            "ground_truth": ground_truth_answer,
                            "tool_false_positive": has_dr_features
                        }
                    }
                )

        # 1.2 GT显示健康/生理性大视杯，但CDR偏大 → 需要纠错 (False Glaucoma)
        if any(kw in gt_lower for kw in ["healthy", "normal", "physiologic", "生理性", "正常"]):
            has_borderline_cdr = self._has_borderline_cdr(tool_results)
            if has_borderline_cdr:
                logger.info("检测到CDR边缘值，但GT=健康 → 调整为 B-false_glaucoma")
                return (
                    ReasoningType(
                        type="B",
                        subtype="false_glaucoma",
                        confidence=0.85,
                        description="纠错: CDR偏大但实为生理性大视杯"
                    ),
                    {
                        "reason": "GT标签为健康，但CDR计算显示边缘值，需要验证是否为生理性大视杯",
                        "evidence": {
                            "ground_truth": ground_truth_answer,
                            "cdr_metrics": has_borderline_cdr
                        }
                    }
                )

        # 2.3 GT显示健康，但工具检测到病灶 → 需要纠错 (Artifact)
        if any(kw in gt_lower for kw in ["healthy", "normal", "no abnormal", "健康", "正常"]):
            has_lesion_detection = self._has_lesion_detection(tool_results)
            if has_lesion_detection:
                logger.info("检测到病灶，但GT=健康 → 调整为 B-artifact")
                return (
                    ReasoningType(
                        type="B",
                        subtype="artifact",
                        confidence=0.80,
                        description="纠错: 工具检测到病灶，但可能是伪影"
                    ),
                    {
                        "reason": "GT标签为健康，但病灶检测工具报告异常，需要验证是否为伪影",
                        "evidence": {
                            "ground_truth": ground_truth_answer,
                            "detected_lesions": has_lesion_detection
                        }
                    }
                )

        # 2.4 工具低置信度 → 需要Crop验证 (Low Confidence Recheck)
        low_confidence_tool = self._find_low_confidence_tool(tool_results)
        if low_confidence_tool:
            logger.info(f"检测到低置信度工具: {low_confidence_tool['tool_name']} → 调整为 B-low_confidence_recheck")
            return (
                ReasoningType(
                    type="B",
                    subtype="low_confidence_recheck",
                    confidence=0.75,
                    description="纠错: 工具置信度低，需要Crop验证"
                ),
                {
                    "reason": "工具返回低置信度结果，需要高清切片验证",
                    "evidence": low_confidence_tool
                }
            )

        # 2.5 检查是否有DR且需要评估DME → Type C (级联)
        if any(kw in gt_lower for kw in ["diabetic retinopathy","hard exudates","exudates"]):
            has_dme_signs = self._has_dme_signs(tool_results)
            if has_dme_signs:
                logger.info("检测到DR伴随DME风险 → 调整为 C-dr_to_dme")
                return (
                    ReasoningType(
                        type="C",
                        subtype="dr_to_dme",
                        confidence=0.85,
                        description="级联推理: dr_to_dme"
                    ),
                    {
                        "reason": "DR患者检测到黄斑水肿风险，需要级联评估",
                        "evidence": {
                            "dme_signs": has_dme_signs
                        }
                    }
                )

        # 3. 无需调整
        return (None, {})

    def _extract_quality_score(self, tool_results: Dict) -> Optional[float]:
        """从工具结果中提取图像质量分数"""
        if not tool_results or not isinstance(tool_results, dict):
            logger.debug("tool_results is None or not a dict")
            return None

        if "quality_assess" not in tool_results:
            return None

        quality_result = tool_results.get("quality_assess")
        if not quality_result or quality_result.get("status") != "success":
            return None

        result_data = quality_result.get("result", {})
        if not isinstance(result_data, dict):
            return None

        # 尝试多种可能的键名（包括fundus_image_toolbox使用的"confs"字段）
        for key in ["quality_score", "score", "overall_quality", "quality", "confs"]:
            if key in result_data:
                score = result_data[key]
                if isinstance(score, (int, float)):
                    return float(score)

        return None

    def _has_dr_features(self, tool_results: Dict) -> Optional[Dict]:
        """检查工具结果中是否有DR特征 (可能是误判)

        ✅ 2026-01-18 更新：大幅提高阈值，避免对GT=AMD样本误触发Type B
        """

        # 检查病灶检测工具（UNet格式）
        if "detect_lesions" in tool_results:
            lesion_result = tool_results["detect_lesions"]
            if lesion_result.get("status") == "success":
                result_data = lesion_result.get("result", {})

                # ✅ UNet格式：检查量化分析指标
                metrics = result_data.get("量化分析指标", {})

                # ✅ 大幅提高阈值，确保GT=AMD能正常生成pure_amd轨迹
                # 只有在DR特征非常明显时才触发Type B
                has_dr_lesions = (
                    metrics.get("硬渗出物区域占比", 0) >= 0.03 or     # ✅ 硬渗出物 >= 5%（提高50倍）
                    metrics.get("微动脉瘤数量", 0) >= 10 or              # ✅ 微动脉瘤 >= 10个（提高10倍）
                    metrics.get("出血区域占比", 0) >= 0.05 or          # ✅ 出血 >= 5%（提高5倍）
                    metrics.get("软渗出物区域占比", 0) >= 0.01          # ✅ 软渗出物 >= 1%（提高10倍）
                )

                if has_dr_lesions:
                    return {
                        "source": "detect_lesions",
                        "硬渗出物区域占比": metrics.get("硬渗出物区域占比", 0),
                        "微动脉瘤数量": metrics.get("微动脉瘤数量", 0),
                        "出血区域占比": metrics.get("出血区域占比", 0),
                        "软渗出物区域占比": metrics.get("软渗出物区域占比", 0)
                    }

        # 检查DR分级工具（DINO格式）
        if "DR_Grading" in tool_results:
            dr_result = tool_results["DR_Grading"]
            if dr_result.get("status") == "success":
                result_data = dr_result.get("result", {})
                prediction = result_data.get("prediction", {})

                if prediction:
                    class_name = prediction.get("class_name", "").lower()
                    class_id = prediction.get("class_id", -1)

                    # 检查是否为DR（排除正常类别0）
                    if class_id > 0 and ("视网膜病变" in class_name or
                                        "轻度" in class_name or "中度" in class_name or
                                        "重度" in class_name or "增生性" in class_name):
                        return {
                            "source": "DR_Grading",
                            "grade": prediction.get("class_name", ""),
                            "predicted_class": class_id,
                            "confidence": prediction.get("confidence", 0)
                        }

        return None

    def _has_dr_grade(self, tool_results: Dict) -> Optional[str]:
        """从DR_Grading工具结果中提取并映射DR分级"""
        if "DR_Grading" not in tool_results:
            return None

        dr_result = tool_results["DR_Grading"]
        if dr_result.get("status") != "success":
            return None

        result_data = dr_result.get("result", {})
        prediction = result_data.get("prediction", {})
        
        if not prediction:
            return None
            
        class_name = str(prediction.get("class_id",-1))

        # 定义映射关系
        grade_mapping = {
            "1": "mild diabetic retinopathy",
            "2": "moderate diabetic retinopathy",
            "3": "severe diabetic retinopathy",
            "4": "proliferative diabetic retinopathy"
        }

        for key, value in grade_mapping.items():
            if key in class_name:
                print("value:",value)
                return value

        return None

    def _has_amd_features(self, tool_results: Dict) -> Optional[Dict]:
        """检查是否有AMD特征（drusen）"""

        # 检查AMD预测工具
        if "AMD_predict" in tool_results:
            amd_result = tool_results["AMD_predict"]
            if amd_result.get("status") == "success":
                result_data = amd_result.get("result", {})

                # 检查整体AMD判断
                results = result_data.get("results", {})
                amd = results.get("amd", {})
                amd_label = amd.get("label", "")

                # 检查是否检测到AMD
                if amd_label == "Yes":
                    return {
                        "source": "AMD_predict",
                        "has_amd": True,
                        "amd_confidence": amd.get("confidence", 0)
                    }

                # 检查是否检测到drusen
                drusen = results.get("drusen", {})
                drusen_label = drusen.get("label", "")
                if drusen_label == "Yes":
                    return {
                        "source": "AMD_predict",
                        "has_amd": True,
                        "drusen_detected": True,
                        "drusen_confidence": drusen.get("confidence", 0)
                    }

        # 检查病灶检测工具中是否提到drusen
        if "detect_lesions" in tool_results:
            lesion_result = tool_results["detect_lesions"]
            if lesion_result.get("status") == "success":
                result_data = lesion_result.get("result", {})

                # ✅ UNet格式检查
                analysis_results = result_data.get("分析结果", {})

                # 检查是否有drusen相关的分割任务（UNet通常不分割drusen，但保留兼容性）
                for task_name in analysis_results.keys():
                    if "drusen" in task_name.lower() or "玻璃膜疣" in task_name:
                        task_result = analysis_results[task_name]
                        status = task_result.get("状态", "")
                        if "完成" in status:
                            metrics = task_result.get("指标", {})
                            # UNet格式可能有"病灶数量"或"区域占比"
                            count = metrics.get("病灶数量", 0)
                            ratio = metrics.get("区域占比", metrics.get("玻璃膜疣区域占比", 0))
                            if count > 0 or ratio > 0:
                                return {
                                    "source": "detect_lesions",
                                    "drusen_count": count,
                                    "drusen_ratio": ratio
                                }

        return None

    def _has_dme_signs(self, tool_results: Dict) -> Optional[Dict]:
        """检查是否有DME风险迹象"""
        if "dme_risk_assess" in tool_results:
            dme_result = tool_results["dme_risk_assess"]
            if dme_result.get("status") == "success":
                result_data = dme_result.get("result", {})
                
                # Check for dme_risk_assessment nested dictionary (new format)
                risk_assessment = result_data.get("dme_risk_assessment", {})
                print("risk_assessment🤔:",risk_assessment)
                if not risk_assessment:
                    # Fallback to direct keys (old format)
                    risk_assessment = result_data
                
                # 检查风险级别
                risk_level = risk_assessment.get("risk_level", -1)
                
                # risk_level could be int (0, 1, 2) or string ("moderate", etc.)
                is_risky = False
                if isinstance(risk_level, int) and risk_level >= 1:
                    is_risky = True
                elif isinstance(risk_level, str) and risk_level.lower() in ["moderate", "high", "中等", "高"]:
                    is_risky = True
                if is_risky:
                    return {
                        "source": "dme_risk_assess",
                        "risk_level": risk_level,
                        "risk_category": risk_assessment.get("risk_category"),
                        "exudates_near_fovea": risk_assessment.get("description") # Map description to exudates info
                    }

        # 检查病灶检测中的硬渗出物
        if "detect_lesions" in tool_results:
            lesion_result = tool_results["detect_lesions"]
            if lesion_result.get("status") == "success":
                result_data = lesion_result.get("result", {})
                if "分析任务" in result_data:
                    # 检查硬渗出物
                    he_task = result_data["分析任务"].get("硬渗出物分割", {})
                    if he_task.get("状态") == "成功":
                        metrics = he_task.get("指标", {})
                        if metrics.get("病灶数量", 0) > 3:  # 硬渗出物较多
                            return {
                                "source": "detect_lesions",
                                "hard_exudates_count": metrics.get("病灶数量")
                            }

        return None

    def _has_borderline_cdr(self, tool_results: Dict) -> Optional[Dict]:
        """检查CDR是否处于边缘值 (0.5-0.6)"""

        if "segment_by_AutoMorphalyzer" not in tool_results:
            return None

        segment_result = tool_results["segment_by_AutoMorphalyzer"]
        if segment_result.get("status") != "success":
            return None

        result_data = segment_result.get("result", {})
        images = result_data.get("images", [])

        if not images or len(images) == 0:
            return None

        optic_disc = images[0].get("optic_disc", {})
        cdr_vertical = optic_disc.get("cdr_vertical")
        cdr_horizontal = optic_disc.get("cdr_horizontal")

        if cdr_vertical is None or cdr_horizontal is None:
            return None

        # 取垂直和水平CDR的平均值
        cdr_avg = (float(cdr_vertical) + float(cdr_horizontal)) / 2

        # 边缘值: 0.5-0.6
        if 0.5 <= cdr_avg <= 0.6:
            return {
                "cdr_vertical": cdr_vertical,
                "cdr_horizontal": cdr_horizontal,
                "cdr_average": cdr_avg
            }

        return None

    def _has_lesion_detection(self, tool_results: Dict) -> Optional[Dict]:
        """检查是否检测到病灶 (可能是伪影)"""

        if "detect_lesions" not in tool_results:
            return None

        lesion_result = tool_results["detect_lesions"]
        if lesion_result.get("status") != "success":
            return None

        result_data = lesion_result.get("result", {})
        metrics = result_data.get("量化分析指标", {})

        # 统计各类病灶
        microaneurysms = metrics.get("微动脉瘤数量", 0)
        hemorrhages_ratio = metrics.get("出血区域占比", 0)
        hemorrhages_size = metrics.get("出血区域总大小", 0)
        hard_exudates_ratio = metrics.get("硬渗出物区域占比", 0)
        soft_exudates_ratio = metrics.get("软渗出物区域占比", 0)

        # ✅ 大幅提高阈值（避免将轻微伪影误判为真实病灶）
        # 参考 TYPE_B_JUDGMENT_ANALYSIS.md 的建议，采用非常保守的阈值
        has_lesions = (
            microaneurysms >= 3 or                      # 至少10个微动脉瘤（排除单点噪声）
            hemorrhages_ratio >= 0.02 or                 # 至少5%出血占比
            hard_exudates_ratio >= 0.01 or               # 至少5%硬渗出物占比
            soft_exudates_ratio >= 0.01                  # 至少1%软渗出物占比
        )

        if has_lesions:
            return {
                "微动脉瘤数量": microaneurysms,
                "出血区域占比": hemorrhages_ratio,
                "出血区域总大小": hemorrhages_size,
                "硬渗出物区域占比": hard_exudates_ratio,
                "软渗出物区域占比": soft_exudates_ratio
            }

        return None

    def _find_low_confidence_tool(self, tool_results: Dict) -> Optional[Dict]:
        """查找低置信度工具结果 (< 0.4)"""

        confidence_threshold = 0.6

        # 检查AMD_predict
        if "AMD_predict" in tool_results:
            amd_result = tool_results["AMD_predict"]
            if amd_result.get("status") == "success":
                result_data = amd_result.get("result", {})
                results = result_data.get("results", {})

                # 只检查整体AMD判断的置信度（而不是组成成分如drusen、pigment的置信度）
                # 根据用户要求：使用整体评分的置信度，而不是其组成成分的置信度
                amd = results.get("amd", {})
                amd_conf = amd.get("confidence", 1.0)
                amd_label = amd.get("label", "")

                # 只有当AMD判断为阳性（Yes）且置信度低时才触发
                if amd_label == "Yes" and amd_conf < confidence_threshold:
                    return {
                        "tool_name": "AMD_predict",
                        "feature": "amd",
                        "confidence": amd_conf,
                        "threshold": confidence_threshold,
                        "label": amd_label
                    }

        # 检查DR_Grading
        if "DR_Grading" in tool_results:
            dr_result = tool_results["DR_Grading"]
            if dr_result.get("status") == "success":
                result_data = dr_result.get("result", {})
                results = result_data.get("results", {})

                # 检查DEEPDR置信度
                deepdr = results.get("DEEPDR", {}).get("diagnosis", {})
                probabilities = deepdr.get("probabilities", [])

                if probabilities:
                    predicted_class = deepdr.get("predicted_class", 0)
                    if 0 <= predicted_class < len(probabilities):
                        prob_entry = probabilities[predicted_class]
                        conf = prob_entry.get("probability", 1.0) if isinstance(prob_entry, dict) else 1.0

                        if conf < confidence_threshold:
                            return {
                                "tool_name": "DR_Grading",
                                "confidence": conf,
                                "threshold": confidence_threshold,
                                "label": deepdr.get("class_label", "")
                            }

        return None

    def _might_become_type_b(self, reasoning_type: ReasoningType, ground_truth_answer: str) -> bool:
        """
        预测样本是否可能动态调整为Type B

        Type B是纠错场景，只有在运行工具后发现工具结果与GT冲突时才会触发。
        此方法基于GT特征预测哪些样本可能触发Type B。

        Args:
            reasoning_type: 初始推理类型
            ground_truth_answer: GT答案

        Returns:
            bool: 是否可能成为Type B
        """
        gt_lower = ground_truth_answer.lower()

        if any(kw in gt_lower for kw in ["diabetic retinopathy"]):
            if reasoning_type.type == "A":
                return True

        # 场景1: GT=AMD，可能误判为DR（false_dr_is_amd）
        # 如果GT包含AMD相关关键词，且初始分类为Type A
        if any(kw in gt_lower for kw in ["amd","drusen", "age-related macular", "age related macular", "macular degeneration"]):
            if reasoning_type.type == "A":
                return True

        # 场景2: GT=健康，可能误判为Glaucoma（false_glaucoma）
        # 如果GT包含健康相关关键词，且初始分类为Type A
        if any(kw in gt_lower for kw in ["healthy", "normal", "physiologic", "生理性", "正常"]):
            if reasoning_type.type == "A":
                return True

        # 场景3: GT=健康，可能误判为有病灶（artifact）
        # 同场景2

        # 场景4: 任何低置信度场景（low_confidence_recheck）
        # 无法在此预测，因为需要工具结果

        return False

    def _might_become_type_e(self) -> bool:
        """
        预测样本是否可能动态调整为Type E

        Type E触发条件：图像质量差（需要运行quality_assess才能确定）
        由于Type E是质量触发的，任何样本都有可能成为Type E。

        Type E的特殊性：
        - 任何图像都可能因为质量差而成为Type E
        - 质量检测在Step 2.5运行，此时尚未知道质量分数
        - 因此无法基于初始分类预测，所有样本都需要运行质量评估

        Returns:
            bool: 是否可能成为Type E（始终返回True）
        """
        # Type E的特殊性：任何样本都可能因为图像质量差而成为Type E
        # 因此当使用--type-filter E时，应该对所有样本运行质量检测
        #
        # 注意：这意味着--type-filter E会导致所有样本都执行质量评估
        # 这是必要的，因为只有运行质量评估后才能确定是否为Type E
        return True

    def _get_type_b_detection_strategy(self, reasoning_type: ReasoningType, ground_truth_answer: str, original_strategy: Dict) -> Dict:
        """
        为可能成为Type B的样本生成纠错检测策略（最小化版本）

        在Type B过滤模式下，使用最小化检测工具集来触发Type B：
        - GT=AMD: [detect_lesions, AMD_predict] (检测DR误判)
        - GT=Healthy: [detect_lesions, segment_by_AutoMorphalyzer] (检测假病灶/假青光眼)

        注意：这是检测策略，用于触发Type B。触发后会在Step 4.5补充执行纠错所需的工具。

        根据Type B设计（2阶段执行）：
        - 假DR是中/重度:
            阶段1(检测): DR_Grading
            阶段2(纠错): 触发后补充segment_by_unet + DR_segmentation_anchor_ETDRS

        - 假DR是AMD:
          阶段1(检测): detect_lesions + AMD_predict
          阶段2(纠错): 触发后补充 fovea_od_localize + crop_roi

        - 假青光眼:
          阶段1(检测): segment_by_AutoMorphalyzer
          阶段2(纠错): 触发后补充 fovea_od_localize + crop_roi

        - 假病灶:
          阶段1(检测): detect_lesions
          阶段2(纠错): 触发后补充 fovea_od_localize + crop_roi

        - 低置信度确认:
          阶段1(检测): 原工具
          阶段2(纠错): 触发后补充 fovea_od_localize + crop_roi

        Args:
            reasoning_type: 初始推理类型
            ground_truth_answer: GT答案
            original_strategy: 原始策略

        Returns:
            Dict: 修改后的策略（Type B专用最小检测工具集）
        """
        gt_lower = ground_truth_answer.lower()

        # 场景0: DR_pure → Type B dr_grading_mismatch检测策略（最小化）
        strategy = original_strategy.copy()
        if any(kw in gt_lower for kw in ["diabetic retinopathy"]):
            tool_sequence = ["DR_Grading"]
            logger.info("  Type B-DR Grading检测策略: [DR_Grading]" )

        # 场景1: GT=AMD → Type B检测策略（最小化）
        elif any(kw in gt_lower for kw in ["amd","drusen", "age-related macular", "age related macular", "macular degeneration"]):
            # Type B检测策略：只需要检测DR误判的工具
            # 1. detect_lesions (检测DR特征，判断是否误报)
            # 2. AMD_predict (确认AMD诊断)
            # 注意：不需要fovea_od_localize和crop_roi，触发B-false_dr_is_amd后会在Step 4.5补充
            tool_sequence = ["detect_lesions", "AMD_predict"]
            logger.info("  Type B-AMD检测策略: [detect_lesions, AMD_predict] (触发后补充fovea+crop)")

        # 场景2: GT=Healthy → Type B检测策略（最小化）
        elif any(kw in gt_lower for kw in ["healthy", "normal", "physiologic", "生理性", "正常"]):
            # Type B检测策略：只需要检测假病灶和假青光眼的工具
            # 1. detect_lesions (检测假病灶，判断是否伪影)
            # 2. segment_by_AutoMorphalyzer (计算CDR，判断是否边缘值)
            # 注意：不需要fovea_od_localize和crop_roi，触发后会在Step 4.5补充
            tool_sequence = ["detect_lesions", "segment_by_AutoMorphalyzer"]
            logger.info("  Type B-Healthy检测策略: [detect_lesions, segment_by_AutoMorphalyzer] (触发后补充fovea+crop)")

        else:
            # 其他GT类型：保持原工具序列
            tool_sequence = list(strategy.get("tool_sequence", []))

        # 更新策略
        strategy["tool_sequence"] = tool_sequence
        strategy["max_steps"] = len(tool_sequence) + 5  # 增加步骤数（考虑触发后补充工具）
        strategy["enable_correction"] = True  # 启用纠错模式

        return strategy

    def _run_quality_check(self, image_path: str) -> tuple:
        """
        运行质量检测工具

        Args:
            image_path: 图像路径

        Returns:
            (quality_result, execution_log): 质量检测结果和执行日志
        """
        try:
            # 如果是相对路径，根据SSH设置处理
            if not os.path.isabs(image_path):
                if not self.disable_ssh:
                    # 使用SSH获取本地路径
                    try:
                        logger.info(f"质量检测前SSH获取图片: {image_path}")
                        local_path = self.ssh_fetcher.get_local_path(image_path)
                        logger.info(f"✓ SSH获取成功，使用本地路径: {local_path}")
                        image_path = local_path
                    except Exception as e:
                        logger.warning(f"SSH获取失败，尝试使用原路径: {e}")
                else:
                    # SSH已禁用，使用服务器上的绝对路径
                    remote_path = os.path.join(self.ssh_fetcher.remote_base_path, image_path)
                    logger.info(f"SSH已禁用，质量检测使用服务器路径: {remote_path}")
                    image_path = remote_path

            start_time = time.time()
            logger.info("运行 quality_assess...")

            # 调用质量评估工具（实际方法名为quality_assess_by_fit）
            result = self.tools_interface.quality_assess_by_fit(image_path)

            execution_time = time.time() - start_time

            # 构建执行日志
            execution_log = {
                "tool_name": "quality_assess",
                "args": {"image_path": image_path},
                "result": result,
                "execution_time": round(execution_time, 2),
                "status": result.get("status", "unknown")
            }

            logger.info(f"✓ quality_assess 完成 (耗时 {execution_time:.2f}s)")

            return result, execution_log

        except Exception as e:
            logger.error(f"✗ quality_assess 失败: {e}")
            execution_log = {
                "tool_name": "quality_assess",
                "args": {"image_path": image_path},
                "result": {"status": "error", "error": str(e)},
                "execution_time": 0,
                "status": "error"
            }
            return None, execution_log

    def _pre_execute_tools(self, image_path: str, strategy: Dict) -> tuple:
        """
        预先执行所有可能需要的工具 (支持SSH远程图片获取)

        Returns:
            (tool_results, execution_log): 工具结果字典和执行日志
        """
        # 如果是相对路径，根据SSH设置处理
        if not os.path.isabs(image_path):
            if not self.disable_ssh:
                # 使用SSH获取本地路径
                try:
                    logger.info(f"工具执行前SSH获取图片: {image_path}")
                    local_path = self.ssh_fetcher.get_local_path(image_path)
                    logger.info(f"✓ SSH获取成功，使用本地路径: {local_path}")
                    image_path = local_path
                except Exception as e:
                    logger.warning(f"SSH获取失败，尝试使用原路径: {e}")
            else:
                # SSH已禁用，使用服务器上的绝对路径
                remote_path = os.path.join(self.ssh_fetcher.remote_base_path, image_path)
                logger.info(f"SSH已禁用，使用服务器路径: {remote_path}")
                image_path = remote_path

        tool_sequence = strategy.get("tool_sequence", [])

        # 解析依赖
        tool_sequence = ReasoningStrategy.resolve_tool_dependencies(tool_sequence)

        logger.info(f"预执行 {len(tool_sequence)} 个工具...")

        tool_results = {}
        execution_log = []

        for tool_name in tool_sequence:
            # 跳过已经执行过的quality_assess (在Step 2.5中已执行)
            if tool_name == "quality_assess":
                logger.info(f"⊙ {tool_name} 已在Step 2.5执行，跳过")
                continue

            # ✅ 初始化 args 变量（防止在 _build_tool_args 失败时出现 UnboundLocalError）
            args = None
            execution_time = 0

            try:
                start_time = time.time()

                # 构建工具参数
                args = self._build_tool_args(tool_name, image_path, tool_results)

                # 执行工具
                result = self.tool_executor.execute_tool(tool_name, args)

                execution_time = time.time() - start_time

                # 记录结果
                tool_results[tool_name] = result
                # ✅ 改进路径提取逻辑（支持目录输出）
                processed_image_path = None
                if result.get("status") == "success":
                    result_data = result.get("result", {})

                    # ✅ 1. 优先检查 segment_by_unet 的 "结果目录" 字段
                    if "结果目录" in result_data and result_data["结果目录"]:
                        output_dir = result_data["结果目录"]
                        # 从目录中查找图片文件
                        if os.path.isdir(output_dir):
                            # 优先返回综合分割图
                            combined_path = os.path.join(output_dir, "dr_combined_analysis.png")
                            if os.path.exists(combined_path):
                                processed_image_path = combined_path
                            else:
                                # 如果综合图不存在，查找任意图片文件
                                images = self._find_images_in_dir(output_dir)
                                if images:
                                    processed_image_path = images[0]

                    # ✅ 2. 如果没找到，检查其他常见的图片路径字段
                    if not processed_image_path:
                        for key in ["cropped_image_path", "output_image_path", "processed_image_path",
                                   "save_path", "image_path", "enhanced_image_path"]:
                            if key in result_data and result_data[key]:
                                path = result_data[key]
                                # 如果路径是目录，从其中查找图片
                                if os.path.isdir(path):
                                    images = self._find_images_in_dir(path)
                                    if images:
                                        processed_image_path = images[0]
                                else:
                                    processed_image_path = path
                                break

                execution_log.append({
                    "tool_name": tool_name,
                    "args": args,
                    "result": result,
                    "processed_image_path": processed_image_path,
                    "execution_time": round(execution_time, 2),
                    "status": result.get("status", "unknown")
                })

                # ✅ 验证路径可访问性
                if processed_image_path:
                    if not os.path.exists(processed_image_path):
                        logger.warning(f"⚠️ {tool_name} 输出路径不存在: {processed_image_path}")
                    else:
                        logger.info(f"✓ {tool_name} 输出路径可访问: {processed_image_path}")

                logger.info(f"✓ {tool_name} 执行成功 ({execution_time:.2f}s)")

            except Exception as e:
                logger.warning(f"✗ {tool_name} 执行失败: {e}")
                error_result = {"status": "error", "error": str(e)}
                tool_results[tool_name] = error_result
                execution_log.append({
                    "tool_name": tool_name,
                    "args": args,
                    "result": error_result,
                    "processed_image_path": None,  # 错误情况无处理后的图片
                    "execution_time": 0,
                    "status": "error"
                })

        return tool_results, execution_log

    def _find_images_in_dir(self, directory: str) -> list:
        """
        在目录中查找图片文件

        Args:
            directory: 目录路径

        Returns:
            list: 图片文件路径列表（按文件名排序）
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        try:
            if os.path.isdir(directory):
                files = os.listdir(directory)
                image_files = [
                    os.path.join(directory, f)
                    for f in files
                    if os.path.splitext(f)[1].lower() in image_extensions
                ]
                return sorted(image_files)  # 按文件名排序
        except Exception as e:
            logger.warning(f"⚠️ 无法读取目录 {directory}: {e}")
        return []

    def _generate_unique_output_path(self, tool_name: str, input_path: str) -> str:
        """
        生成唯一的输出路径（避免多次执行时覆盖）

        Args:
            tool_name: 工具名称
            input_path: 输入文件路径

        Returns:
            str: 唯一的输出路径
        """
        import time
        import uuid

        base_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1]

        # 生成唯一标识符
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # 使用8位UUID

        # 根据工具类型确定后缀和是否使用目录
        tool_configs = {
            "enhance_image": {"suffix": "_enhanced", "use_dir": False},
            "enhance_fundus_image": {"suffix": "_enhanced", "use_dir": False},
            "segment_by_unet": {"suffix": "_lesions", "use_dir": True},
            "detect_lesions": {"suffix": "_lesions", "use_dir": True},
            "segment_by_AutoMorphalyzer": {"suffix": "_glaucoma", "use_dir": True},  # AutoMorphalyzer
            "DR_segmentation_anchor_ETDRS": {"suffix": "_DR_segmentation_anchor_ETDRS", "use_dir": True},
            "crop_roi": {"suffix": "_crop", "use_dir": True},  # ✅ 修改：使用目录，让 crop_roi 在其中自由命名
            "fovea_od_localize": {"suffix": "_fovea_od", "use_dir": True},  # ✅ 添加 fovea_od_localize 配置
        }

        config = tool_configs.get(tool_name, {"suffix": f"_{tool_name}", "use_dir": False})

        # ✅ 修复：使用工具输出目录而不是数据集目录
        # 从环境变量读取输出目录，默认为 ./tool_outputs
        output_base_dir = os.getenv("TOOL_OUTPUT_DIR", "./tool_outputs")
        os.makedirs(output_base_dir, exist_ok=True)  # 确保输出目录存在

        if config["use_dir"]:
            # 创建结果目录（UNet工具）
            result_dir = os.path.join(output_base_dir, f"{base_name}{config['suffix']}_{timestamp}_{unique_id}")
            os.makedirs(result_dir, exist_ok=True)
            return result_dir
        else:
            # 生成文件路径（enhance_image等）
            output_filename = f"{base_name}{config['suffix']}_{timestamp}_{unique_id}{ext}"
            return os.path.join(output_base_dir, output_filename)

    def _build_tool_args(self, tool_name: str, image_path: str, prev_results: Dict) -> Dict:
        """根据工具类型构建参数"""
        base_args = {"image_path": image_path}

        # ✅ 为需要输出路径的工具生成唯一路径
        tools_requiring_unique_path = {
            "enhance_image", "enhance_fundus_image",
            "segment_by_unet", "detect_lesions",  # UNet工具
            "segment_by_AutoMorphalyzer",  # AutoMorphalyzer（青光眼）
            "DR_segmentation_anchor_ETDRS",
            "fovea_od_localize"  # ✅ 添加 Fovea 定位工具，它需要输出路径参数
        }

        if tool_name in tools_requiring_unique_path:
            unique_path = self._generate_unique_output_path(tool_name, image_path)
            if tool_name == "DR_segmentation_anchor_ETDRS":
                base_args["output_path"] = unique_path
                detect_result = prev_results.get("detect_lesions")
                if detect_result and detect_result.get("status") == "success":
                    result_data = detect_result.get("result", {})
                    lesion_root = result_data.get("结果目录") or result_data.get("lesion_analysis_dir")
                    if lesion_root:
                        lesion_dir = os.path.dirname(lesion_root)
                        base_args["lesion_dir"] = lesion_dir
            elif tool_name.startswith("segment_by") or tool_name in ["detect_lesions", "segment_by_AutoMorphalyzer"]:
                base_args["output_path"] = unique_path
            else:
                base_args["output_path"] = unique_path
            logger.info(f"✓ 生成唯一输出路径: {unique_path}")
            return base_args

        # crop_roi需要bbox
        elif tool_name == "crop_roi":
            # 尝试从之前的结果提取位置信息
            if "fovea_od_localize" in prev_results:
                fovea_result = prev_results["fovea_od_localize"]
                if fovea_result.get("status") == "success":
                    # 尝试多种可能的结果结构
                    result_data = fovea_result.get("result", {})

                    # 结构1: result.coordinates.fovea
                    coordinates = result_data.get("coordinates", {})
                    fovea = coordinates.get("fovea", {})

                    # 结构2: result.fovea (兼容旧格式)
                    if not fovea:
                        fovea = result_data.get("fovea", {})

                    if "x" in fovea and "y" in fovea:
                        # 以黄斑为中心裁剪500x500区域
                        x_center = int(fovea["x"])
                        y_center = int(fovea["y"])
                        base_args["bbox"] = {
                            "x1": max(0, x_center - 250),
                            "y1": max(0, y_center - 250),
                            "x2": x_center + 250,
                            "y2": y_center + 250
                        }
                        base_args["super_resolution"] = True
                        base_args["sr_scale"] = 2

                        # ✅ 添加：生成唯一的输出路径
                        unique_path = self._generate_unique_output_path(tool_name, image_path)
                        base_args["output_path"] = unique_path

                        logger.info(f"使用黄斑中心坐标裁剪: ({x_center}, {y_center})")
                        logger.info(f"✓ 为 crop_roi 生成输出路径: {unique_path}")
                        return base_args

            # 默认裁剪中心区域 (降低默认尺寸以兼容小图像)
            base_args["bbox"] = {"x1": 256, "y1": 256, "x2": 768, "y2": 768}
            base_args["super_resolution"] = False

            # ✅ 添加：生成唯一的输出路径
            unique_path = self._generate_unique_output_path(tool_name, image_path)
            base_args["output_path"] = unique_path

            logger.warning("未找到黄斑坐标，使用默认中心区域裁剪")
            logger.info(f"✓ 为 crop_roi 生成输出路径: {unique_path}")

        # rag_query需要query参数
        elif tool_name == "rag_query":
            # 这里需要根据上下文构建查询
            base_args = {"query": "Standard management for the diagnosed condition"}

        # 其他工具一般只需要image_path
        return base_args

    def _prepare_image(self, image_path: str, tool_results: Optional[Dict] = None) -> Dict:
        """准备图像数据 (支持SSH远程获取)

        Args:
            image_path: 原始图像路径
            tool_results: 工具执行结果，用于提取标注图

        Returns:
            包含原始图像和标注图的字典
        """
        if tool_results is None:
            tool_results = {}

        # 如果是相对路径，根据SSH设置处理
        if not os.path.isabs(image_path):
            if not self.disable_ssh:
                # 使用SSH获取本地路径
                try:
                    logger.info(f"检测到相对路径，尝试通过SSH获取: {image_path}")
                    local_path = self.ssh_fetcher.get_local_path(image_path)
                    logger.info(f"✓ SSH获取成功，本地路径: {local_path}")
                    image_path = local_path
                except Exception as e:
                    logger.warning(f"SSH获取失败，尝试使用原路径: {e}")
            else:
                # SSH已禁用，使用服务器上的绝对路径
                remote_path = os.path.join(self.ssh_fetcher.remote_base_path, image_path)
                logger.info(f"SSH已禁用，使用服务器路径: {remote_path}")
                image_path = remote_path

        # 生成原始图像的缩略图（节省token）
        images = self.image_processor.prepare_images_for_lvlm(
            image_path,
            include_thumbnail=True,
            include_full=False,
            thumbnail_size=768
        )

        # 检查是否有标注图
        annotated_image_path = None
        for tool_name, result_data in tool_results.items():
            if isinstance(result_data, dict) and result_data.get("_is_annotated_image"):
                annotated_image_path = result_data.get("_annotated_image_path")
                logger.info(f"✓ 发现标注图: {annotated_image_path}")
                break

        # 如果有标注图，也进行编码
        if annotated_image_path and os.path.exists(annotated_image_path):
            annotated_image = self.image_processor.prepare_images_for_lvlm(
                annotated_image_path,
                include_thumbnail=True,
                include_full=False,
                thumbnail_size=768
            )
            images["annotated_image"] = annotated_image
            logger.info(f"✓ 标注图编码完成")
        elif annotated_image_path:
            logger.warning(f"标注图路径存在但文件不存在: {annotated_image_path}")

        return images

    def _build_system_prompt(self, reasoning_type: ReasoningType, strategy: Dict) -> str:
        """构建系统提示"""
        # 基础提示
        prompt = self.BASE_SYSTEM_PROMPT

        # 添加类型特定指导
        type_suffix = strategy.get("system_prompt_suffix", "")
        if type_suffix:
            prompt += f"\n\n### SPECIFIC GUIDANCE FOR THIS CASE:\n{type_suffix}"

        return prompt

    def _build_user_prompt(
        self,
        question: str,
        image_path: str,
        ground_truth_answer: str,
        image_data: Dict,
        tool_results: Dict,
        strategy: Dict
    ) -> str:
        """
        构建用户提示

        注意：只向AI展示策略工具序列中需要的工具结果，不展示检测阶段的多余工具。
        例如：B-false_glaucoma策略需要[segment_by_AutoMorphalyzer, fovea_od_localize, crop_roi]，
        即使检测阶段执行了detect_lesions，也不展示给AI。
        """
        # 过滤工具结果：只保留策略工具序列中的工具
        strategy_tool_sequence = strategy.get("tool_sequence", [])
        filtered_tool_results = {
            tool_name: result
            for tool_name, result in tool_results.items()
            if tool_name in strategy_tool_sequence
        }

        # 记录过滤情况
        all_tools = set(tool_results.keys())
        shown_tools = set(filtered_tool_results.keys())
        hidden_tools = all_tools - shown_tools

        if hidden_tools:
            logger.info(f"AI可见工具: {sorted(shown_tools)}")
            logger.info(f"隐藏工具（不在策略序列中）: {sorted(hidden_tools)}")
        else:
            logger.info(f"AI可见工具: {sorted(shown_tools)} (全部工具)")

        # 格式化过滤后的工具结果
        tool_results_formatted = self._format_tool_results(filtered_tool_results)

        # 构建提示
        # ✅ 检查是否有标注图
        annotated_image_info = ""
        if "annotated_image" in image_data:
            annotated_data = image_data["annotated_image"]
            if "thumbnail" in annotated_data and annotated_data["thumbnail"]:
                annotated_image_info = "\n* **Annotated Image (ETDRS grid):** [已添加到多模态输入]"
                logger.info(f"✓ 在提示中添加标注图信息")

        prompt = f"""### Task Instructions

**Input Data:**
* **Question:** {question}
* **Image Path:** {image_path}
* **Ground Truth Answer:** {ground_truth_answer}
* **Original Image:** [已添加到多模态输入]{annotated_image_info}

**Pre-executed REAL Tool Results:**
```json
{tool_results_formatted}
```

**Reasoning Type:** {strategy.get('reasoning_type', 'Unknown')}-{strategy.get('reasoning_subtype', 'unknown')}
**Strategy:** {strategy.get('system_prompt_suffix', '').split(':')[0] if strategy and ':' in strategy.get('system_prompt_suffix', '') else 'Standard'}

---

**CRITICAL INSTRUCTIONS:**

1. **Visual Analysis First:** Start with <think> describing what you SEE in the image that matches the ground truth

2. **Use Placeholder for Tool Results:** When you call a tool in <tool_call>, you MUST:
   - Write `<observation>[TOOL_RESULT_PLACEHOLDER:tool_name]</observation>`
   - Do NOT copy the full JSON result (we will insert it automatically later)
   - Example: `<observation>[TOOL_RESULT_PLACEHOLDER:DR_Grading]</observation>`

3. **Reasoning Strategy:** Follow this workflow:
   - Tool Sequence: {' → '.join(strategy.get('tool_sequence', []))}
   - Max Steps: {strategy.get('max_steps', 3)}
   - Enable Correction: {strategy.get('enable_correction', False)}
   - Enable Crop Verification: {strategy.get('enable_crop_verification', False)}

4. **REQUIRED XML Format (YOU MUST INCLUDE ALL THESE TAGS):**
   ```xml
   <think>
   (Visual Analysis): Describe what you see
   (Hypothesis): Medical hypothesis
   (Plan): Which tool to call first
   </think>

   <tool_call>
   Tool: [tool_name]
   Args: {{"arg1": "value1"}}
   </tool_call>

   <observation>[TOOL_RESULT_PLACEHOLDER:tool_name]</observation>

   <think>
   (Analysis): Interpret the result
   (Refinement): Do I need more tools?
   (Plan): Next step
   </think>

   ... (repeat tool_call + observation as needed) ...

   <answer>
   (Final Conclusion): Based on all evidence, the answer is...
   </answer>
   ```

5. **Natural Reasoning:** Present reasoning as if discovering it step-by-step (never mention "ground truth" or "pre-executed results")

6. **MANDATORY TAGS - CRITICAL COMPLETION REQUIREMENT:**
   Your response is INCOMPLETE and INVALID unless it contains:
   - At least one <think> block at the beginning
   - At least one <tool_call> block
   - At least one <observation> block with placeholder (matching each tool_call)
   - Exactly one <answer> block at the end ← YOU MUST REACH THIS TAG

   **IMPORTANT**: Use placeholders in <observation> blocks, do NOT copy long JSON.
   This makes generation much faster and prevents truncation.
   Format: `<observation>[TOOL_RESULT_PLACEHOLDER:tool_name]</observation>`

   Do NOT stop generating until you have written the complete <answer> block.
   Your response is WORTHLESS without the closing <answer> tag.

---

Generate the complete trajectory now (pure XML, NO markdown fences):
"""

        return prompt

    def _format_tool_results(self, tool_results: Dict) -> str:
        """格式化工具结果为可读JSON"""
        formatted = {}
        for tool_name, result in tool_results.items():
            # 简化结果，只保留关键信息
            if result.get("status") == "success":
                formatted[tool_name] = result.get("result", result)
            else:
                formatted[tool_name] = {
                    "status": "error",
                    "error": result.get("error", "Unknown error")
                }

        return json.dumps(formatted, indent=2, ensure_ascii=False)

    def _generate_with_retry(
        self, 
        system_prompt: str, 
        user_prompt: str,
        tool_sequence: list = None,
        images=None
    ) -> str:
        """带重试的生成
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            tool_sequence: 工具序列
            images: 图像列表，用于多模态输入
        """
        all_attempts = []

        for attempt in range(self.retry_times):
            try:
                trajectory = self.api_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    max_tokens=8000,
                    images=images
                )

                # 验证必要标签
                required_tags = ["<think>","<answer>"]
                missing_tags = [tag for tag in required_tags if tag not in trajectory]
                missing_tags = [tag for tag in required_tags if tag not in trajectory]
                              # 如果有工具调用，则需要 <ool_call> 和 <observation> 标签
                if tool_sequence and len(tool_sequence) > 0:
                    required_tags.extend(["<tool_call>", "<observation>"])
                else:
                    # 如果没有工具调用（如healthy），检查是否有不应该出现的标签
                    if "<tool_call>" in trajectory or "<observation>" in trajectory:
                        logger.warning(f"⚠️ 工具序列为空，但生成的轨迹包含 <tool_call> 或 <observation> 标签")

                # 记录这次尝试
                attempt_record = {
                    "attempt": attempt + 1,
                    "trajectory": trajectory,
                    "missing_tags": missing_tags,
                    "success": len(missing_tags) == 0
                }
                all_attempts.append(attempt_record)

                if not missing_tags:
                    logger.info(f"✓ 生成成功 (尝试 {attempt + 1}/{self.retry_times})")
                    return trajectory
                else:
                    logger.warning(f"生成结果缺少必要标签: {missing_tags} (尝试 {attempt + 1}/{self.retry_times})")
                    if attempt < self.retry_times - 1:
                        logger.info("调整提示词强度，重新生成...")

            except Exception as e:
                # 记录异常尝试
                attempt_record = {
                    "attempt": attempt + 1,
                    "trajectory": None,
                    "error": str(e),
                    "success": False
                }
                all_attempts.append(attempt_record)

                logger.warning(f"生成失败 (尝试 {attempt + 1}/{self.retry_times}): {e}")
                if attempt < self.retry_times - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        # 所有尝试都失败了，返回包含所有尝试记录的异常
        error_msg = f"生成失败，已达到最大重试次数 ({self.retry_times})"
        logger.error(error_msg)

        # 将所有尝试记录附加到异常中
        error = RuntimeError(error_msg)
        error.all_attempts = all_attempts  # 附加所有尝试记录
        raise error

    def _summarize_tool_result(self, tool_name: str, tool_result: Dict) -> str:
        """
        将工具结果总结为简洁的自然语言描述

        Args:
            tool_name: 工具名称
            tool_result: 工具执行结果

        Returns:
            str: 工具结果的自然语言总结
        """
        if tool_result.get("status") != "success":
            return f"工具执行失败: {tool_result.get('error', 'Unknown error')}"

        result_data = tool_result.get("result", {})

        # 根据不同工具类型生成总结
        if tool_name == "quality_assess":
            confs = result_data.get("confs", 0)
            assessment = result_data.get("assessment", "未知")
            return f"图像质量评分: {confs:.2f}, 评估结果: {assessment}"

        elif tool_name == "AMD_predict":
            score = result_data.get("simplified_score", {}).get("score", 0)
            results = result_data.get("results", {})
            drusen_info = results.get("drusen", {})
            pigment_info = results.get("pigment", {})
            amd_info = results.get("amd", {})

            summary_parts = []
            summary_parts.append(f"AREDS评分: {score}/5")
            if drusen_info.get("label"):
                summary_parts.append(f"玻璃膜疣: {drusen_info['label']}")
            if pigment_info.get("label"):
                summary_parts.append(f"色素异常: {pigment_info['label']}")
            if amd_info.get("label"):
                summary_parts.append(f"晚期AMD: {amd_info['label']}")

            return ", ".join(summary_parts)

        elif tool_name == "fovea_od_localize":
            coords = result_data.get("coordinates", {})
            fovea = coords.get("fovea", {})
            od = coords.get("optic_disc", {})
            return f"黄斑中心位置: ({fovea.get('x', 0):.0f}, {fovea.get('y', 0):.0f}), 视盘位置: ({od.get('x', 0):.0f}, {od.get('y', 0):.0f})"

        elif tool_name == "crop_roi":
            cropped_path = result_data.get("cropped_image_path", "")
            bbox = result_data.get("original_bbox", {})
            sr_applied = result_data.get("super_resolution_applied", False)
            sr_scale = result_data.get("sr_scale", 1)

            summary_parts = []
            summary_parts.append(f"裁剪区域: x={bbox.get('x1', 0)}, y={bbox.get('y1', 0)}, x2={bbox.get('x2', 0)}, y2={bbox.get('y2', 0)}")
            if sr_applied:
                summary_parts.append(f"超分辨率放大: {sr_scale}x")
            if cropped_path:
                summary_parts.append(f"输出图像: {cropped_path}")

            return ". ".join(summary_parts)

        elif tool_name == "detect_lesions":
            # ✅ UNet病灶分割格式
            tasks = result_data.get("分析结果", {})
            detected_lesions = []

            for task_name, task_result in tasks.items():
                # 跳过整体分类，只关注病灶分割
                if "分类" in task_name:
                    continue

                status = task_result.get("状态", "")
                # 检查是否成功完成（状态包含"完成"即可）
                if "完成" in status:
                    metrics = task_result.get("指标", {})

                    # 根据不同病灶类型提取信息
                    lesion_summary = None
                    if "微动脉瘤" in task_name:
                        count = metrics.get("微动脉瘤数量", 0)
                        ratio = metrics.get("微动脉瘤区域占比", 0)
                        if count > 0:
                            lesion_summary = f"微动脉瘤: {count}个 (占比{ratio:.3%})"
                    elif "出血" in task_name:
                        ratio = metrics.get("出血区域占比", 0)
                        size = metrics.get("出血区域总大小", 0)
                        if ratio > 0.001 or size > 0:  # 阈值0.1%
                            lesion_summary = f"出血: 占比{ratio:.3%} (面积{size:.0f}px)"
                    elif "硬渗出物" in task_name:
                        ratio = metrics.get("硬渗出物区域占比", 0)
                        if ratio > 0.001:  # 阈值0.1%
                            lesion_summary = f"硬渗出物: 占比{ratio:.3%}"
                    elif "软渗出物" in task_name:
                        ratio = metrics.get("软渗出物区域占比", 0)
                        if ratio > 0.001:  # 阈值0.1%
                            lesion_summary = f"软渗出物: 占比{ratio:.3%}"

                    if lesion_summary:
                        detected_lesions.append(lesion_summary)

            if detected_lesions:
                return "检测到: " + ", ".join(detected_lesions)
            else:
                return "未检测到明显病灶"

        elif tool_name == "DR_Grading":
            # ✅ DINO工具格式
            prediction = result_data.get("prediction", {})
            class_name = prediction.get("class_name", "")
            class_id = prediction.get("class_id", -1)
            confidence = prediction.get("confidence", 0)
            return f"DR分级: {class_name} (类别 {class_id}, 置信度: {confidence:.2f})"

        elif tool_name == "dme_risk_assess":
            risk_level = result_data.get("risk_level", "")
            exudates_near_fovea = result_data.get("exudates_near_fovea", False)
            summary = f"DME风险: {risk_level}"
            if exudates_near_fovea:
                summary += ", 黄斑附近有渗出物"
            return summary

        elif tool_name == "DR_segmentation_anchor_ETDRS":
            nested = result_data
            if isinstance(result_data, dict):
                if "results" in result_data and isinstance(result_data["results"], dict):
                    nested = result_data["results"]
            acquired = {}
            if isinstance(nested, dict):
                acquired = nested.get("acquired_parameters", {}) or {}
            fovea_coord = acquired.get("fovea_coord") or acquired.get("fovea") or []
            od_coord = acquired.get("od_coord") or acquired.get("optic_disc") or []
            disc_data = acquired.get("disc_diameter_data", {}) or {}
            output_vis_path = None
            if isinstance(nested, dict):
                output_vis_path = nested.get("output_visualization_path")

            parts = []
            #parts.append("ETDRS网格锚定完成")
            if output_vis_path:
                parts.append(f"已定位中央凹与视盘，并输出ETDRS锚定可视化结果")

            return "；".join(parts)

        elif tool_name == "segment_by_AutoMorphalyzer":
            images = result_data.get("images", [])
            if images and len(images) > 0:
                od = images[0].get("optic_disc", {})
                cdr_v = od.get("cdr_vertical", 0)
                cdr_h = od.get("cdr_horizontal", 0)
                return f"杯盘比垂直: {cdr_v:.2f}, 杯盘比水平: {cdr_h:.2f}"
            return "视盘分割完成"

        elif tool_name == "enhance_image":
            output_path = result_data.get("output_path", "")
            if output_path:
                return f"图像增强完成，输出路径: {output_path}"
            else:
                return "图像增强完成"

        else:
            # 通用总结：返回关键信息
            if isinstance(result_data, dict):
                keys = list(result_data.keys())[:5]  # 最多显示5个键
                info = ", ".join([f"{k}={result_data[k]}" for k in keys])
                return f"工具执行成功: {info}"
            else:
                return "工具执行成功"

    def _replace_tool_placeholders(self, trajectory: str, tool_results: Dict) -> str:
        """
        替换轨迹中的工具占位符为真实结果

        Args:
            trajectory: 包含占位符的轨迹
            tool_results: 工具执行结果字典

        Returns:
            替换后的轨迹
        """
        import re

        # 匹配占位符模式: [TOOL_RESULT_PLACEHOLDER:tool_name]
        placeholder_pattern = r'\[TOOL_RESULT_PLACEHOLDER:(\w+)\]'

        def replace_placeholder(match):
            tool_name = match.group(1)
            if tool_name in tool_results:
                result = tool_results[tool_name]
                # 使用自然语言总结替代原始JSON
                summary = self._summarize_tool_result(tool_name, result)
                return summary
            else:
                logger.warning(f"工具 {tool_name} 的结果未找到，保留占位符")
                return match.group(0)  # 保留原占位符

        # 替换所有占位符
        replaced_trajectory = re.sub(placeholder_pattern, replace_placeholder, trajectory)

        # 统计替换情况
        placeholders_found = re.findall(placeholder_pattern, trajectory)
        placeholders_remaining = re.findall(placeholder_pattern, replaced_trajectory)

        logger.info(f"占位符替换: 发现 {len(placeholders_found)} 个, 成功替换 {len(placeholders_found) - len(placeholders_remaining)} 个")
        if placeholders_remaining:
            logger.warning(f"仍有 {len(placeholders_remaining)} 个占位符未替换: {placeholders_remaining}")

        return replaced_trajectory

    def batch_generate_v2(
        self,
        examples: List[CoTExample],
        output_path: Optional[str] = None,
        save_interval: int = 5,
        completed_indices: set = None,
        existing_results: list = None,
        progress_file: str = None
    ) -> List[Dict]:
        """
        批量生成（V2版本，支持断点续跑）

        Args:
            examples: 示例列表
            output_path: 输出文件路径
            save_interval: 保存间隔
            completed_indices: 已完成的样本索引集合（断点续跑）
            existing_results: 已生成的结果列表（断点续跑）
            progress_file: 进度文件路径（断点续跑）
        """
        if completed_indices is None:
            completed_indices = set()
        if existing_results is None:
            existing_results = []

        results = existing_results.copy()  # 复制已存在的结果
        total = len(examples)
        remaining = total - len(completed_indices)

        logger.info(f"开始批量生成，共 {total} 个样本")
        if completed_indices:
            logger.info(f"  已完成: {len(completed_indices)} 个")
            logger.info(f"  待生成: {remaining} 个")

        # 只处理未完成的样本
        pending_examples = [(i, ex) for i, ex in enumerate(examples) if i not in completed_indices]

        if not pending_examples:
            logger.info(f"✓ 所有样本均已完成，无需生成")
            return results

        logger.info(f"提交 {len(pending_examples)} 个样本到线程池")

        # 用于追踪当前批次完成的索引
        current_batch_completed = set()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有待处理的样本
            future_to_index_example = {
                executor.submit(
                    self.generate_trajectory_with_real_tools,
                    ex.question,
                    ex.image_path,
                    ex.ground_truth_answer
                ): (idx, ex)
                for idx, ex in pending_examples
            }

            with tqdm(total=remaining, desc="生成进度") as pbar:
                for future in as_completed(future_to_index_example):
                    idx, example = future_to_index_example[future]
                    try:
                        gen_result = future.result()

                        result = {
                            "question": example.question,
                            "image_path": example.image_path,
                            "ground_truth_answer": example.ground_truth_answer,
                            "trajectory": gen_result["trajectory"],
                            "metadata": gen_result["metadata"],
                            "tool_execution_log": gen_result["tool_execution_log"],
                            "generation_attempts": gen_result.get("generation_attempts", []),
                            "error": gen_result["error"]
                        }

                        results.append(result)
                        current_batch_completed.add(idx)

                        # 定期保存
                        completed_count = len(results)
                        if completed_count % save_interval == 0 and output_path:
                            self._save_results(results, output_path)

                            # 更新进度文件
                            if progress_file:
                                self._save_progress(
                                    progress_file,
                                    completed_indices | current_batch_completed,
                                    total
                                )

                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"处理示例失败 (索引 {idx}): {e}")
                        results.append({
                            "question": example.question,
                            "image_path": example.image_path,
                            "ground_truth_answer": example.ground_truth_answer,
                            "trajectory": None,
                            "generation_attempts": [],
                            "error": str(e)
                        })
                        current_batch_completed.add(idx)  # 即使失败也标记为完成
                        pbar.update(1)

        # 最终保存
        if output_path:
            self._save_results(results, output_path)

        # 更新最终进度
        if progress_file:
            all_completed = completed_indices | current_batch_completed
            self._save_progress(progress_file, all_completed, total)

            # 如果全部完成，可以选择删除进度文件或保留
            if len(all_completed) == total:
                logger.info(f"✓ 所有样本已完成")
                # 可选：删除进度文件
                # try:
                #     os.remove(progress_file)
                #     logger.info(f"✓ 已删除进度文件: {progress_file}")
                # except:
                #     pass

        # 统计
        success = sum(1 for r in results if r.get("trajectory") is not None)
        logger.info(f"生成完成: 成功 {success}/{total}")

        return results

    def _save_results(self, results: List[Dict], output_path: str):
        """保存结果"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存: {output_path}")

    def _save_progress(self, progress_file: str, completed_indices: set, total: int):
        """保存进度到文件"""
        try:
            progress_data = {
                'completed_indices': sorted(list(completed_indices)),
                'total': total,
                'completed_count': len(completed_indices),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            progress_path = Path(progress_file)
            progress_path.parent.mkdir(parents=True, exist_ok=True)

            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"进度已保存: {len(completed_indices)}/{total}")
        except Exception as e:
            logger.warning(f"保存进度文件失败: {e}")


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 初始化组件
    from dotenv import load_dotenv
    load_dotenv()

    api_client = APIClient(model="gpt-4o")
    tool_executor = RealToolExecutor(
        tools_interface_path="../agents/fundus_tools_agent",
        enable_cache=True
    )

    generator = CoTGeneratorV2(api_client, tool_executor)

    # 测试单个生成
    result = generator.generate_trajectory_with_real_tools(
        question="What is shown in the image? A: Healthy B: Mild DR C: Moderate DR",
        image_path="./test_images/sample.jpg",
        ground_truth_answer="B"
    )

    print("\n=== 生成结果 ===")
    print(f"轨迹长度: {len(result['trajectory']) if result['trajectory'] else 0}")
    print(f"推理类型: {result['metadata']['reasoning_type']}-{result['metadata']['reasoning_subtype']}")
    print(f"工具执行: {len(result['tool_execution_log'])} 个")
    print(f"\n轨迹内容:\n{result['trajectory'][:500]}...")
