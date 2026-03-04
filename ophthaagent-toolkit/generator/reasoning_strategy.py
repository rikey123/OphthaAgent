"""
推理策略生成器
根据推理类型生成具体的工具调用策略和Prompt指导
"""
import logging
from typing import Dict, List, Optional
from reasoning_classifier import ReasoningType

logger = logging.getLogger(__name__)


class ReasoningStrategy:
    """根据推理类型生成具体的工具调用策略"""

    # 工具依赖关系
    TOOL_DEPENDENCIES = {
        "crop_roi": ["fovea_od_localize"],  # Crop需要先定位
        "dme_risk_assess": ["detect_lesions", "fovea_od_localize"],  # DME需要病灶和定位
        "DR_segmentation_anchor_ETDRS": ["detect_lesions"], # Anchor需要病灶
    }

    # 统一的XML格式要求（所有策略通用）
    XML_FORMAT_REQUIREMENT = """
**CRITICAL: XML FORMAT REQUIREMENT (MANDATORY FOR ALL RESPONSES):**

Your response MUST follow this strict XML structure:

1. Start with <think> block (visual analysis + hypothesis + plan)
2. Each <invoke> MUST be immediately followed by <observation>
3. Add <think> blocks between tool calls for reasoning
4. End with exactly one <answer> block
5. DO NOT use markdown code fences (no ```xml or ```)

**Example structure:**


<invoke>
Tool: [tool_name]
Args: {{"arg": "value"}}
</invoke>

<observation>
[Provide a concise natural language summary of the tool result - DO NOT copy raw JSON]
</observation>

<think>
(Analysis of result)
</think>

<answer>
(Final conclusion)
</answer>
"""

    @classmethod
    def get_strategy(cls, reasoning_type: ReasoningType) -> Dict:
        """
        返回推理策略配置

        Returns:
            dict: {
                "tool_sequence": List[str],  # 工具调用顺序
                "enable_correction": bool,  # 是否启用纠错
                "enable_crop_verification": bool,  # 是否启用Crop验证
                "enable_cascade": bool,  # 是否启用级联
                "enable_parallel": bool,  # 是否启用并行
                "max_steps": int,  # 最大步骤数
                "system_prompt_suffix": str,  # 额外的系统提示
                "correction_trigger": str,  # 纠错触发条件
                "cascade_rule": str  # 级联规则
            }
        """
        # Input validation
        if not reasoning_type:
            logger.error("get_strategy() called with None reasoning_type, using default")
            return cls._get_default_strategy_with_metadata("A", "pure_dr", 0.5)

        if not isinstance(reasoning_type, ReasoningType):
            logger.error(f"Invalid reasoning_type type: {type(reasoning_type)}, using default")
            return cls._get_default_strategy_with_metadata("A", "pure_dr", 0.5)

        strategies = {
            "A": cls._get_type_a_strategies(),
            "B": cls._get_type_b_strategies(),
            "C": cls._get_type_c_strategies(),
            "D": cls._get_type_d_strategies(),
            "E": cls._get_type_e_strategies()
        }

        type_strategies = strategies.get(reasoning_type.type, {})
        strategy = type_strategies.get(reasoning_type.subtype, cls._get_default_strategy())

        # 添加元数据 - 防御性检查
        if not strategy or not isinstance(strategy, dict):
            logger.error(f"Strategy is None or invalid after lookup for {reasoning_type.type}-{reasoning_type.subtype}")
            return cls._get_default_strategy_with_metadata(reasoning_type.type, reasoning_type.subtype, reasoning_type.confidence)

        strategy["reasoning_type"] = reasoning_type.type
        strategy["reasoning_subtype"] = reasoning_type.subtype
        strategy["confidence"] = reasoning_type.confidence

        return strategy

    @classmethod
    def _get_type_a_strategies(cls) -> Dict:
        """Type A: 标准SOP策略"""
        return {
            "healthy": {
                "tool_sequence": [],  # 不调用任何工具
                "enable_correction": False,
                "enable_crop_verification": False,
                "max_steps": 1,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**STANDARD SOP - HEALTHY (纯视觉分析):**

CRITICAL: For healthy fundus cases, DO NOT use any tool calls.
Generate ONLY visual analysis with <think> and <answer> blocks.

For healthy fundus:
1. Visual inspection ONLY: Describe normal optic disc, macula, and retinal vessels
2. NO tool calls - pure visual assessment is sufficient for healthy diagnosis
3. Conclude directly with healthy assessment based on visual observation
4. Recommend routine follow-up

**Required Format (NO TOOL CALLS):**

<think>
(Visual Analysis): I observe a clear fundus photograph with well-defined anatomical structures. The optic disc appears normal with sharp margins and healthy pink coloration. The cup-to-disc ratio appears physiologic (approximately 0.3-0.4). The macula shows an intact foveal reflex without drusen, hemorrhages, or pigmentary changes. Retinal vessels display normal caliber and branching patterns, with no signs of arteriovenous nicking, tortuosity, or abnormal neovascularization. The peripheral retina is clear without hemorrhages, exudates, cotton-wool spots, or other pathological lesions.

(Assessment): All major fundus structures appear within normal limits. No signs of diabetic retinopathy, age-related macular degeneration, glaucomatous optic neuropathy, or other retinal pathology are evident.

(Conclusion): This represents a healthy fundus with no pathological findings detected on visual examination.
</think>

<answer>
Based on comprehensive visual assessment of this fundus photograph:

**Findings:**
- Optic disc: Normal size, shape, and coloration with healthy neuroretinal rim
- Macula: Intact foveal reflex, no drusen or pigmentary abnormalities
- Retinal vessels: Normal caliber, no arteriovenous changes or neovascularization
- Peripheral retina: Clear, no hemorrhages, exudates, or other lesions

**Diagnosis:** Healthy fundus with no pathological findings

**Recommendation:** Routine annual eye examination for ongoing monitoring

This fundus shows no evidence of diabetic retinopathy, macular degeneration, glaucoma, or other retinal disease.
</answer>

**CRITICAL REMINDER**:
- DO NOT include <tool_call> blocks
- DO NOT include <observation> blocks
- ONLY use <think> and <answer> blocks
- Pure visual analysis is sufficient for healthy diagnosis
"""
            },

            "pure_amd": {
                "tool_sequence": ["AMD_predict", "fovea_od_localize", "crop_roi"],
                "enable_crop_verification": True,
                "max_steps": 4,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**STANDARD SOP - AMD:**
1. Visual analysis: Check for drusen/pigment changes in macula
2. Call AMD_predict for classification
3. If confidence is borderline or early AMD, crop macula for detailed verification
4. Conclude with AMD stage and recommendation"""
            },

            "pure_glaucoma": {
                "tool_sequence": ["segment_by_AutoMorphalyzer"],
                "enable_correction": False,
                "max_steps": 3,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**STANDARD SOP - GLAUCOMA:**
1. Visual analysis: Check optic cup size
2. Call segment_by_AutoMorphalyzer to calculate CDR
3. Interpret CDR value and conclude"""
            },

            "pure_dr": {
                "tool_sequence": ["DR_Grading"],
                "enable_correction": False,
                "max_steps": 3,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**STANDARD SOP - DR:**
1. Visual analysis: Identify hemorrhages/microaneurysms
2. Call DR_Grading for severity classification
3. Conclude with DR stage"""
            },
            "pure_seg": {
                "tool_sequence": ["detect_lesions",
                    "DR_segmentation_anchor_ETDRS"],
                "enable_correction": False,
                "max_steps": 3,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**STANDARD SOP - DR:**
1. Visual analysis: Identify hemorrhages/microaneurysms
2. call detect_lesions for quantitative analysis
3. Call DR_segmentation_anchor_ETDRS for lesion distribution
    Analyzing a fundus image overlayed with an ETDRS grid and segmentation masks.
    - PINK: Microaneurysms (MA)
    - RED: Hemorrhages (HE)
    - CYAN/BLUE: Hard Exudates (EX)
    - YELLOW: Soft Exudates / Cotton Wool Spots (CWS/SE)
    - GREEN: blood vessels
    - WHITE: optic cup
4. Artifact Rejection Rules: identify specific False Positive Patterns
    - Vessel Interference: Linear/"string-of-pearls" within 1-2px of vessel mask.
    - Anatomical Reflexes: Bright rims at Optic Disc or Fovea center.
    - Edge/Shadows: Dark streaks/blotches at image boundary.
    - Common confusion (e.g., EX and bright reflections, MA and HE) 
5. Generate a <think> block to describe the lesion distribution:
- Report each type lesion using ETDRS subfields and retinal quadrants.
- <think>The pink dots along the superior arcade appear to be vessel artifacts (linear shape), but the red blotches in the temporal quadrant are genuine hemorrhages.Distribution: "Validated Hard Exudates (Cyan) form a circinate ring in the superior inner subfield, threatening the fovea.<think>
3. Conclude with DR lession"""
            }
        }

    @classmethod
    def _get_type_b_strategies(cls) -> Dict:
        """Type B: 纠错与鉴别策略"""
        return {
            "dr_grading_mismatch": {
                "tool_sequence": [
                    "DR_Grading",
                    "detect_lesions",
                    "DR_segmentation_anchor_ETDRS"
                ],
                "enable_correction": True,
                "max_steps": 6,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**CORRECTION MODE - DR GRADING MISMATCH:**
When your visual assessment of DR severity contradicts the DR_Grading output:
1. First call DR_Grading.
2. CRITICAL CHECKPOINT: Compare your visual impression with the DR_Grading result:
   - Under-grade: Obvious lesions (MAs/Hemes/CWS) → Tool: Normal/Mild.
   - Over-grade: Healthy appearance or minimal changes → Tool: Moderate or higher.
   - Severity gap: Significant classification divergence (e.g., Mild vs. Severe).
3. Generate a <think> block explaining the discrepancy:
    ”The visual assessment contradicts the DR_Grading result.“
    - Brief visual re-evaluation description
    - Hypothesis whether the tool under or over-grade and why (e.g., absent key signs).
4. call detect_lesions for quantitative analysis
6. Call DR_segmentation_anchor_ETDRS for lesion distribution
    Analyzing a fundus image overlayed with an ETDRS grid and segmentation masks.
    - PINK: Microaneurysms (MA)
    - RED: Hemorrhages (HE)
    - CYAN/BLUE: Hard Exudates (EX)
    - YELLOW: Soft Exudates / Cotton Wool Spots (CWS/SE)
    - GREEN: blood vessels
    - WHITE: optic cup
7. Artifact Rejection Rules: identify specific False Positive Patterns
    - Vessel Interference: Linear/"string-of-pearls" within 1-2px of vessel mask.
    - Anatomical Reflexes: Bright rims at Optic Disc or Fovea center.
    - Edge/Shadows: Dark streaks/blotches at image boundary.
    - Common confusion (e.g., EX and bright reflections, MA and HE) 
8. Generate a <think> block to describe the lesion distribution:
    - Report the distribution of each type lesion using ETDRS subfields and retinal quadrants.
    - Reject the Artifact and focus on the valid lesions.
    - Apply the ICDR Scale to the VALIDATED lesions
9. CORRECT your final diagnosis accordingly.
    

** EXAMPLE:**
<think> 
The model leans towards Proliferative Diabetic Retinopathy.
However, upon visual re-evaluation:
- the visual evidence of frank neovascularization (new vessels on the disc or elsewhere) is not immediately obvious to me
- whereas the signs of severe ischemia (cotton wool spots, extensive hemorrhages) are prominent.
- The model might be over-grading severe NPDR features as PDR.
I must quantify with lesion analysis. </think>

<tool_call>
Tool: detect_lesions
Args: {{"image_path": "..."}}
</tool_call>

"""
            },
            "false_dr_is_amd": {
                "tool_sequence": ["detect_lesions", "fovea_od_localize", "AMD_predict"],
                "enable_correction": True,
                "correction_trigger": "hard_exudates_in_macula",
                "max_steps": 5,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**CRITICAL CORRECTION MODE - FALSE DR:**
When you detect "hard exudates" or similar lesions:
1. First call detect_lesions
2. CRITICAL CHECKPOINT: If exudates are:
   - Concentrated in macular center (not peripheral)
   - Have soft/blurred edges (not sharp)
   - Confluent and yellowish-white
   → This is MORE likely AMD drusen than DR hard exudates
3. Generate a <think> block explaining the discrepancy:
    "The lesions reported as 'hard exudates' have characteristics inconsistent with DR:
   - Central location (macula-centric, not scattered)
   - Soft edges (drusen-like, not sharp exudates)
   - No accompanying hemorrhages typical of DR"
4. Call AMD_predict to confirm
5. CORRECT your final diagnosis to AMD (not DR)

**EXAMPLE:**
<think>
The detect_lesions tool identified lesions classified as "hard exudates."
However, upon visual re-evaluation:
- These lesions are strictly concentrated at the foveal center
- They exhibit soft, blurred margins characteristic of drusen
- No dot-blot hemorrhages or microaneurysms are present
This pattern is inconsistent with DR and highly suggestive of AMD.
I must verify with AMD-specific analysis.
</think>

<tool_call>
Tool: AMD_predict
Args: {{"image_path": "..."}}
</tool_call>"""
            },

            "false_glaucoma": {
                "tool_sequence": ["segment_by_AutoMorphalyzer", "fovea_od_localize", "crop_roi"],
                "enable_crop_verification": True,
                "correction_trigger": "cdr_borderline",
                "max_steps": 5,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**CORRECTION MODE - PHYSIOLOGIC LARGE CUP:**
When CDR is borderline (0.5-0.6):
1. Calculate CDR with segment_by_AutoMorphalyzer
2. If CDR ≈ 0.5-0.6, this is borderline
3. Crop optic disc for detailed rim inspection
4. Check for glaucomatous signs:
   - Rim thinning (especially inferior/superior)
   - Notching or focal rim loss
   - RNFL defects
5. If rim is healthy, pink, and symmetrical → CORRECT to "physiologic large cup, not glaucoma"

Visual hallucination tip: Describe seeing "healthy pink neuroretinal rim without notching" """
            },

            "artifact": {
                "tool_sequence": ["detect_lesions", "fovea_od_localize", "crop_roi"],
                "enable_correction": True,
                "max_steps": 4,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**ARTIFACT VERIFICATION:**
If tool detects microaneurysms/hemorrhages:
1. Crop the detected region for close inspection
2. Visual verification checklist:
   - Is it on a vessel wall? (likely artifact)
   - Does it have a reflective/shiny appearance? (artifact)
   - Is the shape too regular/geometric? (artifact)
3. If artifact confirmed, REJECT the detection as false positive
4. Conclude as healthy or other diagnosis"""
            },

            "low_confidence_recheck": {
                "tool_sequence": ["AMD_predict", "fovea_od_localize", "crop_roi"],
                "enable_crop_verification": True,
                "max_steps": 4,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**LOW CONFIDENCE RECHECK:**
When initial tool returns low confidence (<0.7):
1. Identify the uncertain region (e.g., macula for AMD)
2. Crop that region for high-res inspection
3. Re-evaluate visually with enhanced detail
4. Confirm or revise diagnosis"""
            }
        }

    @classmethod
    def _get_type_c_strategies(cls) -> Dict:
        """Type C: 深度级联策略"""
        return {
            "dr_to_dme": {
                "tool_sequence": ["DR_Grading", "detect_lesions", "fovea_od_localize", "dme_risk_assess"],
                "enable_cascade": True,
                "cascade_rule": "if exudates_near_macula then trigger_dme_assessment",
                "max_steps": 5,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**CASCADE REASONING - DR to DME:**
Workflow:
1. Grade DR severity
2. Detect and localize lesions
3. CASCADE TRIGGER: If hard exudates are within 500μm of fovea:
   → MUST trigger DME risk assessment
4. Call dme_risk_assess
5. Synthesize: "Severe NPDR with high-risk DME"

Visual cue: Describe seeing "hard exudates encroaching on the foveal center" """
            },

            "pdr_to_nvg": {
                "tool_sequence": ["DR_Grading", "segment_by_AutoMorphalyzer"],
                "enable_cascade": True,
                "cascade_rule": "if pdr_and_high_cdr then neovascular_glaucoma",
                "max_steps": 4,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**CASCADE REASONING - PDR to NVG:**
Workflow:
1. Grade DR → confirms PDR (proliferative)
2. CASCADE TRIGGER: PDR patients are at risk for neovascular glaucoma
3. Call segment_by_AutoMorphalyzer to check CDR
4. If CDR > 0.7 → confirms NVG (secondary glaucoma)
5. Synthesize: "PDR complicated by neovascular glaucoma - urgent referral needed"

This is a high-risk emergency scenario."""
            }
        }

    @classmethod
    def _get_type_d_strategies(cls) -> Dict:
        """Type D: 多病并发策略"""
        return {
            "dr_and_amd": {
                "tool_sequence": ["DR_Grading", "detect_lesions", "AMD_predict"],
                "enable_parallel": True,
                "max_steps": 5,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**MULTI-DISEASE - DR + AMD:**
Two independent disease processes:
1. Peripheral retina: Hemorrhages/microaneurysms (DR features)
2. Central macula: Drusen/pigment (AMD features)

Workflow:
1. Visual: Identify BOTH peripheral DR lesions AND central AMD changes
2. Call DR_Grading → confirms DR
3. Call detect_lesions → quantify DR lesions
4. Call AMD_predict → confirms AMD
5. Synthesize: "Patient has BOTH non-proliferative DR and dry AMD (concurrent diseases)"

Key: Explain why they coexist (diabetes + aging)"""
            },

            "dr_and_glaucoma": {
                "tool_sequence": ["DR_Grading", "segment_by_AutoMorphalyzer"],
                "enable_parallel": True,
                "max_steps": 4,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**MULTI-DISEASE - DR + GLAUCOMA:**
1. Visual: See both small hemorrhages AND large optic cup
2. DR_Grading → Mild NPDR
3. segment_by_AutoMorphalyzer → CDR 0.75
4. Reasoning: "DR is mild and does not explain the large cup. Glaucoma is a separate, independent condition."
5. Conclude: "Primary open-angle glaucoma coexisting with mild DR (two separate pathologies)" """
            },

            "multi_disease": {
                "tool_sequence": ["quality_assess", "DR_Grading", "AMD_predict", "segment_by_AutoMorphalyzer"],
                "enable_parallel": True,
                "max_steps": 6,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**MULTI-DISEASE - GENERAL:**
Systematically evaluate each major disease category:
1. DR: Check for hemorrhages/exudates
2. AMD: Check macula for drusen
3. Glaucoma: Check optic cup

Report all findings independently and explain coexistence."""
            }
        }

    @classmethod
    def _get_type_e_strategies(cls) -> Dict:
        """Type E: 拒诊与异常策略"""
        return {
            "low_quality": {
                "tool_sequence": ["quality_assess"],
                "enable_rejection": True,
                "rejection_threshold": 0.3,
                "max_steps": 2,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**REJECTION MODE - LOW QUALITY:**
Workflow:
1. Visual: Image appears dark/blurry/unclear
2. Call quality_assess
3. If quality score < 0.3 (Poor):
   → MUST REJECT diagnosis
4. Conclude: "Image quality is insufficient for reliable diagnosis. Please retake the image with better lighting/focus."

DO NOT attempt diagnosis on poor-quality images."""
            },

            "enhance_then_diagnose": {
                "tool_sequence": ["quality_assess", "enhance_image", "quality_assess", "DR_Grading"],
                "enable_conditional_enhance": True,
                "max_steps": 6,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**CONDITIONAL ENHANCEMENT:**
1. quality_assess → borderline (0.3-0.6)
2. enhance_image to improve visibility
3. quality_assess again → check if improved
4. If improved → proceed with diagnosis
5. If still poor → reject

This demonstrates adaptive strategy."""
            },

            "undiagnosable": {
                "tool_sequence": [],
                "enable_rejection": True,
                "max_steps": 1,
                "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + """
**IMMEDIATE REJECTION:**
Directly conclude as undiagnosable due to [reason].
No tool calls needed."""
            }
        }

    @classmethod
    def _get_default_strategy(cls) -> Dict:
        """默认策略"""
        return {
            "tool_sequence": ["quality_assess", "DR_Grading"],
            "enable_correction": False,
            "enable_crop_verification": False,
            "enable_cascade": False,
            "enable_parallel": False,
            "max_steps": 3,
            "system_prompt_suffix": cls.XML_FORMAT_REQUIREMENT + "Standard diagnostic workflow.",
            "correction_trigger": None,
            "cascade_rule": None
        }

    @classmethod
    def _get_default_strategy_with_metadata(cls, rtype: str, subtype: str, confidence: float) -> Dict:
        """获取带元数据的默认策略"""
        strategy = cls._get_default_strategy()
        strategy["reasoning_type"] = rtype
        strategy["reasoning_subtype"] = subtype
        strategy["confidence"] = confidence
        return strategy

    @classmethod
    def resolve_tool_dependencies(cls, requested_tools: List[str]) -> List[str]:
        """
        解析工具依赖，返回正确的执行顺序

        Args:
            requested_tools: 请求的工具列表

        Returns:
            解析依赖后的工具列表（正确顺序）
        """
        ordered = []
        for tool in requested_tools:
            # 添加依赖工具
            if tool in cls.TOOL_DEPENDENCIES:
                for dep in cls.TOOL_DEPENDENCIES[tool]:
                    if dep not in ordered and dep not in requested_tools:
                        ordered.append(dep)

            # 添加工具本身
            if tool not in ordered:
                ordered.append(tool)

        return ordered


# 测试代码
if __name__ == "__main__":
    import json
    from reasoning_classifier_V0 import ReasoningType

    logging.basicConfig(level=logging.INFO)

    # 测试各类型策略
    test_types = [
        ReasoningType(type="A", subtype="healthy", confidence=0.9),
        ReasoningType(type="B", subtype="false_dr_is_amd", confidence=0.85),
        ReasoningType(type="C", subtype="dr_to_dme", confidence=0.85),
        ReasoningType(type="D", subtype="dr_and_amd", confidence=0.9),
        ReasoningType(type="E", subtype="low_quality", confidence=0.95)
    ]

    print("=== 推理策略测试 ===\n")
    for rtype in test_types:
        strategy = ReasoningStrategy.get_strategy(rtype)

        print(f"Type {rtype.type} - {rtype.subtype}:")
        print(f"  工具序列: {strategy['tool_sequence']}")
        print(f"  最大步骤: {strategy['max_steps']}")
        print(f"  启用纠错: {strategy.get('enable_correction', False)}")
        print(f"  启用Crop: {strategy.get('enable_crop_verification', False)}")
        print()

    # 测试依赖解析
    print("=== 工具依赖解析测试 ===")
    test_sequences = [
        ["crop_roi", "DR_Grading"],
        ["dme_risk_assess", "AMD_predict"]
    ]

    for seq in test_sequences:
        resolved = ReasoningStrategy.resolve_tool_dependencies(seq)
        print(f"原始: {seq}")
        print(f"解析后: {resolved}")
        print()
