"""
推理类型分类器
根据question和ground_truth_answer自动分类推理类型(A-E)
"""
import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReasoningType:
    """推理类型数据类"""
    type: str  # A, B, C, D, E
    subtype: str  # 具体子类型
    confidence: float  # 置信度
    description: str = ""  # 描述


class ReasoningTypeClassifier:
    """根据ground_truth_answer和question自动分类推理类型"""

    # Type A: 标准SOP (40%)
    TYPE_A_PATTERNS = {
        "healthy": {
            "patterns": ["healthy", "normal", "no abnormalities", "健康", "正常"],
            "keywords": ["healthy", "normal"],
            "anti_patterns": ["not healthy", "abnormal","normal optic disc","abnormality"]
        },
        "pure_seg":{
            "patterns": ["hard exudates","exudates","microaneurysms","soft exudates","haemorrhages","red small dots","cotton wool spots","drusen"],
            "anti_patterns": ["diabetic retinopathy"]
        },
        
        "pure_dr": {
            "patterns": ["diabetic retinopathy","hard exudates","microaneurysms","soft exudates","haemorrhages","red small dots","cotton wool spots"],
            "anti_patterns": ["combined"]
        },
        "pure_amd": {
            "patterns": ["amd","drusen", "age-related macular", "age related macular", "macular degeneration","玻璃膜疣", "黄斑变性","normal macula"],
            "anti_patterns": ["diabetic retinopathy", "glaucoma"]
        },
        "pure_glaucoma": {
            "patterns": ["glaucoma", "increased cup disc", "cdr", "optic disc cupping","optic disc"],
            "anti_patterns": ["dr", "diabetic", "amd"]
        }
    }

    # Type B: 纠错与鉴别 (25%)
    TYPE_B_PATTERNS = {
        "dr_grading_mismatch": {
            "patterns": ["diabetic retinopathy", "hard exudates","microaneurysms","soft exudates","haemorrhages","red small dots","cotton wool spots"],
            "keywords": ["diabetic retinopathy"],
            "anti_patterns": ["normal", "healthy", "age-related macular","amd", "glaucoma"]
        },
        "false_dr_is_amd": {
            "question_patterns": ["hard exudates", "硬性渗出"],
            "answer_patterns": ["amd", "drusen", "not dr", "非糖尿病"],
            "keywords": ["macula", "central", "黄斑中心"]
        },
        "false_glaucoma": {
            "patterns": ["physiologic", "normal cup", "not glaucoma", "生理性", "正常视杯"],
            "keywords": ["large cup", "大视杯", "borderline"]
        },
        "artifact": {
            "patterns": ["artifact", "reflection", "false positive", "伪影", "反光"],
            "keywords": ["not real", "vessel wall", "血管壁"]
        },
        "low_confidence_recheck": {
            "patterns": ["uncertain", "需要复查", "low confidence", "置信度"],
            "keywords": ["crop", "detail", "细节", "zoom"]
        }
    }

    # Type C: 深度级联 (15%)
    TYPE_C_PATTERNS = {
        "dr_to_dme": {
            "patterns": ["dme", "diabetic macular edema", "macular edema"],
            "triggers": [ "dme", "diabetic macular edema", "macular edema"],
            "sequence": ["dr", "dme"]
        },
        "pdr_to_nvg": {
            "patterns": ["neovascularization", "neovascularisation","nvg", "新生血管性青光眼"],
            "triggers": ["proliferative", "new vessels"],
            "sequence": ["pdr", "glaucoma"]
        }
    }

    # Type D: 多病并发 (15%)
    TYPE_D_PATTERNS = {
        "dr_and_amd": {
            "required": ["dr", "amd"],
            "keywords": ["combined", "both", "并发", "合并"]
        },
        "dr_and_glaucoma": {
            "required": ["dr", "glaucoma"],
            "keywords": ["combined", "both", "并发", "合并"]
        },
        "multi_disease": {
            "min_disease_count": 2
        }
    }

    # Type E: 拒诊与异常 (5%)
    TYPE_E_PATTERNS = {
        "low_quality": {
            "patterns": ["poor quality","poor", "unclear", "blurry", "模糊", "质量差", "过暗"],
            "keywords": ["cannot diagnose", "无法诊断", "重新拍摄"]
        },
        "enhance_then_diagnose": {
            "patterns": ["after enhancement", "增强后", "质量提升"],
            "keywords": ["enhance", "improve", "增强"]
        },
        "undiagnosable": {
            "patterns": ["undiagnosable", "无法诊断", "insufficient"],
            "keywords": ["quality", "clarity"]
        }
    }

    @classmethod
    def classify(cls, question: str, ground_truth_answer: str) -> Optional[ReasoningType]:
        """
        分类推理类型

        Args:
            question: 问题文本
            ground_truth_answer: 标准答案

        Returns:
            ReasoningType对象，如果无法分类则返回None
        """
        q_lower = question.lower()
        a_lower = ground_truth_answer.lower()

        # 如果答案是单个字母(A/B/C/D)，尝试从question中提取对应的完整文本
        combined_text = cls._extract_answer_from_question(question, ground_truth_answer)

        # 按优先级检查 (E > D > C > A)
        # 注意: 初始化阶段不分类为Type B (纠错)，B类仅在运行工具后根据冲突动态调整
        # 这确保了所有单病种样本首先进入标准SOP流程

        # Type E: 拒诊 (最高优先级)
        type_e = cls._check_type_e(q_lower, combined_text)
        if type_e:
            return type_e

        # Type D: 多病并发 (检查是否真的有多个独立疾病)
        type_d = cls._check_type_d(q_lower, combined_text)
        if type_d:
            return type_d

        # Type C: 级联推理 (如 DR -> DME)
        type_c = cls._check_type_c(q_lower, combined_text)
        if type_c:
            return type_c

        # Type A: 标准SOP (基础诊断)
        type_a = cls._check_type_a(q_lower, combined_text)
        if type_a:
            return type_a

        # 完全无法分类，返回None（由调用方决定是否跳过）
        logger.debug(f"无法分类样本 - Question: {question[:100]}..., Answer: {ground_truth_answer[:100]}...")
        return None

    @classmethod
    def _extract_answer_from_question(cls, question: str, answer: str) -> str:
        """
        从question中提取答案对应的完整文本

        例如:
        Question: "What is shown? A: Healthy B: Mild DR C: Severe DR"
        Answer: "B"
        返回: "mild dr"

        Answer: "B,C"
        返回: "mild dr severe dr"
        """
        answer_stripped = answer.strip().upper()

        # 检查是否为多选答案 (如 "B,C,D" 或 "A,B,F")
        if ',' in answer_stripped:
            # 分割多个选项
            options = [opt.strip() for opt in answer_stripped.split(',')]
            extracted_texts = []

            for opt in options:
                if len(opt) == 1 and opt in 'ABCDEFGH':
                    # 尝试从问题中提取该选项对应的文本
                    patterns = [
                        rf"{opt}:\s*([^\n]+)",  # B: xxx (到换行符)
                        rf"{opt}\)\s*([^\n]+)",  # B) xxx (到换行符)
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, question, re.IGNORECASE)
                        if match:
                            extracted_texts.append(match.group(1).strip().lower())
                            break

            # 返回所有提取的文本，用空格连接
            if extracted_texts:
                return ' '.join(extracted_texts)
            else:
                return answer.lower()

        # 如果答案不是单个字母，直接返回小写答案
        if len(answer_stripped) != 1 or answer_stripped not in 'ABCDEFGH':
            return answer.lower()

        # 单选答案：尝试匹配模式 "B: Mild DR\n" 或 "B) Mild DR\n"
        patterns = [
            rf"{answer_stripped}:\s*([^\n]+)",  # B: xxx (到换行符)
            rf"{answer_stripped}\)\s*([^\n]+)",  # B) xxx (到换行符)
        ]

        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                return extracted.lower()

        # 如果找不到，返回原答案
        return answer.lower()

    @classmethod
    def _check_type_e(cls, q: str, a: str) -> Optional[ReasoningType]:
        """检查Type E: 拒诊与异常"""

        # 优先检查 enhance_then_diagnose (需要特殊处理)
        enhance_patterns = ["after enhancement", "增强后", "enhanced", "improve"]
        if any(p in a for p in enhance_patterns):
            # 确认不是单纯的质量差
            if "cannot diagnose" not in a and "无法诊断" not in a:
                return ReasoningType(
                    type="E",
                    subtype="enhance_then_diagnose",
                    confidence=0.90,
                    description="拒诊/异常: enhance_then_diagnose"
                )

        # 检查其他 Type E 子类型
        for subtype, config in cls.TYPE_E_PATTERNS.items():
            if subtype == "enhance_then_diagnose":
                continue  # 已经在上面处理了

            patterns = config.get("patterns", [])
            keywords = config.get("keywords", [])

            # 检查模式匹配
            pattern_match = any(p in q or p in a for p in patterns)
            keyword_match = any(k in q or k in a for k in keywords)

            if pattern_match:
                confidence = 0.95 if keyword_match else 0.85
                return ReasoningType(
                    type="E",
                    subtype=subtype,
                    confidence=confidence,
                    description=f"拒诊/异常: {subtype}"
                )

        return None

    @classmethod
    def _check_type_d(cls, q: str, a: str) -> Optional[ReasoningType]:
        """检查Type D: 多病并发"""
        # 统计疾病数量 - 只在答案文本中检查，不包括问题中的选项
        disease_keywords = {
            "dr": ["diabetic retinopathy"],
            "amd": ["amd","age-related macular","macular degeneration","黄斑变性", "drusen","age related macular"],
            "glaucoma": ["glaucoma","increased cup disc", "cdr", "optic disc cupping","abnormal optic disc"]
        }

        detected_diseases = []
        for disease, keywords in disease_keywords.items():
            # 只在答案文本中检查
            if any(kw in a for kw in keywords):
                detected_diseases.append(disease)

        # 检查是否有多个疾病
        if len(detected_diseases) >= 2:
            # 确定具体子类型
            if "dr" in detected_diseases and "amd" in detected_diseases:
                subtype = "dr_and_amd"
            elif "dr" in detected_diseases and "glaucoma" in detected_diseases:
                subtype = "dr_and_glaucoma"
            else:
                subtype = "multi_disease"

            return ReasoningType(
                type="D",
                subtype=subtype,
                confidence=0.9,
                description=f"多病并发: {', '.join(detected_diseases)}"
            )

        return None

    @classmethod
    def _check_type_c(cls, q: str, a: str) -> Optional[ReasoningType]:
        """检查Type C: 深度级联"""
        for subtype, config in cls.TYPE_C_PATTERNS.items():
            patterns = config.get("patterns", [])
            triggers = config.get("triggers", [])

            pattern_match = any(p in a for p in patterns)
            trigger_match = any(t in q or t in a for t in triggers)
    
            if pattern_match and trigger_match:
                return ReasoningType(
                    type="C",
                    subtype=subtype,
                    confidence=0.85,
                    description=f"级联推理: {subtype}"
                )

    @classmethod
    def _check_type_b(cls, q: str, a: str) -> Optional[ReasoningType]:
        """检查Type B: 纠错与鉴别"""
        for subtype, config in cls.TYPE_B_PATTERNS.items():
            # false_dr_is_amd 特殊处理
            if subtype == "false_dr_is_amd":
                q_match = any(p in q for p in config.get("question_patterns", []))
                a_match = any(p in a for p in config.get("answer_patterns", []))
                if q_match and a_match:
                    return ReasoningType(
                        type="B",
                        subtype=subtype,
                        confidence=0.88,
                        description="纠错: 假DR实为AMD"
                    )

            # 其他子类型
            else:
                patterns = config.get("patterns", [])
                keywords = config.get("keywords", [])
                anti_patterns = config.get("anti_patterns", [])

                pattern_match = any(p in q or p in a for p in patterns)
                keyword_match = any(k in q or k in a for k in keywords)
                anti_match = any(ap in a for ap in anti_patterns)

                if pattern_match and not anti_match:
                    confidence = 0.85 if keyword_match else 0.75
                    return ReasoningType(
                        type="B",
                        subtype=subtype,
                        confidence=confidence,
                        description=f"纠错鉴别: {subtype}"
                    )

        return None

    @classmethod
    def _check_type_a(cls, q: str, a: str) -> Optional[ReasoningType]:
        """检查Type A: 标准SOP"""
        for subtype, config in cls.TYPE_A_PATTERNS.items():
            patterns = config.get("patterns", [])
            anti_patterns = config.get("anti_patterns", [])
            keywords = config.get("keywords", [])

            pattern_match = any(p in q or p in a for p in patterns)
            anti_match = any(ap in a for ap in anti_patterns)

            if subtype == "healthy":
                keyword_match = any(k in a for k in keywords)
                if pattern_match and keyword_match and not anti_match:
                    return ReasoningType(
                        type="A",
                        subtype=subtype,
                        confidence=0.75,
                        description=f"标准SOP: {subtype}"
                    )
            else:
                if pattern_match and not anti_match:
                    return ReasoningType(
                        type="A",
                        subtype=subtype,
                        confidence=0.75,
                        description=f"标准SOP: {subtype}"
                    )

        return None

    @classmethod
    def get_distribution_target(cls) -> Dict[str, float]:
        """
        获取目标分布比例

        Returns:
            dict: {"A": 0.4, "B": 0.25, "C": 0.15, "D": 0.15, "E": 0.05}
        """
        return {
            "A": 0.40,  # 标准SOP
            "B": 0.25,  # 纠错鉴别
            "C": 0.15,  # 深度级联
            "D": 0.15,  # 多病并发
            "E": 0.05   # 拒诊异常
        }


# 测试代码
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    # 测试用例
    test_cases = [
        {
            "question": "What is shown in the image? A: Healthy B: Mild DR",
            "answer": "A",
            "expected": "A-healthy"
        },
        {
            "question": "What is shown in the diabetic retinopathy image",
            "answer": "microaneurysms",
            "expected": "A-pure_seg"
        },
        {
            "question": "Does the image show hard exudates?",
            "answer": "Yes, but they are actually AMD drusen, not DR exudates",
            "expected": "B-false_dr_is_amd"
        },
        {
            "question": "What is the diagnosis?",
            "answer": "Severe NPDR with diabetic macular edema",
            "expected": "C-dr_to_dme"
        },
        {
            "question": "What conditions are present?",
            "answer": "Patient has both DR and AMD",
            "expected": "D-dr_and_amd"
        },
        {
            "question": "Diagnose this image",
            "answer": "Image quality is too poor to diagnose",
            "expected": "E-low_quality"
        }
    ]

    print("=== 推理类型分类测试 ===\n")
    for i, case in enumerate(test_cases, 1):
        result = ReasoningTypeClassifier.classify(case["question"], case["answer"])
        actual = f"{result.type}-{result.subtype}"

        print(f"Case {i}:")
        print(f"  Question: {case['question']}")
        print(f"  Answer: {case['answer']}")
        print(f"  Expected: {case['expected']}")
        print(f"  Actual: {actual}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Match: {'✓' if actual == case['expected'] else '✗'}")
        print()

    # 分布统计
    print("=== 目标分布 ===")
    distribution = ReasoningTypeClassifier.get_distribution_target()
    print(json.dumps(distribution, indent=2))
