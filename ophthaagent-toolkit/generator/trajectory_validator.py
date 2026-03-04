"""
轨迹验证工具
验证生成的推理轨迹质量
"""
import json
import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    xml_valid: bool
    tool_results_match: bool
    reasoning_coherent: bool
    correction_demonstrated: bool
    ground_truth_reached: bool
    quality_score: float
    issues: List[str]


class TrajectoryValidator:
    """轨迹质量验证器"""

    @staticmethod
    def validate(
        trajectory: str,
        tool_execution_log: List[Dict],
        ground_truth: str,
        reasoning_type: str = None
    ) -> ValidationResult:
        """
        验证轨迹质量

        Args:
            trajectory: 生成的轨迹文本
            tool_execution_log: 工具执行日志
            ground_truth: 标准答案
            reasoning_type: 推理类型（用于特定验证）

        Returns:
            ValidationResult对象
        """
        issues = []

        # 1. XML格式验证
        xml_valid = TrajectoryValidator._validate_xml_format(trajectory, issues)

        # 2. 工具结果匹配验证
        tool_results_match = TrajectoryValidator._validate_tool_results(
            trajectory,
            tool_execution_log,
            issues
        )

        # 3. 推理连贯性验证
        reasoning_coherent = TrajectoryValidator._validate_reasoning_coherence(
            trajectory,
            issues
        )

        # 4. 纠错验证（仅Type B）
        correction_demonstrated = False
        if reasoning_type and reasoning_type.startswith("B"):
            correction_demonstrated = TrajectoryValidator._validate_correction(
                trajectory,
                issues
            )

        # 5. 答案正确性验证
        ground_truth_reached = TrajectoryValidator._validate_answer(
            trajectory,
            ground_truth,
            issues
        )

        # 计算质量分数
        quality_score = TrajectoryValidator._calculate_quality_score(
            xml_valid,
            tool_results_match,
            reasoning_coherent,
            correction_demonstrated if reasoning_type and reasoning_type.startswith("B") else True,
            ground_truth_reached
        )

        return ValidationResult(
            xml_valid=xml_valid,
            tool_results_match=tool_results_match,
            reasoning_coherent=reasoning_coherent,
            correction_demonstrated=correction_demonstrated,
            ground_truth_reached=ground_truth_reached,
            quality_score=quality_score,
            issues=issues
        )

    @staticmethod
    def _validate_xml_format(trajectory: str, issues: List[str]) -> bool:
        """验证XML格式"""
        if not trajectory:
            issues.append("轨迹为空")
            return False

        # 检查必需标签
        required_tags = ["<think>", "</think>", "<answer>", "</answer>"]
        for tag in required_tags:
            if tag not in trajectory:
                issues.append(f"缺少必需标签: {tag}")
                return False

        # 检查tool_call和observation配对
        tool_call_count = trajectory.count("<tool_call>")
        observation_count = trajectory.count("<observation>")

        if tool_call_count != observation_count:
            issues.append(f"tool_call ({tool_call_count}) 和 observation ({observation_count}) 数量不匹配")
            return False

        # 检查顺序：必须以<think>开头
        if not trajectory.strip().startswith("<think>"):
            issues.append("轨迹未以<think>开头")
            return False

        return True

    @staticmethod
    def _validate_tool_results(
        trajectory: str,
        tool_execution_log: List[Dict],
        issues: List[str]
    ) -> bool:
        """验证工具结果是否被正确使用"""
        if not tool_execution_log:
            return True

        # 提取observation块中的内容
        observation_pattern = r'<observation>(.*?)</observation>'
        observations = re.findall(observation_pattern, trajectory, re.DOTALL)

        # 检查每个observation是否包含真实工具结果的关键信息
        matches = 0
        for log in tool_execution_log:
            tool_name = log["tool_name"]
            result = log["result"]

            # 检查是否有observation提到这个工具的结果
            for obs in observations:
                # 检查是否包含关键字段
                if "status" in result and result["status"] in obs:
                    matches += 1
                    break

        match_rate = matches / len(tool_execution_log) if tool_execution_log else 0

        if match_rate < 0.5:
            issues.append(f"工具结果匹配率过低: {match_rate:.1%}")
            return False

        return True

    @staticmethod
    def _validate_reasoning_coherence(trajectory: str, issues: List[str]) -> bool:
        """验证推理连贯性"""
        # 提取所有think块
        think_pattern = r'<think>(.*?)</think>'
        thinks = re.findall(think_pattern, trajectory, re.DOTALL)

        if len(thinks) == 0:
            issues.append("没有找到think块")
            return False

        # 检查每个think块是否有实质内容（不只是空白）
        empty_thinks = sum(1 for t in thinks if len(t.strip()) < 20)

        if empty_thinks > len(thinks) * 0.3:
            issues.append(f"过多空洞的think块: {empty_thinks}/{len(thinks)}")
            return False

        # 检查是否有常见的推理关键词
        reasoning_keywords = [
            "visual", "observe", "analysis", "hypothesis", "plan",
            "观察", "分析", "假设", "计划", "发现"
        ]

        keyword_found = any(kw in trajectory.lower() for kw in reasoning_keywords)

        if not keyword_found:
            issues.append("缺少推理关键词")
            return False

        return True

    @staticmethod
    def _validate_correction(trajectory: str, issues: List[str]) -> bool:
        """验证纠错过程（仅Type B）"""
        # 检查是否有纠错关键词
        correction_keywords = [
            "however", "but", "correction", "revise", "actually",
            "但是", "然而", "纠正", "修正", "实际上", "重新评估"
        ]

        has_correction = any(kw in trajectory.lower() for kw in correction_keywords)

        if not has_correction:
            issues.append("Type B轨迹应包含纠错过程")
            return False

        # 检查是否有多个工具调用（纠错通常需要多步）
        tool_call_count = trajectory.count("<tool_call>")

        if tool_call_count < 2:
            issues.append("Type B轨迹应包含多个工具调用")
            return False

        return True

    @staticmethod
    def _validate_answer(trajectory: str, ground_truth: str, issues: List[str]) -> bool:
        """验证答案正确性"""
        # 提取answer块
        answer_pattern = r'<answer>(.*?)</answer>'
        answers = re.findall(answer_pattern, trajectory, re.DOTALL)

        if not answers:
            issues.append("没有找到answer块")
            return False

        final_answer = answers[-1].lower()
        gt_lower = ground_truth.lower()

        # 简单的包含检查（可以改进为更智能的匹配）
        # 提取ground truth中的关键词
        gt_keywords = re.findall(r'\b\w+\b', gt_lower)

        # 检查关键词出现率
        keyword_matches = sum(1 for kw in gt_keywords if len(kw) > 3 and kw in final_answer)
        match_rate = keyword_matches / len(gt_keywords) if gt_keywords else 0

        if match_rate < 0.3:
            issues.append(f"答案与ground truth匹配度过低: {match_rate:.1%}")
            return False

        return True

    @staticmethod
    def _calculate_quality_score(
        xml_valid: bool,
        tool_results_match: bool,
        reasoning_coherent: bool,
        correction_ok: bool,
        answer_correct: bool
    ) -> float:
        """计算质量分数（0-1）"""
        weights = {
            "xml_valid": 0.2,
            "tool_results_match": 0.25,
            "reasoning_coherent": 0.25,
            "correction_ok": 0.1,
            "answer_correct": 0.2
        }

        score = (
            weights["xml_valid"] * (1.0 if xml_valid else 0.0) +
            weights["tool_results_match"] * (1.0 if tool_results_match else 0.0) +
            weights["reasoning_coherent"] * (1.0 if reasoning_coherent else 0.0) +
            weights["correction_ok"] * (1.0 if correction_ok else 0.0) +
            weights["answer_correct"] * (1.0 if answer_correct else 0.0)
        )

        return round(score, 3)


def validate_results_file(input_file: str, output_report: str = None):
    """验证整个结果文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    validation_report = {
        "total": len(results),
        "validated": 0,
        "avg_quality_score": 0.0,
        "pass_rate": 0.0,
        "issue_summary": {},
        "details": []
    }

    total_score = 0
    pass_count = 0

    for i, result in enumerate(results):
        if not result.get("trajectory"):
            continue

        validation = TrajectoryValidator.validate(
            trajectory=result["trajectory"],
            tool_execution_log=result.get("tool_execution_log", []),
            ground_truth=result["ground_truth_answer"],
            reasoning_type=result.get("metadata", {}).get("reasoning_type")
        )

        validation_report["validated"] += 1
        total_score += validation.quality_score

        if validation.quality_score >= 0.7:
            pass_count += 1

        # 统计问题
        for issue in validation.issues:
            validation_report["issue_summary"][issue] = \
                validation_report["issue_summary"].get(issue, 0) + 1

        # 保存详细信息
        validation_report["details"].append({
            "index": i,
            "question": result["question"][:100],
            "quality_score": validation.quality_score,
            "xml_valid": validation.xml_valid,
            "tool_results_match": validation.tool_results_match,
            "reasoning_coherent": validation.reasoning_coherent,
            "issues": validation.issues
        })

    # 计算平均分
    if validation_report["validated"] > 0:
        validation_report["avg_quality_score"] = round(
            total_score / validation_report["validated"], 3
        )
        validation_report["pass_rate"] = round(
            pass_count / validation_report["validated"], 3
        )

    # 保存报告
    if output_report is None:
        output_report = input_file.replace(".json", "_validation_report.json")

    with open(output_report, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)

    logger.info(f"验证报告已保存: {output_report}")

    return validation_report


# CLI工具
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="验证CoT轨迹质量")
    parser.add_argument("input", help="输入结果文件")
    parser.add_argument("--output", help="输出验证报告文件（可选）")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"文件不存在: {args.input}")
        sys.exit(1)

    report = validate_results_file(args.input, args.output)

    print("\n" + "=" * 60)
    print("验证报告摘要")
    print("=" * 60)
    print(f"总样本数: {report['total']}")
    print(f"已验证: {report['validated']}")
    print(f"平均质量分: {report['avg_quality_score']:.3f}")
    print(f"通过率: {report['pass_rate']:.1%} (≥0.7)")

    print("\n常见问题:")
    for issue, count in sorted(report['issue_summary'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {issue}: {count}次")

    print("\n" + "=" * 60)
