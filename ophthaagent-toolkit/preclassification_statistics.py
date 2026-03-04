#!/usr/bin/env python3
"""
预分类统计脚本
对指定文件夹中的数据集进行预分类，统计分类结果
"""
import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter

# 添加 generator 目录到路径
current_dir = Path(__file__).parent
generator_dir = current_dir / "generator"
sys.path.insert(0, str(generator_dir))

from reasoning_classifier import ReasoningType, ReasoningTypeClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> list:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_question_and_answer(item: dict) -> tuple:
    """
    从数据项中提取问题和答案
    
    Args:
        item: 数据项字典
        
    Returns:
        (question, ground_truth_answer) 元组
    """
    conversations = item.get("conversations", [])
    
    question = ""
    ground_truth_answer = ""
    
    for conv in conversations:
        if conv.get("from") == "human":
            question = conv.get("value", "")
        elif conv.get("from") == "gpt":
            ground_truth_answer = conv.get("value", "")
    
    return question, ground_truth_answer


def classify_dataset(data: list, dataset_name: str) -> dict:
    """
    对数据集进行分类
    
    Args:
        data: 数据列表
        dataset_name: 数据集名称
        
    Returns:
        分类统计结果
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始分类数据集: {dataset_name}")
    logger.info(f"{'='*60}")
    
    # 统计变量
    total_count = len(data)
    classified_count = 0
    unclassified_count = 0
    truly_unclassified_count = 0  # 真正无法分类的样本（描述中包含"无法分类"）
    
    # 分类结果统计
    type_stats = defaultdict(int)
    subtype_stats = defaultdict(int)
    
    # 未分类样本的答案标签统计
    unclassified_answers = []
    truly_unclassified_answers = []  # 真正无法分类的样本的答案标签
    
    # 分类结果详情
    classification_results = []
    
    for idx, item in enumerate(data, 1):
        # 提取问题和答案
        question, ground_truth_answer = extract_question_and_answer(item)
        
        if not question or not ground_truth_answer:
            logger.warning(f"样本 {idx}: 缺少问题或答案，跳过")
            unclassified_count += 1
            unclassified_answers.append(ground_truth_answer)
            continue
        
        # 进行分类
        try:
            reasoning_type = ReasoningTypeClassifier.classify(question, ground_truth_answer)
            
            # 检查是否为"无法分类"的样本
            is_truly_unclassified = "无法分类" in reasoning_type.description
            
            if is_truly_unclassified:
                truly_unclassified_count += 1
                truly_unclassified_answers.append(ground_truth_answer)
                # 不更新类型统计，因为这是无法分类的样本
            else:
                # 只有真正分类的样本才更新类型统计
                type_stats[reasoning_type.type] += 1
                subtype_stats[f"{reasoning_type.type}-{reasoning_type.subtype}"] += 1
                classified_count += 1
            
            # 记录分类结果
            classification_results.append({
                "id": item.get("id"),
                "type": reasoning_type.type,
                "subtype": reasoning_type.subtype,
                "confidence": reasoning_type.confidence,
                "description": reasoning_type.description,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "answer": ground_truth_answer,
                "is_truly_unclassified": is_truly_unclassified
            })
            
            if idx % 100 == 0:
                logger.info(f"已处理 {idx}/{total_count} 个样本...")
                
        except Exception as e:
            logger.error(f"样本 {idx}: 分类失败 - {e}")
            unclassified_count += 1
            unclassified_answers.append(ground_truth_answer)
    
    # 输出统计结果
    logger.info(f"\n{'='*60}")
    logger.info(f"分类统计结果: {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"总样本数: {total_count}")
    logger.info(f"已分类样本数: {classified_count} ({classified_count/total_count*100:.2f}%)")
    logger.info(f"未分类样本数: {unclassified_count} ({unclassified_count/total_count*100:.2f}%)")
    logger.info(f"真正无法分类样本数: {truly_unclassified_count} ({truly_unclassified_count/total_count*100:.2f}%)")
    
    logger.info(f"\n按类型统计:")
    for type_name in sorted(type_stats.keys()):
        count = type_stats[type_name]
        percentage = count / total_count * 100
        logger.info(f"  Type {type_name}: {count} ({percentage:.2f}%)")
    
    logger.info(f"\n按子类型统计 (前10):")
    sorted_subtypes = sorted(subtype_stats.items(), key=lambda x: x[1], reverse=True)
    for subtype, count in sorted_subtypes[:10]:
        percentage = count / total_count * 100
        logger.info(f"  {subtype}: {count} ({percentage:.2f}%)")
    
    if unclassified_answers:
        logger.info(f"\n未分类样本的答案标签统计:")
        answer_counter = Counter(unclassified_answers)
        for answer, count in answer_counter.most_common(20):
            logger.info(f"  '{answer}': {count}")
    
    if truly_unclassified_answers:
        logger.info(f"\n真正无法分类样本的答案标签统计 (前20):")
        answer_counter = Counter(truly_unclassified_answers)
        for answer, count in answer_counter.most_common(20):
            logger.info(f"  '{answer}': {count}")
    
    return {
        "dataset_name": dataset_name,
        "total_count": total_count,
        "classified_count": classified_count,
        "unclassified_count": unclassified_count,
        "truly_unclassified_count": truly_unclassified_count,
        "type_stats": dict(type_stats),
        "subtype_stats": dict(subtype_stats),
        "unclassified_answers": unclassified_answers,
        "truly_unclassified_answers": truly_unclassified_answers,
        "classification_results": classification_results
    }


def main():
    """主函数"""
    # 数据集目录
    dataset_dir = "<ANON_ABS_PATH> Train"
    
    # 获取所有 JSONL 文件
    dataset_files = []
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith('.jsonl') and not file_name.startswith('.'):
            dataset_files.append(os.path.join(dataset_dir, file_name))
    
    if not dataset_files:
        logger.error(f"未找到数据集文件在目录: {dataset_dir}")
        return
    
    logger.info(f"找到 {len(dataset_files)} 个数据集文件")
    
    # 对每个数据集进行分类
    all_results = []
    for dataset_file in sorted(dataset_files):
        dataset_name = os.path.basename(dataset_file)
        
        # 加载数据
        try:
            data = load_jsonl(dataset_file)
            logger.info(f"加载数据集 {dataset_name}: {len(data)} 个样本")
            
            # 分类
            result = classify_dataset(data, dataset_name)
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 失败: {e}")
    
    # 汇总统计
    logger.info(f"\n{'='*60}")
    logger.info(f"汇总统计")
    logger.info(f"{'='*60}")
    
    total_samples = sum(r["total_count"] for r in all_results)
    total_classified = sum(r["classified_count"] for r in all_results)
    total_unclassified = sum(r["unclassified_count"] for r in all_results)
    total_truly_unclassified = sum(r["truly_unclassified_count"] for r in all_results)
    
    logger.info(f"所有数据集总样本数: {total_samples}")
    logger.info(f"已分类样本总数: {total_classified} ({total_classified/total_samples*100:.2f}%)")
    logger.info(f"未分类样本总数: {total_unclassified} ({total_unclassified/total_samples*100:.2f}%)")
    logger.info(f"真正无法分类样本总数: {total_truly_unclassified} ({total_truly_unclassified/total_samples*100:.2f}%)")
    
    # 汇总类型统计
    logger.info(f"\n汇总按类型统计:")
    total_type_stats = defaultdict(int)
    for result in all_results:
        for type_name, count in result["type_stats"].items():
            total_type_stats[type_name] += count
    
    for type_name in sorted(total_type_stats.keys()):
        count = total_type_stats[type_name]
        percentage = count / total_samples * 100
        logger.info(f"  Type {type_name}: {count} ({percentage:.2f}%)")
    
    # 汇总未分类答案
    all_unclassified_answers = []
    for result in all_results:
        all_unclassified_answers.extend(result["unclassified_answers"])
    
    if all_unclassified_answers:
        logger.info(f"\n汇总未分类样本的答案标签统计 (前20):")
        answer_counter = Counter(all_unclassified_answers)
        for answer, count in answer_counter.most_common(20):
            logger.info(f"  '{answer}': {count}")
    
    # 汇总真正无法分类的答案
    all_truly_unclassified_answers = []
    for result in all_results:
        all_truly_unclassified_answers.extend(result["truly_unclassified_answers"])
    
    if all_truly_unclassified_answers:
        logger.info(f"\n汇总真正无法分类样本的答案标签统计 (前20):")
        answer_counter = Counter(all_truly_unclassified_answers)
        for answer, count in answer_counter.most_common(20):
            logger.info(f"  '{answer}': {count}")
    
    # 保存详细结果到文件
    output_file = os.path.join(current_dir, "preclassification_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n详细分类结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
