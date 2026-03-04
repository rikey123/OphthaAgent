"""
VQA数据集加载器
"""
import json
import csv
import ast
import os
from typing import List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CoTExample:
    """CoT数据示例"""
    question: str
    image_path: str
    ground_truth_answer: str
    trajectory: Optional[str] = None
    error: Optional[str] = None


class DatasetLoader:
    """VQA数据集加载器"""
    
    def load_dataset(self, dataset_path: str, image_root: Optional[str] = None) -> List[CoTExample]:
        """
        加载VQA数据集

        支持的数据格式：
        1. JSON Lines格式 (.jsonl) - 每行一个JSON对象
        2. JSON数组格式 (.json) - 包含示例数组的JSON文件
        3. CSV格式 (.csv) - 预标注数据，包含image, attributes, categories列

        每个示例应包含以下字段：
        - question: 问题文本 (CSV格式自动生成)
        - image_path 或 image: 图片路径
        - answer 或 ground_truth_answer: 标准答案 (CSV格式从categories列提取)

        Args:
            dataset_path: 数据集文件路径
            image_root: 图片根目录 (可选)

        Returns:
            CoTExample列表
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")

        examples = []

        # 判断文件格式
        if dataset_path.endswith('.jsonl'):
            examples = self._load_jsonl(dataset_path, image_root=image_root)
        elif dataset_path.endswith('.json'):
            examples = self._load_json(dataset_path, image_root=image_root)
        elif dataset_path.endswith('.csv'):
            examples = self._load_csv(dataset_path, image_root=image_root)
        else:
            raise ValueError(f"不支持的文件格式: {dataset_path}。支持: .json, .jsonl, .csv")

        logger.info(f"成功加载 {len(examples)} 个示例")
        return examples
    
    def _load_jsonl(self, file_path: str, image_root: Optional[str] = None) -> List[CoTExample]:
        """加载JSON Lines格式"""
        examples = []
        dataset_dir = os.path.dirname(os.path.abspath(file_path))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    example = self._parse_example(data, line_num, dataset_dir=dataset_dir, image_root=image_root)
                    if example:
                        examples.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行JSON解析失败: {e}")
                    continue
        
        return examples
    
    def _load_json(self, file_path: str, image_root: Optional[str] = None) -> List[CoTExample]:
        """加载JSON数组格式"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        dataset_dir = os.path.dirname(os.path.abspath(file_path))
        
        # 支持两种格式：
        # 1. 直接是数组
        # 2. 包含"data"或"examples"字段的对象
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get('data', data.get('examples', data.get('items', [])))
            if not items:
                # 尝试直接解析为单个示例
                example = self._parse_example(data, 0, dataset_dir=dataset_dir, image_root=image_root)
                if example:
                    return [example]
        else:
            raise ValueError(f"不支持的JSON格式: {file_path}")
        
        for idx, item in enumerate(items):
            example = self._parse_example(item, idx, dataset_dir=dataset_dir, image_root=image_root)
            if example:
                examples.append(example)
        
        return examples

    def _load_csv(self, file_path: str, image_root: Optional[str] = None) -> List[CoTExample]:
        """
        加载CSV预标注数据格式

        CSV格式:
        - 列: , image, atributes, categories
        - image: 图片相对路径
        - attributes: 属性列表 (通常为空 [])
        - categories: 疾病/特征类别列表,如 "['diabetic retinopathy', 'laser scar']"

        生成问题格式:
        - 单个类别: "What is shown in this fundus image?"
        - 多个类别: "What conditions are present in this fundus image?"

        Args:
            file_path: CSV文件路径
            image_root: 图片根目录

        Returns:
            CoTExample列表
        """
        examples = []
        dataset_dir = os.path.dirname(os.path.abspath(file_path))

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):  # start=2因为第1行是标题
                try:
                    # 提取图片路径
                    image_path = row.get('image', '').strip()
                    if not image_path:
                        logger.warning(f"第 {row_num} 行缺少图片路径")
                        continue

                    # 处理图片路径
                    if not os.path.isabs(image_path):
                        if image_root:
                            candidate = os.path.join(image_root, image_path)
                            if os.path.exists(candidate):
                                image_path = candidate
                        elif dataset_dir:
                            candidate = os.path.join(dataset_dir, image_path)
                            if os.path.exists(candidate):
                                image_path = candidate

                    # 提取categories列表
                    categories_str = row.get('categories', '[]').strip()
                    try:
                        # 使用ast.literal_eval安全解析Python列表字符串
                        categories = ast.literal_eval(categories_str)
                        if not isinstance(categories, list):
                            categories = [categories_str]
                    except (ValueError, SyntaxError):
                        logger.warning(f"第 {row_num} 行categories格式错误: {categories_str}")
                        continue

                    if not categories:
                        logger.warning(f"第 {row_num} 行categories为空")
                        continue

                    # 生成问题文本
                    if len(categories) == 1:
                        question = "What is shown in this fundus image?"
                    else:
                        question = "What conditions are present in this fundus image?"

                    # 生成答案文本 (使用逗号连接所有类别)
                    ground_truth_answer = ', '.join(categories)

                    example = CoTExample(
                        question=question,
                        image_path=image_path,
                        ground_truth_answer=ground_truth_answer
                    )
                    examples.append(example)

                except Exception as e:
                    logger.warning(f"第 {row_num} 行解析失败: {e}")
                    continue

        return examples

    def _parse_example(self, data: dict, line_num: int, dataset_dir: Optional[str] = None, image_root: Optional[str] = None) -> CoTExample:
        """
        解析单个示例
        
        支持两种格式：
        1. 标准格式：包含 question, image_path/image, answer/ground_truth_answer 字段
        2. 对话格式：包含 conversations 数组，其中包含 human 和 gpt 的对话
        
        Args:
            data: 示例数据字典
            line_num: 行号（用于错误提示）
            
        Returns:
            CoTExample对象，如果解析失败返回None
        """
        # 尝试解析对话格式（conversations数组）
        if 'conversations' in data and isinstance(data['conversations'], list):
            question = None
            ground_truth_answer = None

            # 从conversations数组中提取问题和答案
            for conv in data['conversations']:
                if isinstance(conv, dict):
                    from_role = conv.get('from', '').lower()
                    value = conv.get('value', '')

                    if from_role == 'human' and not question:
                        question = value
                    elif from_role in ['gpt', 'assistant', 'model'] and not ground_truth_answer:
                        ground_truth_answer = value

            if not question:
                logger.warning(f"第 {line_num} 行conversations中缺少human的问题")
                return None

            if not ground_truth_answer:
                logger.warning(f"第 {line_num} 行conversations中缺少gpt的答案")
                return None

            # 对话格式解析成功，跳过标准格式检查
        else:
            # 标准格式：直接提取字段
            question = data.get('question') or data.get('Question') or data.get('q')
            ground_truth_answer = (
                data.get('ground_truth_answer') or
                data.get('correct_answer') or  # 新增：支持correct_answer字段
                data.get('answer') or
                data.get('Answer') or
                data.get('gt_answer') or
                data.get('a')
            )

            if not question:
                logger.warning(f"第 {line_num} 行缺少问题字段")
                return None

            if not ground_truth_answer:
                logger.warning(f"第 {line_num} 行缺少答案字段")
                return None
        
        # 提取图片路径（支持多种字段名）
        raw_image_path = (
            data.get('image_path') or
            data.get('image') or
            data.get('Image') or
            data.get('img_path') or
            data.get('image_file')
        )
        if not raw_image_path:
            logger.warning(f"第 {line_num} 行缺少图片路径字段")
            return None

        # 处理 image 字段为列表的情况（如 conversations 格式）
        if isinstance(raw_image_path, list):
            if len(raw_image_path) > 0:
                raw_image_path = raw_image_path[0]
            else:
                logger.warning(f"第 {line_num} 行的 image 列表为空")
                return None

        image_path = raw_image_path
        if not os.path.isabs(image_path):
            # 优先使用传入的 image_root
            if image_root:
                candidate = os.path.join(image_root, image_path)
                if os.path.exists(candidate):
                    image_path = candidate
            # 其次使用数据集所在目录
            if not os.path.isabs(image_path) and dataset_dir:
                candidate = os.path.join(dataset_dir, image_path)
                if os.path.exists(candidate):
                    image_path = candidate
        
        # 注意：图片路径保持原样，不进行自动解析
        # 如果需要处理相对路径，应该在调用时传入完整路径
        
        return CoTExample(
            question=question,
            image_path=image_path,
            ground_truth_answer=ground_truth_answer
        )

