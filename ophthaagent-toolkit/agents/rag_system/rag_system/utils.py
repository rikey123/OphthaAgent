"""
工具函数模块
"""
import os
from typing import List, Dict, Any


def format_retrieval_results(results: List[Dict[str, Any]], max_text_length: int = 200) -> str:
    """
    格式化检索结果为可读字符串
    
    Args:
        results: 检索结果列表
        max_text_length: 文本最大显示长度
    
    Returns:
        格式化的字符串
    """
    if not results:
        return "没有找到相关结果"
    
    output = []
    output.append(f"找到 {len(results)} 个相关结果:\n")
    
    for i, result in enumerate(results, 1):
        text = result["text"]
        score = result["score"]
        metadata = result["metadata"]
        
        # 截断文本
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        
        output.append(f"--- 结果 {i} (相似度: {score:.4f}) ---")
        output.append(f"文本: {text}")
        output.append(f"来源: {metadata.get('filename', 'unknown')}")
        output.append(f"分块策略: {metadata.get('chunking_strategy', 'unknown')}")
        output.append("")
    
    return "\n".join(output)


def print_retrieval_results(results: List[Dict[str, Any]], max_text_length: int = 200):
    """
    打印检索结果
    
    Args:
        results: 检索结果列表
        max_text_length: 文本最大显示长度
    """
    print(format_retrieval_results(results, max_text_length))


def create_sample_config(output_path: str, config_type: str = "local"):
    """
    创建示例配置文件
    
    Args:
        output_path: 输出路径
        config_type: 配置类型 ("local" 或 "openai")
    """
    import json
    
    if config_type == "local":
        config = {
            "embedding": {
                "type": "local",
                "model_path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "device": "cpu",
                "batch_size": 32,
                "max_length": 512,
                "normalize_embeddings": True
            },
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? "],
                "semantic_threshold": 0.5,
                "hybrid_strategies": ["sentence", "fixed"],
                "hybrid_weights": [0.6, 0.4]
            },
            "vector_store": {
                "type": "faiss",
                "index_path": "./rag_index",
                "index_type": "Flat",
                "metric": "cosine",
                "nlist": 100,
                "nprobe": 10
            },
            "pdf_directory": "./sample_pdfs",
            "recursive": True,
            "top_k": 5,
            "score_threshold": None
        }
    
    elif config_type == "openai":
        config = {
            "embedding": {
                "type": "openai",
                "model_name": "text-embedding-3-small",
                "batch_size": 100
            },
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? "]
            },
            "vector_store": {
                "type": "faiss",
                "index_path": "./rag_index",
                "index_type": "Flat",
                "metric": "cosine"
            },
            "pdf_directory": "./sample_pdfs",
            "recursive": True,
            "top_k": 5,
            "score_threshold": None
        }
    
    else:
        raise ValueError(f"不支持的配置类型: {config_type}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"示例配置已保存到: {output_path}")

