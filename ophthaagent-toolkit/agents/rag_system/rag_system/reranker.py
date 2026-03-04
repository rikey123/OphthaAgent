"""
Reranker模块
用于对检索结果进行重新排序，提高检索质量
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class BaseReranker(ABC):
    """Reranker抽象基类"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重新排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top-k个结果
        
        Returns:
            [(doc_index, score), ...] 按分数降序排列
        """
        pass


class LocalReranker(BaseReranker):
    """本地Reranker模型"""
    
    def __init__(
        self,
        model_path: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        初始化本地Reranker模型
        
        Args:
            model_path: 模型路径（HuggingFace模型名或本地路径）
            device: 设备 ("cpu" 或 "cuda")
            batch_size: 批处理大小
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
        
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        
        print(f"正在加载Reranker模型: {model_path}")
        self.model = CrossEncoder(model_path, device=device)
        print(f"Reranker模型加载完成")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """对文档进行重新排序"""
        if not documents:
            return []
        
        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]
        
        # 计算相关性分数
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # 创建 (索引, 分数) 对并排序
        scored_docs = [(i, float(score)) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top-k
        if top_k is not None:
            scored_docs = scored_docs[:top_k]
        
        return scored_docs


class APIReranker(BaseReranker):
    """API Reranker（支持硅基流动等）"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        api_key: str = None,
        base_url: str = "https://api.siliconflow.cn/v1",
        batch_size: int = 100
    ):
        """
        初始化API Reranker
        
        Args:
            model_name: 模型名称
            api_key: API密钥（如果为None则从环境变量读取）
            base_url: API基础URL
            batch_size: 批处理大小
        """
        try:
            import requests
        except ImportError:
            raise ImportError("请安装requests: pip install requests")
        
        import os
        
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.batch_size = batch_size
        self.requests = requests
        
        # 获取API密钥
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
            if not self.api_key:
                raise ValueError("必须提供api_key或设置OPENAI_API_KEY/SILICONFLOW_API_KEY环境变量")
        
        print(f"API Reranker初始化完成: {model_name}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """对文档进行重新排序"""
        if not documents:
            return []
        
        # 准备请求数据
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k if top_k is not None else len(documents)
        }
        
        # 发送请求
        response = self.requests.post(url, json=data, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Rerank API请求失败: {response.status_code} {response.text}")
        
        result = response.json()
        
        # 解析结果
        scored_docs = []
        for item in result.get("results", []):
            index = item.get("index")
            score = item.get("relevance_score")
            scored_docs.append((index, float(score)))
        
        return scored_docs


class CohereReranker(BaseReranker):
    """Cohere Reranker API"""
    
    def __init__(
        self,
        model_name: str = "rerank-multilingual-v3.0",
        api_key: str = None
    ):
        """
        初始化Cohere Reranker
        
        Args:
            model_name: 模型名称
            api_key: Cohere API密钥
        """
        try:
            import cohere
        except ImportError:
            raise ImportError("请安装cohere: pip install cohere")
        
        import os
        
        self.model_name = model_name
        
        # 获取API密钥
        if api_key:
            self.client = cohere.Client(api_key)
        else:
            api_key = os.environ.get("COHERE_API_KEY")
            if not api_key:
                raise ValueError("必须提供api_key或设置COHERE_API_KEY环境变量")
            self.client = cohere.Client(api_key)
        
        print(f"Cohere Reranker初始化完成: {model_name}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """对文档进行重新排序"""
        if not documents:
            return []
        
        # 调用Cohere API
        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=top_k if top_k is not None else len(documents)
        )
        
        # 解析结果
        scored_docs = []
        for item in response.results:
            scored_docs.append((item.index, float(item.relevance_score)))
        
        return scored_docs


def create_reranker(config) -> BaseReranker:
    """
    工厂函数：根据配置创建Reranker
    
    Args:
        config: RerankerConfig对象
    
    Returns:
        BaseReranker实例
    """
    reranker_type = config.type.lower()
    
    if reranker_type == "local":
        if not config.model_path:
            raise ValueError("本地Reranker需要指定model_path")
        
        return LocalReranker(
            model_path=config.model_path,
            device=config.device,
            batch_size=config.batch_size
        )
    
    elif reranker_type == "api":
        return APIReranker(
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            batch_size=config.batch_size
        )
    
    elif reranker_type == "cohere":
        return CohereReranker(
            model_name=config.model_name,
            api_key=config.api_key
        )
    
    else:
        raise ValueError(f"不支持的Reranker类型: {reranker_type}")

