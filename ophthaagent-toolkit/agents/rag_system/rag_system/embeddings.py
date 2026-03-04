"""
嵌入模型模块
支持本地Transformer模型和OpenAI API
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from tqdm import tqdm


class BaseEmbedding(ABC):
    """嵌入模型抽象基类"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """对文档列表进行嵌入"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """对查询文本进行嵌入"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入向量的维度"""
        pass


class LocalEmbedding(BaseEmbedding):
    """本地Transformer模型嵌入"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True
    ):
        """
        初始化本地嵌入模型
        
        Args:
            model_path: 模型路径（可以是HuggingFace模型名或本地路径）
            device: 设备 ("cpu" 或 "cuda")
            batch_size: 批处理大小
            max_length: 最大序列长度
            normalize_embeddings: 是否归一化嵌入向量
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
        
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        
        print(f"正在加载本地模型: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        print(f"模型加载完成，嵌入维度: {self._dimension}")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """批量嵌入文档"""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_query(self, text: str) -> np.ndarray:
        """嵌入单个查询"""
        embedding = self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embedding[0]
    
    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI API嵌入"""

    def __init__(
            self,
            model_name: str = "text-embedding-3-small",
            api_key: str = None,
            batch_size: int = 100
    ):
        """
        初始化OpenAI嵌入模型

        Args:
            model_name: OpenAI模型名称
            api_key: API密钥（如果为None则从环境变量读取）
            batch_size: 批处理大小
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        self.model_name = model_name
        self.batch_size = batch_size

        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # 从环境变量读取

        # 获取嵌入维度
        self._dimension = self._get_dimension()
        print(f"API嵌入模型初始化完成: {model_name}, 嵌入维度: {self._dimension}")

    def _get_dimension(self) -> int:
        """获取嵌入维度"""
        # OpenAI模型的维度
        openai_dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }

        # 硅基流动模型的维度
        siliconflow_dimension_map = {
            "BAAI/bge-large-zh-v1.5": 1024,
            "BAAI/bge-large-en-v1.5": 1024,
            "netease-youdao/bce-embedding-base_v1": 768,
            "BAAI/bge-m3": 1024,
            "Pro/BAAI/bge-m3": 1024,
            "Qwen/Qwen3-Embedding-8B": 4096,
            "Qwen/Qwen3-Embedding-4B": 2560,
            "Qwen/Qwen3-Embedding-0.6B": 1024
        }

        # 先检查硅基流动模型
        if self.model_name in siliconflow_dimension_map:
            return siliconflow_dimension_map[self.model_name]

        # 再检查OpenAI模型
        if self.model_name in openai_dimension_map:
            return openai_dimension_map[self.model_name]

        # 如果都不在，尝试实际调用获取维度
        try:
            print(f"未知模型 {self.model_name}，尝试实际调用获取维度...")
            response = self.client.embeddings.create(
                model=self.model_name,
                input=["test"]
            )
            dim = len(response.data[0].embedding)
            print(f"检测到维度: {dim}")
            return dim
        except Exception as e:
            print(f"无法获取维度，使用默认值1024: {e}")
            return 1024

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """批量嵌入文档"""
        if not texts:
            return np.array([])

        embeddings = []

        # 分批处理
        for i in tqdm(range(0, len(texts), self.batch_size), desc="嵌入文档"):
            batch = texts[i:i + self.batch_size]

            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        """嵌入单个查询"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text]
        )

        return np.array(response.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedding(config) -> BaseEmbedding:
    """
    工厂函数：根据配置创建嵌入模型
    
    Args:
        config: EmbeddingConfig对象
    
    Returns:
        BaseEmbedding实例
    """
    if config.type == "local":
        if not config.model_path:
            raise ValueError("本地模型需要指定model_path")
        
        return LocalEmbedding(
            model_path=config.model_path,
            device=config.device,
            batch_size=config.batch_size,
            max_length=config.max_length,
            normalize_embeddings=config.normalize_embeddings
        )
    
    elif config.type == "openai":
        if not config.model_name:
            config.model_name = "text-embedding-3-small"
        
        return OpenAIEmbedding(
            model_name=config.model_name,
            batch_size=config.batch_size
        )
    
    else:
        raise ValueError(f"不支持的嵌入类型: {config.type}")

