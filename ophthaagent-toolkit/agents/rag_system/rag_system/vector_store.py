"""
向量数据库模块
支持FAISS，可扩展到其他向量数据库
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pickle
import os


class BaseVectorStore(ABC):
    """向量数据库抽象基类"""
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """添加向量"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """搜索相似向量"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存索引"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载索引"""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS向量数据库实现"""
    
    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        metric: str = "cosine",
        nlist: int = 100
    ):
        """
        初始化FAISS向量存储
        
        Args:
            dimension: 向量维度
            index_type: 索引类型 ("Flat", "IVFFlat", "HNSW")
            metric: 距离度量 ("cosine", "l2", "ip")
            nlist: IVF索引的聚类中心数量
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("请安装faiss: pip install faiss-cpu 或 faiss-gpu")
        
        self.faiss = faiss
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        
        # 创建索引
        self.index = self._create_index()
        
        # 存储文本和元数据
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        
        print(f"FAISS索引初始化完成: type={index_type}, metric={metric}, dimension={dimension}")
    
    def _create_index(self):
        """创建FAISS索引"""
        # 根据度量类型选择索引
        if self.metric == "cosine":
            # 余弦相似度：先归一化，再用内积
            if self.index_type == "Flat":
                index = self.faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IVFFlat":
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, self.faiss.METRIC_INNER_PRODUCT)
            elif self.index_type == "HNSW":
                index = self.faiss.IndexHNSWFlat(self.dimension, 32, self.faiss.METRIC_INNER_PRODUCT)
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        elif self.metric == "l2":
            if self.index_type == "Flat":
                index = self.faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IVFFlat":
                quantizer = self.faiss.IndexFlatL2(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, self.faiss.METRIC_L2)
            elif self.index_type == "HNSW":
                index = self.faiss.IndexHNSWFlat(self.dimension, 32, self.faiss.METRIC_L2)
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        elif self.metric == "ip":
            # 内积
            if self.index_type == "Flat":
                index = self.faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IVFFlat":
                quantizer = self.faiss.IndexFlatIP(self.dimension)
                index = self.faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, self.faiss.METRIC_INNER_PRODUCT)
            elif self.index_type == "HNSW":
                index = self.faiss.IndexHNSWFlat(self.dimension, 32, self.faiss.METRIC_INNER_PRODUCT)
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        else:
            raise ValueError(f"不支持的度量类型: {self.metric}")
        
        return index
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        添加向量到索引
        
        Args:
            vectors: 向量数组 (n, dimension)
            texts: 文本列表
            metadatas: 元数据列表
        """
        if len(vectors) != len(texts) or len(vectors) != len(metadatas):
            raise ValueError("vectors, texts, metadatas的长度必须相同")
        
        # 确保向量是float32类型
        vectors = vectors.astype(np.float32)
        
        # 如果使用余弦相似度，需要归一化
        if self.metric == "cosine":
            self.faiss.normalize_L2(vectors)
        
        # 如果是IVF索引且未训练，先训练
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            print(f"训练IVF索引 (nlist={self.nlist})...")
            self.index.train(vectors)
            print("训练完成")
        
        # 添加向量
        self.index.add(vectors)
        
        # 保存文本和元数据
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        print(f"已添加 {len(vectors)} 个向量，总数: {self.index.ntotal}")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量 (dimension,)
            top_k: 返回top-k个结果
        
        Returns:
            [(text, score, metadata), ...] 列表
        """
        if self.index.ntotal == 0:
            return []
        
        # 确保查询向量是正确的形状和类型
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        
        # 如果使用余弦相似度，需要归一化
        if self.metric == "cosine":
            self.faiss.normalize_L2(query_vector)
        
        # 如果是IVF索引，设置nprobe
        if self.index_type == "IVFFlat":
            self.index.nprobe = min(10, self.nlist)
        
        # 搜索
        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, top_k)
        
        # 构建结果
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS返回-1表示无效结果
                continue
            
            # 转换距离为相似度分数
            if self.metric == "cosine" or self.metric == "ip":
                score = float(dist)  # 内积，越大越相似
            else:  # l2
                score = float(1.0 / (1.0 + dist))  # 转换为相似度
            
            results.append((
                self.texts[idx],
                score,
                self.metadatas[idx]
            ))
        
        return results
    
    def save(self, path: str):
        """
        保存索引和元数据
        
        Args:
            path: 保存目录路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存FAISS索引
        index_path = os.path.join(path, "index.faiss")
        self.faiss.write_index(self.index, index_path)
        
        # 保存文本和元数据
        metadata_path = os.path.join(path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                "texts": self.texts,
                "metadatas": self.metadatas,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "nlist": self.nlist
            }, f)
        
        print(f"索引已保存到: {path}")
    
    def load(self, path: str):
        """
        加载索引和元数据
        
        Args:
            path: 索引目录路径
        """
        # 加载FAISS索引
        index_path = os.path.join(path, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        self.index = self.faiss.read_index(index_path)
        
        # 加载文本和元数据
        metadata_path = os.path.join(path, "metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        self.texts = data["texts"]
        self.metadatas = data["metadatas"]
        self.dimension = data["dimension"]
        self.index_type = data["index_type"]
        self.metric = data["metric"]
        self.nlist = data.get("nlist", 100)
        
        print(f"索引已加载: {self.index.ntotal} 个向量")
    
    def clear(self):
        """清空索引"""
        self.index = self._create_index()
        self.texts = []
        self.metadatas = []
        print("索引已清空")


def create_vector_store(config, dimension: int) -> BaseVectorStore:
    """
    工厂函数：根据配置创建向量数据库
    
    Args:
        config: VectorStoreConfig对象
        dimension: 向量维度
    
    Returns:
        BaseVectorStore实例
    """
    if config.type == "faiss":
        return FAISSVectorStore(
            dimension=dimension,
            index_type=config.index_type,
            metric=config.metric,
            nlist=config.nlist
        )
    else:
        raise ValueError(f"不支持的向量数据库类型: {config.type}")

