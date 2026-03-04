"""
配置管理模块
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    type: str  # "local" or "openai"
    model_path: Optional[str] = None  # 本地模型路径
    model_name: Optional[str] = None  # OpenAI模型名称
    device: str = "cpu"  # "cpu" or "cuda"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True


@dataclass
class ChunkingConfig:
    """分块策略配置"""
    strategy: str  # "fixed", "sentence", "semantic", "recursive", "hybrid"
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? "])
    
    # 语义分块特定参数
    semantic_threshold: float = 0.5
    
    # 混合策略参数
    hybrid_strategies: List[str] = field(default_factory=lambda: ["sentence", "fixed"])
    hybrid_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])


@dataclass
class VectorStoreConfig:
    """向量数据库配置"""
    type: str = "faiss"  # 目前支持faiss，可扩展
    index_path: Optional[str] = None
    index_type: str = "Flat"  # "Flat", "IVFFlat", "HNSW"
    metric: str = "cosine"  # "cosine", "l2", "ip"
    
    # IVF参数
    nlist: int = 100
    nprobe: int = 10


@dataclass
class RerankerConfig:
    """Reranker配置"""
    enabled: bool = False  # 是否启用Reranker
    type: str = "local"  # "local", "api", "cohere"
    model_path: Optional[str] = None  # 本地模型路径
    model_name: Optional[str] = None  # API模型名称
    api_key: Optional[str] = None  # API密钥
    base_url: str = "https://api.siliconflow.cn/v1"  # API基础URL
    device: str = "cpu"  # "cpu" or "cuda"
    batch_size: int = 32
    top_k: Optional[int] = None  # Rerank后返回的结果数


@dataclass
class LLMConfig:
    """LLM生成器配置"""
    enabled: bool = False  # 是否启用LLM生成
    type: str = "openai"  # "openai", "local"
    model_path: Optional[str] = None  # 本地模型路径
    model_name: str = "gpt-3.5-turbo"  # API模型名称
    api_key: Optional[str] = None  # API密钥
    base_url: Optional[str] = None  # API基础URL
    device: str = "cpu"  # "cpu" or "cuda" (本地模型)
    temperature: float = 0.7  # 温度参数
    max_tokens: int = 1000  # 最大token数
    system_prompt: Optional[str] = None  # 系统提示词
    include_sources: bool = True  # 是否在回答中包含来源引用


@dataclass
class DocumentLoaderConfig:
    """文档加载器配置"""
    use_pymupdf: bool = True  # 使用PyMuPDF还是pdfplumber
    enable_ocr: bool = False  # 启用OCR
    enable_table_extraction: bool = False  # 启用表格提取
    ocr_language: str = "chi_sim+eng"  # OCR语言
    min_text_length: int = 50  # 判断是否需要OCR的阈值


@dataclass
class RAGConfig:
    """RAG系统总配置"""
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    vector_store: VectorStoreConfig
    reranker: Optional[RerankerConfig] = None
    llm: Optional[LLMConfig] = None
    document_loader: Optional[DocumentLoaderConfig] = None
    
    # 文档加载配置
    pdf_directory: str = ""
    recursive: bool = True
    
    # 检索配置
    top_k: int = 5
    score_threshold: Optional[float] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典创建配置"""
        embedding_config = EmbeddingConfig(**config_dict.get("embedding", {}))
        chunking_config = ChunkingConfig(**config_dict.get("chunking", {}))
        vector_store_config = VectorStoreConfig(**config_dict.get("vector_store", {}))
        
        # Reranker配置（可选）
        reranker_config = None
        if "reranker" in config_dict:
            reranker_config = RerankerConfig(**config_dict["reranker"])
        
        # LLM配置（可选）
        llm_config = None
        if "llm" in config_dict:
            llm_config = LLMConfig(**config_dict["llm"])
        
        # 文档加载器配置（可选）
        document_loader_config = None
        if "document_loader" in config_dict:
            document_loader_config = DocumentLoaderConfig(**config_dict["document_loader"])
        
        return cls(
            embedding=embedding_config,
            chunking=chunking_config,
            vector_store=vector_store_config,
            reranker=reranker_config,
            llm=llm_config,
            document_loader=document_loader_config,
            pdf_directory=config_dict.get("pdf_directory", ""),
            recursive=config_dict.get("recursive", True),
            top_k=config_dict.get("top_k", 5),
            score_threshold=config_dict.get("score_threshold")
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> "RAGConfig":
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "embedding": self.embedding.__dict__,
            "chunking": self.chunking.__dict__,
            "vector_store": self.vector_store.__dict__,
            "pdf_directory": self.pdf_directory,
            "recursive": self.recursive,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold
        }
        
        if self.reranker is not None:
            result["reranker"] = self.reranker.__dict__
        
        if self.llm is not None:
            result["llm"] = self.llm.__dict__
        
        if self.document_loader is not None:
            result["document_loader"] = self.document_loader.__dict__
        
        return result
    
    def to_json(self, json_path: str):
        """保存为JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

