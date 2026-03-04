"""
RAG System - 生产级检索增强生成系统

支持特性:
- 本地和API嵌入模型
- 多种分块策略（固定大小、句子边界、语义分块、递归分块、混合策略）
- FAISS向量数据库（可扩展）
- 灵活的配置管理
"""

from .config import RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig, RerankerConfig, LLMConfig, DocumentLoaderConfig
from .embeddings import BaseEmbedding, LocalEmbedding, OpenAIEmbedding, create_embedding
from .document_loader import Document, PDFLoader
from .document_loader_enhanced import EnhancedPDFLoader
from .chunking_strategies import (
    TextChunk,
    BaseChunker,
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    HybridChunker,
    create_chunker
)
from .vector_store import BaseVectorStore, FAISSVectorStore, create_vector_store
from .reranker import BaseReranker, LocalReranker, APIReranker, CohereReranker, create_reranker
from .llm_generator import BaseLLMGenerator, OpenAILLMGenerator, LocalLLMGenerator, create_llm_generator, format_answer_with_sources
from .rag_system import RAGSystem, create_rag_system
from .utils import format_retrieval_results, print_retrieval_results, create_sample_config

__version__ = "1.0.0"

__all__ = [
    # Config
    "RAGConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "VectorStoreConfig",
    "RerankerConfig",
    "LLMConfig",
    "DocumentLoaderConfig",
    
    # Embeddings
    "BaseEmbedding",
    "LocalEmbedding",
    "OpenAIEmbedding",
    "create_embedding",
    
    # Document Loader
    "Document",
    "PDFLoader",
    "EnhancedPDFLoader",
    
    # Chunking
    "TextChunk",
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "HybridChunker",
    "create_chunker",
    
    # Vector Store
    "BaseVectorStore",
    "FAISSVectorStore",
    "create_vector_store",
    
    # Reranker
    "BaseReranker",
    "LocalReranker",
    "APIReranker",
    "CohereReranker",
    "create_reranker",
    
    # LLM Generator
    "BaseLLMGenerator",
    "OpenAILLMGenerator",
    "LocalLLMGenerator",
    "create_llm_generator",
    "format_answer_with_sources",
    
    # RAG System
    "RAGSystem",
    "create_rag_system",
    
    # Utils
    "format_retrieval_results",
    "print_retrieval_results",
    "create_sample_config",
]

