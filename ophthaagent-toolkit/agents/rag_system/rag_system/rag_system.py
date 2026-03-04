"""
RAG系统核心模块
整合所有组件，提供统一的接口
"""
from typing import List, Tuple, Dict, Any, Optional
import os
from .config import RAGConfig
from .embeddings import create_embedding, BaseEmbedding
from .document_loader import PDFLoader, Document
from .document_loader_enhanced import EnhancedPDFLoader
from .chunking_strategies import create_chunker, BaseChunker, TextChunk
from .vector_store import create_vector_store, BaseVectorStore
from .reranker import create_reranker, BaseReranker
from .llm_generator import create_llm_generator, BaseLLMGenerator


class RAGSystem:
    """RAG系统主类"""
    
    def __init__(self, config: RAGConfig):
        """
        初始化RAG系统
        
        Args:
            config: RAGConfig配置对象
        """
        self.config = config
        
        # 初始化嵌入模型
        print("=" * 50)
        print("初始化嵌入模型...")
        self.embedding_model: BaseEmbedding = create_embedding(config.embedding)
        
        # 初始化向量数据库
        print("=" * 50)
        print("初始化向量数据库...")
        self.vector_store: BaseVectorStore = create_vector_store(
            config.vector_store,
            dimension=self.embedding_model.dimension
        )
        
        # 初始化分块器
        print("=" * 50)
        print("初始化分块器...")
        self.chunker: BaseChunker = create_chunker(
            config.chunking,
            embedding_model=self.embedding_model if config.chunking.strategy == "semantic" else None
        )
        
        # 初始化文档加载器
        if config.document_loader:
            self.pdf_loader = EnhancedPDFLoader(
                use_pymupdf=config.document_loader.use_pymupdf,
                enable_ocr=config.document_loader.enable_ocr,
                enable_table_extraction=config.document_loader.enable_table_extraction,
                ocr_language=config.document_loader.ocr_language,
                min_text_length=config.document_loader.min_text_length
            )
        else:
            self.pdf_loader = PDFLoader(use_pymupdf=True)
        
        # 初始化Reranker（如果启用）
        self.reranker: Optional[BaseReranker] = None
        if config.reranker and config.reranker.enabled:
            print("=" * 50)
            print("初始化Reranker...")
            self.reranker = create_reranker(config.reranker)
        
        # 初始化LLM生成器（如果启用）
        self.llm_generator: Optional[BaseLLMGenerator] = None
        if config.llm and config.llm.enabled:
            print("=" * 50)
            print("初始化LLM生成器...")
            self.llm_generator = create_llm_generator(config.llm)
        
        print("=" * 50)
        print("RAG系统初始化完成！")
        print("=" * 50)
    
    def build_index(
        self,
        pdf_directory: Optional[str] = None,
        recursive: Optional[bool] = None
    ):
        """
        构建索引：加载PDF文档、分块、嵌入、存储
        
        Args:
            pdf_directory: PDF文件目录（如果为None则使用配置中的目录）
            recursive: 是否递归遍历子目录（如果为None则使用配置中的设置）
        """
        # 使用参数或配置中的值
        pdf_dir = pdf_directory or self.config.pdf_directory
        is_recursive = recursive if recursive is not None else self.config.recursive
        
        if not pdf_dir:
            raise ValueError("必须指定pdf_directory")
        
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"目录不存在: {pdf_dir}")
        
        print("\n" + "=" * 50)
        print("开始构建索引...")
        print("=" * 50)
        
        # 1. 加载PDF文档
        print("\n步骤 1/4: 加载PDF文档")
        documents = self.pdf_loader.load_directory(pdf_dir, recursive=is_recursive)
        
        if not documents:
            print("警告: 没有加载到任何文档")
            return
        
        # 2. 文档分块
        print(f"\n步骤 2/4: 文档分块 (策略: {self.config.chunking.strategy})")
        chunks = self.chunker.chunk_documents(documents)
        print(f"生成 {len(chunks)} 个文本块")
        
        if not chunks:
            print("警告: 没有生成任何文本块")
            return
        
        # 3. 生成嵌入
        print(f"\n步骤 3/4: 生成嵌入向量")
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        embeddings = self.embedding_model.embed_documents(texts)
        print(f"生成 {len(embeddings)} 个嵌入向量")
        
        # 4. 存储到向量数据库
        print(f"\n步骤 4/4: 存储到向量数据库")
        self.vector_store.add_vectors(embeddings, texts, metadatas)
        
        print("\n" + "=" * 50)
        print("索引构建完成！")
        print("=" * 50)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回top-k个结果（如果为None则使用配置中的值）
            score_threshold: 分数阈值（如果为None则使用配置中的值）
        
        Returns:
            检索结果列表，每个结果包含: text, score, metadata
        """
        # 使用参数或配置中的值
        k = top_k or self.config.top_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        
        # 1. 生成查询嵌入
        query_embedding = self.embedding_model.embed_query(query)
        
        # 2. 向量搜索
        # 如果启用了Reranker，需要获取更多的初始结果
        if self.reranker:
            # 获取更多的候选结果用于Rerank
            initial_k = k * 3 if k * 3 <= 100 else 100  # 最多100个
            results = self.vector_store.search(query_embedding, top_k=initial_k)
        else:
            results = self.vector_store.search(query_embedding, top_k=k)
        
        # 3. 过滤和格式化结果
        formatted_results = []
        for text, score, metadata in results:
            # 应用分数阈值
            if threshold is not None and score < threshold:
                continue
            
            formatted_results.append({
                "text": text,
                "score": score,
                "metadata": metadata
            })
        
        # 4. 如果启用了Reranker，进行Rerank
        if self.reranker and formatted_results:
            formatted_results = self._rerank_results(query, formatted_results, k)
        
        return formatted_results
    
    def save_index(self, path: Optional[str] = None):
        """
        保存索引
        
        Args:
            path: 保存路径（如果为None则使用配置中的路径）
        """
        save_path = path or self.config.vector_store.index_path
        
        if not save_path:
            raise ValueError("必须指定保存路径")
        
        self.vector_store.save(save_path)
    
    def load_index(self, path: Optional[str] = None):
        """
        加载索引
        
        Args:
            path: 索引路径（如果为None则使用配置中的路径）
        """
        load_path = path or self.config.vector_store.index_path
        
        if not load_path:
            raise ValueError("必须指定索引路径")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"索引路径不存在: {load_path}")
        
        self.vector_store.load(load_path)
    
    def clear_index(self):
        """清空索引"""
        self.vector_store.clear()
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        使用Reranker对结果进行重新排序
        
        Args:
            query: 查询文本
            results: 初始检索结果
            top_k: 返回top-k个结果
        
        Returns:
            重新排序后的结果
        """
        # 提取文本
        documents = [r["text"] for r in results]
        
        # 调用Reranker
        rerank_top_k = self.config.reranker.top_k if self.config.reranker.top_k else top_k
        reranked_indices = self.reranker.rerank(query, documents, top_k=rerank_top_k)
        
        # 按照Rerank结果重新排序
        reranked_results = []
        for idx, rerank_score in reranked_indices:
            result = results[idx].copy()
            result["rerank_score"] = rerank_score  # 添加Rerank分数
            result["original_score"] = result["score"]  # 保留原始分数
            result["score"] = rerank_score  # 使用Rerank分数作为主分数
            reranked_results.append(result)
        
        return reranked_results
    
    def query_with_llm(
        self,
        query: str,
        top_k: int = None,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        使用LLM生成器对检索结果进行总结和润色
        
        Args:
            query: 查询文本
            top_k: 检索top-k个结果
            temperature: 温度参数（如果为None则使用配置中的值）
            max_tokens: 最大token数（如果为None则使用配置中的值）
            system_prompt: 系统提示词（如果为None则使用配置中的值）
        
        Returns:
            包含answer和sources的字典
        """
        if not self.llm_generator:
            raise ValueError("未启用LLM生成器，请在配置中设置 llm.enabled=True")
        
        # 1. 检索相关文档
        retrieval_results = self.retrieve(query, top_k=top_k)
        
        if not retrieval_results:
            return {
                "answer": "抱歉，没有找到相关信息。",
                "sources": [],
                "query": query
            }
        
        # 2. 使用LLM生成回答
        temp = temperature if temperature is not None else self.config.llm.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.llm.max_tokens
        
        result = self.llm_generator.generate_with_sources(
            query=query,
            retrieval_results=retrieval_results,
            system_prompt=system_prompt,
            temperature=temp,
            max_tokens=max_tok,
            include_sources=self.config.llm.include_sources
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "embedding_model": {
                "type": self.config.embedding.type,
                "dimension": self.embedding_model.dimension,
                "model_path": self.config.embedding.model_path,
                "model_name": self.config.embedding.model_name
            },
            "chunking_strategy": self.config.chunking.strategy,
            "vector_store": {
                "type": self.config.vector_store.type,
                "index_type": self.config.vector_store.index_type,
                "metric": self.config.vector_store.metric,
                "total_vectors": self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
            },
            "config": {
                "top_k": self.config.top_k,
                "score_threshold": self.config.score_threshold
            }
        }
    
    def print_stats(self):
        """打印系统统计信息"""
        stats = self.get_stats()
        
        print("\n" + "=" * 50)
        print("RAG系统统计信息")
        print("=" * 50)
        
        print("\n嵌入模型:")
        for key, value in stats["embedding_model"].items():
            print(f"  {key}: {value}")
        
        print(f"\n分块策略: {stats['chunking_strategy']}")
        
        print("\n向量数据库:")
        for key, value in stats["vector_store"].items():
            print(f"  {key}: {value}")
        
        print("\n检索配置:")
        for key, value in stats["config"].items():
            print(f"  {key}: {value}")
        
        print("=" * 50)


# 便捷函数
def create_rag_system(config_path: str) -> RAGSystem:
    """
    从配置文件创建RAG系统
    
    Args:
        config_path: 配置文件路径（JSON格式）
    
    Returns:
        RAGSystem实例
    """
    config = RAGConfig.from_json(config_path)
    return RAGSystem(config)

