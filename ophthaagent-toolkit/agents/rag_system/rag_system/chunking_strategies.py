"""
文档分块策略模块
支持多种分块方法：固定大小、句子边界、语义分块、递归分块、混合策略
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import re
import numpy as np
from .document_loader import Document


class TextChunk:
    """文本块数据类"""
    
    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata
    
    def __repr__(self):
        return f"TextChunk(length={len(self.text)}, metadata={self.metadata})"


class BaseChunker(ABC):
    """分块器抽象基类"""
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[TextChunk]:
        """对文档进行分块"""
        pass
    
    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """批量分块"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks


class FixedSizeChunker(BaseChunker):
    """固定大小分块（带重叠滑动窗口）"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Document) -> List[TextChunk]:
        text = document.content
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                metadata = document.metadata.copy()
                metadata.update({
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                    "chunking_strategy": "fixed_size"
                })
                
                chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                chunk_id += 1
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks


class SentenceChunker(BaseChunker):
    """基于句子边界的分块"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 1,
        separators: Optional[List[str]] = None
    ):
        """
        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 句子重叠数量
            separators: 句子分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            # 默认分隔符（支持中英文）
            self.separators = ["。", "！", "？", ". ", "! ", "? ", "\n\n"]
        else:
            self.separators = separators
    
    def chunk_document(self, document: Document) -> List[TextChunk]:
        text = document.content
        
        # 分割成句子
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # 如果当前句子加入后超过chunk_size，先保存当前chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                
                metadata = document.metadata.copy()
                metadata.update({
                    "chunk_id": chunk_id,
                    "sentence_count": len(current_chunk),
                    "chunking_strategy": "sentence"
                })
                
                chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                chunk_id += 1
                
                # 保留重叠的句子
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # 保存最后一个chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            metadata = document.metadata.copy()
            metadata.update({
                "chunk_id": chunk_id,
                "sentence_count": len(current_chunk),
                "chunking_strategy": "sentence"
            })
            chunks.append(TextChunk(text=chunk_text, metadata=metadata))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 构建正则表达式模式
        pattern = "|".join(re.escape(sep) for sep in self.separators)
        
        # 分割并保留分隔符
        parts = re.split(f"({pattern})", text)
        
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i]
            separator = parts[i + 1] if i + 1 < len(parts) else ""
            combined = sentence + separator
            if combined.strip():
                sentences.append(combined)
        
        # 处理最后一部分
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1])
        
        return sentences


class RecursiveChunker(BaseChunker):
    """递归字符分块（类似LangChain的RecursiveCharacterTextSplitter）"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Args:
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            separators: 分隔符列表（按优先级排序）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            # 默认分隔符（从大到小）
            self.separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        else:
            self.separators = separators
    
    def chunk_document(self, document: Document) -> List[TextChunk]:
        text = document.content
        chunks_text = self._split_text(text, self.separators)
        
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            metadata = document.metadata.copy()
            metadata.update({
                "chunk_id": i,
                "chunking_strategy": "recursive"
            })
            chunks.append(TextChunk(text=chunk_text, metadata=metadata))
        
        return chunks
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归分割文本"""
        final_chunks = []
        
        # 选择当前分隔符
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []
        
        # 分割文本
        if separator:
            splits = text.split(separator)
        else:
            splits = [text]
        
        # 合并小块
        current_chunk = ""
        for split in splits:
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    final_chunks.append(current_chunk.rstrip(separator))
                
                # 如果单个split太大，继续递归分割
                if len(split) > self.chunk_size and new_separators:
                    sub_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split + separator
        
        if current_chunk:
            final_chunks.append(current_chunk.rstrip(separator))
        
        # 应用重叠
        if self.chunk_overlap > 0:
            final_chunks = self._apply_overlap(final_chunks)
        
        return [chunk for chunk in final_chunks if chunk.strip()]
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """应用重叠策略"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # 从前一个chunk取重叠部分
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            overlapped_chunk = overlap_text + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks


class SemanticChunker(BaseChunker):
    """基于语义相似度的分块"""
    
    def __init__(
        self,
        embedding_model,
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """
        Args:
            embedding_model: 嵌入模型实例
            threshold: 相似度阈值（低于此值则分割）
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_document(self, document: Document) -> List[TextChunk]:
        text = document.content
        
        # 先按句子分割
        sentence_chunker = SentenceChunker(chunk_size=10000)  # 大chunk_size以获取所有句子
        temp_chunks = sentence_chunker.chunk_document(document)
        
        if not temp_chunks:
            return []
        
        sentences = sentence_chunker._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [TextChunk(text=text, metadata={**document.metadata, "chunking_strategy": "semantic"})]
        
        # 计算句子嵌入
        embeddings = self.embedding_model.embed_documents(sentences)
        
        # 计算相邻句子的相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # 根据相似度分割
        chunks = []
        current_chunk = [sentences[0]]
        current_length = len(sentences[0])
        chunk_id = 0
        
        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            next_length = len(next_sentence)
            
            # 判断是否需要分割
            should_split = (
                sim < self.threshold or
                current_length + next_length > self.max_chunk_size
            )
            
            if should_split and current_length >= self.min_chunk_size:
                chunk_text = "".join(current_chunk)
                metadata = document.metadata.copy()
                metadata.update({
                    "chunk_id": chunk_id,
                    "chunking_strategy": "semantic",
                    "avg_similarity": np.mean([similarities[j] for j in range(max(0, i - len(current_chunk) + 1), i + 1)])
                })
                chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                chunk_id += 1
                
                current_chunk = [next_sentence]
                current_length = next_length
            else:
                current_chunk.append(next_sentence)
                current_length += next_length
        
        # 保存最后一个chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            metadata = document.metadata.copy()
            metadata.update({
                "chunk_id": chunk_id,
                "chunking_strategy": "semantic"
            })
            chunks.append(TextChunk(text=chunk_text, metadata=metadata))
        
        return chunks
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


class HybridChunker(BaseChunker):
    """混合分块策略"""
    
    def __init__(
        self,
        chunkers: List[BaseChunker],
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            chunkers: 分块器列表
            weights: 权重列表（用于后续检索时的加权）
        """
        self.chunkers = chunkers
        
        if weights is None:
            self.weights = [1.0 / len(chunkers)] * len(chunkers)
        else:
            if len(weights) != len(chunkers):
                raise ValueError("权重数量必须与分块器数量相同")
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def chunk_document(self, document: Document) -> List[TextChunk]:
        """使用所有分块器并合并结果"""
        all_chunks = []
        
        for i, chunker in enumerate(self.chunkers):
            chunks = chunker.chunk_document(document)
            
            # 为每个chunk添加混合策略信息
            for chunk in chunks:
                chunk.metadata["hybrid_strategy_index"] = i
                chunk.metadata["hybrid_weight"] = self.weights[i]
                chunk.metadata["chunking_strategy"] = f"hybrid_{chunk.metadata.get('chunking_strategy', 'unknown')}"
            
            all_chunks.extend(chunks)
        
        return all_chunks


def create_chunker(config, embedding_model=None) -> BaseChunker:
    """
    工厂函数：根据配置创建分块器
    
    Args:
        config: ChunkingConfig对象
        embedding_model: 嵌入模型（语义分块需要）
    
    Returns:
        BaseChunker实例
    """
    strategy = config.strategy.lower()
    
    if strategy == "fixed":
        return FixedSizeChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    elif strategy == "sentence":
        return SentenceChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators
        )
    
    elif strategy == "recursive":
        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators
        )
    
    elif strategy == "semantic":
        if embedding_model is None:
            raise ValueError("语义分块需要提供embedding_model")
        
        return SemanticChunker(
            embedding_model=embedding_model,
            threshold=config.semantic_threshold,
            min_chunk_size=config.chunk_size // 2,
            max_chunk_size=config.chunk_size * 2
        )
    
    elif strategy == "hybrid":
        if not config.hybrid_strategies:
            raise ValueError("混合策略需要指定hybrid_strategies")
        
        chunkers = []
        for sub_strategy in config.hybrid_strategies:
            sub_config = type(config)(
                strategy=sub_strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=config.separators,
                semantic_threshold=config.semantic_threshold
            )
            chunkers.append(create_chunker(sub_config, embedding_model))
        
        return HybridChunker(
            chunkers=chunkers,
            weights=config.hybrid_weights
        )
    
    else:
        raise ValueError(f"不支持的分块策略: {strategy}")

