"""
文档加载模块
支持从目录加载PDF文件
"""
import os
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Document:
    """文档数据类"""
    content: str
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class PDFLoader:
    """PDF文档加载器"""
    
    def __init__(self, use_pymupdf: bool = True):
        """
        初始化PDF加载器
        
        Args:
            use_pymupdf: 是否使用PyMuPDF（默认），否则使用pdfplumber
        """
        self.use_pymupdf = use_pymupdf
        
        if use_pymupdf:
            try:
                import fitz  # PyMuPDF
                self.fitz = fitz
            except ImportError:
                raise ImportError("请安装PyMuPDF: pip install PyMuPDF")
        else:
            try:
                import pdfplumber
                self.pdfplumber = pdfplumber
            except ImportError:
                raise ImportError("请安装pdfplumber: pip install pdfplumber")
    
    def load_file(self, file_path: str) -> Document:
        """
        加载单个PDF文件
        
        Args:
            file_path: PDF文件路径
        
        Returns:
            Document对象
        """
        if self.use_pymupdf:
            return self._load_with_pymupdf(file_path)
        else:
            return self._load_with_pdfplumber(file_path)
    
    def _load_with_pymupdf(self, file_path: str) -> Document:
        """使用PyMuPDF加载PDF"""
        doc = self.fitz.open(file_path)
        
        text_parts = []
        page_count = len(doc)  # 先保存页数

        for page_num in range(page_count):
            page = doc[page_num]
            text_parts.append(page.get_text())

        content = "\n\n".join(text_parts)

        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "page_count": page_count,
            "file_size": os.path.getsize(file_path)
        }
        
        # 尝试获取PDF元数据
        try:
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata.update({
                    "title": pdf_metadata.get("title", ""),
                    "author": pdf_metadata.get("author", ""),
                    "subject": pdf_metadata.get("subject", ""),
                })
        except:
            pass
        
        doc.close()
        
        return Document(content=content, metadata=metadata)
    
    def _load_with_pdfplumber(self, file_path: str) -> Document:
        """使用pdfplumber加载PDF"""
        with self.pdfplumber.open(file_path) as pdf:
            text_parts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            content = "\n\n".join(text_parts)
            
            metadata = {
                "source": file_path,
                "filename": os.path.basename(file_path),
                "page_count": len(pdf.pages),
                "file_size": os.path.getsize(file_path)
            }
            
            # 尝试获取PDF元数据
            try:
                pdf_metadata = pdf.metadata
                if pdf_metadata:
                    metadata.update({
                        "title": pdf_metadata.get("Title", ""),
                        "author": pdf_metadata.get("Author", ""),
                        "subject": pdf_metadata.get("Subject", ""),
                    })
            except:
                pass
        
        return Document(content=content, metadata=metadata)
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        show_progress: bool = True
    ) -> List[Document]:
        """
        从目录加载所有PDF文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归遍历子目录
            show_progress: 是否显示进度
        
        Returns:
            Document对象列表
        """
        pdf_files = self._find_pdf_files(directory, recursive)
        
        if not pdf_files:
            print(f"警告: 在目录 {directory} 中未找到PDF文件")
            return []
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        documents = []
        
        if show_progress:
            from tqdm import tqdm
            pdf_files = tqdm(pdf_files, desc="加载PDF文件")
        
        for pdf_file in pdf_files:
            try:
                doc = self.load_file(pdf_file)
                documents.append(doc)
            except Exception as e:
                print(f"加载文件失败 {pdf_file}: {str(e)}")
                continue
        
        print(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def _find_pdf_files(self, directory: str, recursive: bool) -> List[str]:
        """查找目录中的所有PDF文件"""
        pdf_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        pdf_files.append(file_path)
        
        return sorted(pdf_files)

