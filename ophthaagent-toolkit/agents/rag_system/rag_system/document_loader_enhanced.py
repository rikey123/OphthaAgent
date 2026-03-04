"""
增强的文档加载模块
支持OCR（光学字符识别）和表格提取
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import re


@dataclass
class Document:
    """文档数据类"""
    content: str
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class EnhancedPDFLoader:
    """增强的PDF文档加载器（支持OCR和表格提取）"""
    
    def __init__(
        self,
        use_pymupdf: bool = True,
        enable_ocr: bool = False,
        enable_table_extraction: bool = False,
        ocr_language: str = "chi_sim+eng",
        min_text_length: int = 50
    ):
        """
        初始化增强PDF加载器
        
        Args:
            use_pymupdf: 是否使用PyMuPDF（默认），否则使用pdfplumber
            enable_ocr: 是否启用OCR
            enable_table_extraction: 是否启用表格提取
            ocr_language: OCR语言（chi_sim=简体中文, chi_tra=繁体中文, eng=英文）
            min_text_length: 判断是否需要OCR的最小文本长度阈值
        """
        self.use_pymupdf = use_pymupdf
        self.enable_ocr = enable_ocr
        self.enable_table_extraction = enable_table_extraction
        self.ocr_language = ocr_language
        self.min_text_length = min_text_length
        
        # 导入基础PDF库
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
        
        # 导入OCR库（如果启用）
        if enable_ocr:
            try:
                import pytesseract
                from PIL import Image
                self.pytesseract = pytesseract
                self.Image = Image
                
                # 检查tesseract是否安装
                try:
                    pytesseract.get_tesseract_version()
                except:
                    print("警告: Tesseract未安装或未配置，OCR功能可能无法使用")
                    print("请安装Tesseract: https://github.com/tesseract-ocr/tesseract")
            except ImportError:
                raise ImportError("请安装pytesseract和Pillow: pip install pytesseract Pillow")
        
        # 导入表格提取库（如果启用）
        if enable_table_extraction:
            try:
                import pdfplumber
                self.pdfplumber_for_table = pdfplumber
            except ImportError:
                raise ImportError("表格提取需要pdfplumber: pip install pdfplumber")
        
        print(f"增强PDF加载器初始化完成:")
        print(f"  - OCR: {'启用' if enable_ocr else '禁用'}")
        print(f"  - 表格提取: {'启用' if enable_table_extraction else '禁用'}")
    
    def load_file(self, file_path: str) -> Document:
        """
        加载单个PDF文件
        
        Args:
            file_path: PDF文件路径
        
        Returns:
            Document对象
        """
        print(f"正在加载: {os.path.basename(file_path)}")
        
        # 1. 尝试常规文本提取
        text_content, page_count = self._extract_text(file_path)
        
        # 2. 检查是否需要OCR
        needs_ocr = self._needs_ocr(text_content, page_count)
        
        if needs_ocr and self.enable_ocr:
            print(f"  检测到扫描PDF，启用OCR...")
            ocr_content = self._extract_with_ocr(file_path)
            text_content = ocr_content if ocr_content else text_content
        
        # 3. 提取表格（如果启用）
        table_content = ""
        table_count = 0
        if self.enable_table_extraction:
            table_content, table_count = self._extract_tables(file_path)
        
        # 4. 合并内容
        final_content = self._merge_content(text_content, table_content)
        
        # 5. 构建元数据
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "page_count": page_count,
            "file_size": os.path.getsize(file_path),
            "used_ocr": needs_ocr and self.enable_ocr,
            "table_count": table_count,
            "content_length": len(final_content)
        }
        
        print(f"  完成: {len(final_content)}字符, {table_count}个表格, OCR={'是' if metadata['used_ocr'] else '否'}")
        
        return Document(content=final_content, metadata=metadata)
    
    def _extract_text(self, file_path: str) -> tuple:
        """提取PDF文本"""
        if self.use_pymupdf:
            return self._extract_text_pymupdf(file_path)
        else:
            return self._extract_text_pdfplumber(file_path)
    
    def _extract_text_pymupdf(self, file_path: str) -> tuple:
        """使用PyMuPDF提取文本"""
        doc = self.fitz.open(file_path)
        
        text_parts = []
        page_count = len(doc)  # 先保存页数

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[页码 {page_num + 1}]\n{text}")

        content = "\n\n".join(text_parts)
        doc.close()

        return content, page_count

    def _extract_text_pdfplumber(self, file_path: str) -> tuple:
        """使用pdfplumber提取文本"""
        with self.pdfplumber.open(file_path) as pdf:
            text_parts = []
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"[页码 {page_num + 1}]\n{text}")

            content = "\n\n".join(text_parts)
            return content, len(pdf.pages)

    def _needs_ocr(self, text_content: str, page_count: int) -> bool:
        """判断是否需要OCR"""
        if not text_content or len(text_content.strip()) < self.min_text_length:
            return True

        # 计算平均每页文本长度
        avg_text_per_page = len(text_content) / page_count if page_count > 0 else 0

        # 如果平均每页文本很少，可能是扫描PDF
        if avg_text_per_page < self.min_text_length:
            return True

        return False

    def _extract_with_ocr(self, file_path: str) -> str:
        """使用OCR提取文本"""
        if not self.enable_ocr:
            return ""

        try:
            doc = self.fitz.open(file_path)
            ocr_parts = []
            page_count = len(doc)  # 先保存页数

            for page_num in range(page_count):
                page = doc[page_num]
                
                # 将页面转换为图像
                pix = page.get_pixmap(matrix=self.fitz.Matrix(2, 2))  # 2倍缩放提高质量
                img_data = pix.tobytes("png")
                
                # 使用PIL加载图像
                img = self.Image.open(io.BytesIO(img_data))
                
                # OCR识别
                text = self.pytesseract.image_to_string(
                    img,
                    lang=self.ocr_language,
                    config='--psm 6'  # 假设统一的文本块
                )
                
                if text.strip():
                    ocr_parts.append(f"[页码 {page_num + 1}]\n{text}")
            
            doc.close()
            return "\n\n".join(ocr_parts)
        
        except Exception as e:
            print(f"  OCR失败: {e}")
            return ""
    
    def _extract_tables(self, file_path: str) -> tuple:
        """提取表格"""
        if not self.enable_table_extraction:
            return "", 0
        
        try:
            with self.pdfplumber_for_table.open(file_path) as pdf:
                table_parts = []
                table_count = 0
                
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    if tables:
                        for table_idx, table in enumerate(tables):
                            table_count += 1
                            
                            # 转换表格为文本
                            table_text = self._format_table(table, page_num + 1, table_idx + 1)
                            table_parts.append(table_text)
                
                return "\n\n".join(table_parts), table_count
        
        except Exception as e:
            print(f"  表格提取失败: {e}")
            return "", 0
    
    def _format_table(self, table: List[List], page_num: int, table_num: int) -> str:
        """格式化表格为文本"""
        if not table:
            return ""
        
        lines = [f"[表格 {table_num} - 页码 {page_num}]"]
        
        # 处理表头
        if table[0]:
            header = " | ".join([str(cell) if cell else "" for cell in table[0]])
            lines.append(header)
            lines.append("-" * len(header))
        
        # 处理数据行
        for row in table[1:]:
            if row:
                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                lines.append(row_text)
        
        return "\n".join(lines)
    
    def _merge_content(self, text_content: str, table_content: str) -> str:
        """合并文本和表格内容"""
        parts = []
        
        if text_content and text_content.strip():
            parts.append(text_content)
        
        if table_content and table_content.strip():
            parts.append("\n\n=== 提取的表格 ===\n\n" + table_content)
        
        return "\n\n".join(parts)
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_pattern: str = "*.pdf"
    ) -> List[Document]:
        """
        从目录加载所有PDF文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归搜索子目录
            file_pattern: 文件匹配模式
        
        Returns:
            Document对象列表
        """
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise ValueError(f"目录不存在: {directory}")
        
        # 查找PDF文件
        if recursive:
            pdf_files = list(directory_path.rglob(file_pattern))
        else:
            pdf_files = list(directory_path.glob(file_pattern))
        
        if not pdf_files:
            print(f"警告: 在 {directory} 中没有找到PDF文件")
            return []
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 加载所有文件
        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.load_file(str(pdf_file))
                documents.append(doc)
            except Exception as e:
                print(f"加载失败 {pdf_file.name}: {e}")
        
        print(f"成功加载 {len(documents)} 个文档")
        return documents


# 为了向后兼容，添加一个简单的别名
class PDFLoader(EnhancedPDFLoader):
    """向后兼容的PDF加载器"""
    
    def __init__(self, use_pymupdf: bool = True):
        super().__init__(
            use_pymupdf=use_pymupdf,
            enable_ocr=False,
            enable_table_extraction=False
        )


# 需要导入io模块用于OCR
import io

