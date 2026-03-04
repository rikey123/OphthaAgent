"""
LLM生成模块
用于对RAG检索结果进行润色和总结
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class BaseLLMGenerator(ABC):
    """LLM生成器抽象基类"""
    
    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            生成的回答
        """
        pass


class OpenAILLMGenerator(BaseLLMGenerator):
    """OpenAI API LLM生成器（兼容硅基流动等）"""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        base_url: str = None,
        default_system_prompt: str = None
    ):
        """
        初始化OpenAI LLM生成器
        
        Args:
            model_name: 模型名称
            api_key: API密钥（如果为None则从环境变量读取）
            base_url: API基础URL（如果为None则从环境变量读取）
            default_system_prompt: 默认系统提示词
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")
        
        import os
        
        self.model_name = model_name
        
        # 创建客户端
        if api_key and base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # 从环境变量读取
            self.client = OpenAI()
        
        # 默认系统提示词
        self.default_system_prompt = default_system_prompt or self._get_default_prompt()
        
        print(f"LLM生成器初始化完成: {model_name}")
    
    def _get_default_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个专业的AI助手，擅长根据提供的上下文信息回答用户问题。

请遵循以下原则：
1. 基于提供的上下文信息回答问题，不要编造信息
2. 如果上下文中没有相关信息，请明确告知用户
3. 回答要准确、简洁、有条理
4. 如果需要，可以引用上下文中的关键信息
5. 使用友好、专业的语气"""
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """生成回答"""
        # 使用提供的或默认的系统提示词
        sys_prompt = system_prompt or self.default_system_prompt
        
        # 构建用户消息
        user_message = f"""上下文信息：
{context}

用户问题：
{query}

请根据上述上下文信息回答用户问题。"""
        
        # 调用API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_with_sources(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        生成带来源的回答
        
        Args:
            query: 用户查询
            retrieval_results: 检索结果列表
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
            include_sources: 是否在回答中包含来源引用
        
        Returns:
            包含answer和sources的字典
        """
        # 构建上下文
        context_parts = []
        for i, result in enumerate(retrieval_results, 1):
            source = result['metadata'].get('filename', 'unknown')
            page = result['metadata'].get('page', 'N/A')
            text = result['text']
            
            context_parts.append(f"[文档{i}] 来源: {source}, 页码: {page}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # 如果需要包含来源引用，修改系统提示词
        if include_sources and system_prompt is None:
            sys_prompt = self.default_system_prompt + "\n\n注意：在回答中适当引用文档编号（如[文档1]），以标明信息来源。"
        else:
            sys_prompt = system_prompt
        
        # 生成回答
        answer = self.generate(
            query=query,
            context=context,
            system_prompt=sys_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 提取来源信息
        sources = []
        for result in retrieval_results:
            sources.append({
                "filename": result['metadata'].get('filename', 'unknown'),
                "page": result['metadata'].get('page', 'N/A'),
                "score": result.get('score', 0),
                "text": result['text'][:200] + "..."  # 只保留前200字符
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query
        }


class LocalLLMGenerator(BaseLLMGenerator):
    """本地LLM生成器（使用transformers）"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        default_system_prompt: str = None
    ):
        """
        初始化本地LLM生成器
        
        Args:
            model_path: 模型路径
            device: 设备（"cpu" 或 "cuda"）
            default_system_prompt: 默认系统提示词
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("请安装transformers和torch: pip install transformers torch")
        
        self.device = device
        self.model_path = model_path
        
        print(f"正在加载本地LLM模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.default_system_prompt = default_system_prompt or self._get_default_prompt()
        
        print(f"本地LLM模型加载完成")
    
    def _get_default_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个专业的AI助手，擅长根据提供的上下文信息回答用户问题。

请遵循以下原则：
1. 基于提供的上下文信息回答问题，不要编造信息
2. 如果上下文中没有相关信息，请明确告知用户
3. 回答要准确、简洁、有条理
4. 如果需要，可以引用上下文中的关键信息
5. 使用友好、专业的语气"""
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """生成回答"""
        import torch
        
        # 使用提供的或默认的系统提示词
        sys_prompt = system_prompt or self.default_system_prompt
        
        # 构建提示
        prompt = f"""{sys_prompt}

上下文信息：
{context}

用户问题：
{query}

回答："""
        
        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分（去除prompt）
        answer = response[len(prompt):].strip()
        
        return answer
    
    def generate_with_sources(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """生成带来源的回答"""
        # 构建上下文
        context_parts = []
        for i, result in enumerate(retrieval_results, 1):
            source = result['metadata'].get('filename', 'unknown')
            page = result['metadata'].get('page', 'N/A')
            text = result['text']
            
            context_parts.append(f"[文档{i}] 来源: {source}, 页码: {page}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # 生成回答
        answer = self.generate(
            query=query,
            context=context,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 提取来源信息
        sources = []
        for result in retrieval_results:
            sources.append({
                "filename": result['metadata'].get('filename', 'unknown'),
                "page": result['metadata'].get('page', 'N/A'),
                "score": result.get('score', 0),
                "text": result['text'][:200] + "..."
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query
        }


def create_llm_generator(config) -> BaseLLMGenerator:
    """
    工厂函数：根据配置创建LLM生成器
    
    Args:
        config: LLMConfig对象
    
    Returns:
        BaseLLMGenerator实例
    """
    llm_type = config.type.lower()
    
    if llm_type == "openai":
        return OpenAILLMGenerator(
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            default_system_prompt=config.system_prompt
        )
    
    elif llm_type == "local":
        if not config.model_path:
            raise ValueError("本地LLM需要指定model_path")
        
        return LocalLLMGenerator(
            model_path=config.model_path,
            device=config.device,
            default_system_prompt=config.system_prompt
        )
    
    else:
        raise ValueError(f"不支持的LLM类型: {llm_type}")


def format_answer_with_sources(result: Dict[str, Any]) -> str:
    """
    格式化带来源的回答
    
    Args:
        result: generate_with_sources的返回结果
    
    Returns:
        格式化的字符串
    """
    output = []
    
    output.append("=" * 70)
    output.append("问题：" + result['query'])
    output.append("=" * 70)
    output.append("\n回答：")
    output.append(result['answer'])
    output.append("\n" + "-" * 70)
    output.append("参考来源：")
    
    for i, source in enumerate(result['sources'], 1):
        output.append(f"\n[{i}] {source['filename']} (页码: {source['page']}, 相关度: {source['score']:.4f})")
        output.append(f"    {source['text']}")
    
    output.append("=" * 70)
    
    return "\n".join(output)

