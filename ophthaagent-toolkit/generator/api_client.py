"""
大模型API客户端 - 统一使用OpenAI兼容格式
支持通过base_url和model参数指定不同的模型
"""
import os
from typing import Optional
import logging
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # 也尝试从项目根目录加载
    load_dotenv()

logger = logging.getLogger(__name__)


class APIClient:
    """统一的API客户端接口 - 使用OpenAI兼容格式"""
    
    # 预设的模型配置（可选，也可以通过base_url和model直接指定）
    MODEL_PRESETS = {
        "gpt-4o": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY"
        },
        "gpt-4": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4",
            "api_key_env": "OPENAI_API_KEY"
        },
        "gpt-3.5-turbo": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-3.5-turbo",
            "api_key_env": "OPENAI_API_KEY"
        },
        "claude-3.5-sonnet": {
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-3-5-sonnet-20241022",
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "qwen-plus": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
            "api_key_env": "DASHSCOPE_API_KEY"
        },
        "qwen-max": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-max",
            "api_key_env": "DASHSCOPE_API_KEY"
        },
        "qwen-72b": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",  # 根据实际情况调整
            "api_key_env": "DASHSCOPE_API_KEY"
        }
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化API客户端
        
        配置优先级：命令行参数 > .env文件 > 预设默认值
        
        Args:
            model: 模型名称。可以是预设名称（如"gpt-4o"）或自定义模型名
                 如果为None，从.env文件的MODEL_NAME读取
            api_key: API密钥。如果为None，从.env文件读取
            base_url: API基础URL。如果为None，从.env文件或预设配置读取
            
        示例:
            # 使用.env文件中的配置（推荐）
            client = APIClient()
            
            # 使用预设模型
            client = APIClient(model="gpt-4o")
            
            # 使用自定义模型
            client = APIClient(
                model="custom-model-name",
                base_url="https://api.example.com/v1",
                api_key="your-api-key"
            )
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
        
        # 从.env文件读取配置（如果未通过参数提供）
        model = model or os.getenv("MODEL_NAME") or os.getenv("MODEL") or "gpt-4o"
        api_key = api_key or os.getenv("API_KEY")
        base_url = base_url or os.getenv("BASE_URL")
        
        self.model_name = model
        
        # 检查是否是预设模型
        if model.lower() in self.MODEL_PRESETS:
            preset = self.MODEL_PRESETS[model.lower()]
            # base_url优先级：参数 > .env > 预设
            self.base_url = base_url or os.getenv("BASE_URL") or preset["base_url"]
            self.model = preset["model"]
            api_key_env = preset["api_key_env"]
        else:
            # 自定义模型
            # base_url优先级：参数 > .env > 必须提供
            self.base_url = base_url or os.getenv("BASE_URL")
            if not self.base_url:
                raise ValueError(
                    f"自定义模型 '{model}' 需要提供 base_url 参数或在.env文件中设置BASE_URL。"
                    f"或者使用预设模型: {', '.join(self.MODEL_PRESETS.keys())}"
                )
            self.model = model
            api_key_env = "API_KEY"  # 默认环境变量名
        
        # 获取API密钥（优先级：参数 > .env中的API_KEY > 预设的api_key_env > 通用API_KEY）
        self.api_key = (
            api_key or 
            os.getenv("API_KEY") or 
            os.getenv(api_key_env)
        )
        
        if not self.api_key:
            raise ValueError(
                f"未提供API密钥。请在.env文件中设置API_KEY或{api_key_env}，"
                f"或通过环境变量/--api-key参数传入"
            )
        
        # 创建OpenAI客户端（兼容所有OpenAI格式的API）
        # 处理代理配置 - 如果环境变量中设置了NO_PROXY或DISABLE_PROXY，则禁用代理
        import httpx

        http_client = None
        if os.getenv("NO_PROXY") or os.getenv("DISABLE_PROXY"):
            # 明确禁用代理 - 传递 mounts 参数禁用所有代理
            # 或者简单地设置 trust_env=False 来忽略环境变量中的代理设置
            http_client = httpx.Client(trust_env=False)
            logger.info("已禁用代理（通过NO_PROXY或DISABLE_PROXY环境变量）")
        else:
            # 尝试使用系统代理，但忽略socks代理错误
            try:
                # 测试是否能创建默认客户端（会尝试使用系统代理）
                test_client = httpx.Client()
                test_client.close()
            except Exception as e:
                error_str = str(e).lower()
                if "socks" in error_str or ("proxy" in error_str and "unknown scheme" in error_str):
                    logger.warning(f"检测到不支持的代理配置，将禁用代理。错误: {e}")
                    logger.warning("提示: 可以安装 'pip install httpx[socks]' 支持socks代理，或设置 DISABLE_PROXY=1 环境变量")
                    http_client = httpx.Client(trust_env=False)

        # 创建客户端
        if http_client:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, http_client=http_client)
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        logger.info(f"初始化API客户端: model={self.model}, base_url={self.base_url}")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        images=None
    ) -> str:
        """生成文本
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            temperature: 温度参数（默认0.7）
            max_tokens: 最大token数（默认2000）
            images: 图像列表，用于多模态输入
            
        Returns:
            生成的文本
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            user_content = []
            user_content.append({"type": "text", "text": user_prompt})
            
            if images:
                for idx, image in enumerate(images):
                    user_content.append(image)
                    logger.info(f"✓ 添加图像 #{idx+1} 到API请求: {image.get('type', 'unknown')}")
                    if image.get('type') == 'image_url':
                        url = image.get('image_url', {}).get('url', '')
                        if url.startswith('data:image/jpeg;base64,'):
                            base64_length = len(url) - len('data:image/jpeg;base64,')
                            logger.info(f"  - 图像类型: JPEG base64 编码")
                            logger.info(f"  - Base64 长度: {base64_length} 字符")
                        else:
                            logger.info(f"  - 图像URL: {url[:100]}...")
            
            messages.append({"role": "user", "content": user_content})
            
            # 记录完整的消息结构（用于调试）
            logger.info(f"📤 发送给API的消息结构:")
            logger.info(f"  - 系统提示词长度: {len(system_prompt) if system_prompt else 0} 字符")
            logger.info(f"  - 用户消息内容类型数: {len(user_content)}")
            logger.info(f"  - 包含图像数: {len(images) if images else 0}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info(f"✓ API调用成功，收到响应")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API调用失败 (model={self.model}, base_url={self.base_url}): {e}")
            raise
