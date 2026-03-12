"""LLM 服务抽象层

支持通义千问和 Kimi 两种 LLM 提供商。
"""
import os
from abc import ABC, abstractmethod
from pathlib import Path

# 加载 .env 文件
from dotenv import load_dotenv

# 尝试加载项目根目录的 .env 文件
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from config import (  # noqa: E402
    LLM_PROVIDER,
    QWEN_MODEL,
    QWEN_BASE_URL,
    KIMI_MODEL,
    KIMI_BASE_URL,
)


class BaseLLMService(ABC):
    """LLM 服务的抽象基类"""

    @abstractmethod
    def chat(self, system_prompt: str, user_message: str) -> str:
        """发送消息给 LLM，返回回答文本

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息

        Returns:
            LLM 的回答文本
        """
        pass


class QwenLLMService(BaseLLMService):
    """通义千问实现，使用 OpenAI 兼容 API

    DashScope 提供 OpenAI 兼容的 API 接口
    """

    def __init__(
        self,
        model: str = QWEN_MODEL,
        base_url: str = QWEN_BASE_URL,
    ) -> None:
        """初始化通义千问服务

        Args:
            model: 模型名称，默认 qwen-turbo
            base_url: API 端点，默认从 settings 读取
        """
        self.model = model
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        """获取 OpenAI 兼容客户端"""
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量")

            self._client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                # 设置超时和代理
                timeout=120.0,
                # 注意：OpenAI SDK 会自动读取环境变量中的代理设置
            )
        return self._client

    def chat(self, system_prompt: str, user_message: str) -> str:
        """调用通义千问 API

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息

        Returns:
            LLM 回答
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content


class KimiLLMService(BaseLLMService):
    """Kimi 实现，兼容 OpenAI SDK

    Kimi API 兼容 OpenAI SDK，使用 Moonshot 的 API 地址
    """

    def __init__(
        self,
        model: str = KIMI_MODEL,
        base_url: str = KIMI_BASE_URL,
    ) -> None:
        """初始化 Kimi 服务

        Args:
            model: 模型名称
            base_url: API 地址
        """
        self.model = model
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        """获取 OpenAI 兼容客户端"""
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("MOONSHOT_API_KEY")
            if not api_key:
                raise ValueError("未设置 MOONSHOT_API_KEY 环境变量")

            self._client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
            )
        return self._client

    def chat(self, system_prompt: str, user_message: str) -> str:
        """调用 Kimi API

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息

        Returns:
            LLM 回答
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content


def get_llm_service(provider: str | None = None) -> BaseLLMService:
    """工厂函数，根据配置返回对应 LLM 实现

    Args:
        provider: LLM 提供商，可选 "qwen" 或 "kimi"，默认从 settings 读取

    Returns:
        LLM 服务实例
    """
    if provider is None:
        provider = LLM_PROVIDER

    if provider == "qwen":
        return QwenLLMService()
    elif provider == "kimi":
        return KimiLLMService()
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")
