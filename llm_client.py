#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大模型客户端模块
================
统一接口，支持：
- 本地 Ollama（默认）
- 云端 OpenAI（兼容 API）
- 其他兼容 API

使用方式：
    from llm_client import create_client

    client = create_client(provider="ollama", model="qwen2.5:1.5b")
    response = client.generate("你好")
"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# 检查 requests 是否可用
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class BaseLLMClient(ABC):
    """大模型客户端基类"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs
    ) -> Dict[str, Any]:
        """生成回答，始终返回统一格式的字典"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass


class OllamaClient(BaseLLMClient):
    """Ollama 本地客户端"""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        if not HAS_REQUESTS:
            print("⚠️ 请安装 requests: pip install requests")
            self._available = False
            return False

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name") for m in models]
                if any(self.model in n for n in model_names):
                    self._available = True
                    return True
                else:
                    print(f"⚠️ 模型 {self.model} 未安装，请运行: ollama pull {self.model}")
                    self._available = False
                    return False
            self._available = False
            return False
        except Exception as e:
            print(f"⚠️ Ollama 未运行: {e}")
            self._available = False
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs
    ) -> Dict[str, Any]:
        # 检查依赖
        if not HAS_REQUESTS:
            return {
                "success": False,
                "result": "请安装 requests: pip install requests",
                "provider": "ollama"
            }

        if not self.is_available():
            return {
                "success": False,
                "result": "Ollama 不可用，请确保已启动并安装模型",
                "provider": "ollama"
            }

        full_prompt = ""
        if system_prompt:
            full_prompt += f"系统指令：{system_prompt}\n\n"
        if context:
            full_prompt += f"上下文信息：\n{context}\n\n"
        full_prompt += f"用户：{prompt}\n\n助手："

        try:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get("timeout", self.timeout)
            )
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": True,
                    "result": result.get("response", ""),
                    "provider": "ollama",
                    "model": self.model
                }
            else:
                return {
                    "success": False,
                    "result": f"HTTP {resp.status_code}: {resp.text}",
                    "provider": "ollama"
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "result": f"请求超时 ({self.timeout}s)",
                "provider": "ollama"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "result": "无法连接到 Ollama 服务",
                "provider": "ollama"
            }
        except Exception as e:
            return {
                "success": False,
                "result": f"错误: {e}",
                "provider": "ollama"
            }


class OpenAIClient(BaseLLMClient):
    """OpenAI 兼容 API 客户端（支持云端）"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        api_key_env: str = "OPENAI_API_KEY"
    ):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.base_url = base_url
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        if not HAS_REQUESTS:
            print("⚠️ 请安装 requests: pip install requests")
            self._available = False
            return False

        if not self.api_key:
            print("⚠️ 未设置 API Key，请设置 OPENAI_API_KEY 环境变量或传入 api_key")
            self._available = False
            return False

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            resp = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            if resp.status_code == 200:
                self._available = True
                return True
            else:
                print(f"⚠️ API 验证失败: {resp.status_code}")
                self._available = False
                return False
        except requests.exceptions.Timeout:
            print("⚠️ API 连接超时")
            self._available = False
            return False
        except Exception as e:
            print(f"⚠️ API 连接失败: {e}")
            self._available = False
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs
    ) -> Dict[str, Any]:
        if not HAS_REQUESTS:
            return {
                "success": False,
                "result": "请安装 requests: pip install requests",
                "provider": "openai"
            }

        if not self.is_available():
            return {
                "success": False,
                "result": "OpenAI API 不可用，请检查 API Key",
                "provider": "openai"
            }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.append({"role": "user", "content": f"上下文信息：\n{context}"})
        messages.append({"role": "user", "content": prompt})

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", self.timeout)
            )
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": True,
                    "result": result["choices"][0]["message"]["content"],
                    "provider": "openai",
                    "model": self.model
                }
            else:
                return {
                    "success": False,
                    "result": f"HTTP {resp.status_code}: {resp.text}",
                    "provider": "openai"
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "result": f"请求超时 ({self.timeout}s)",
                "provider": "openai"
            }
        except Exception as e:
            return {
                "success": False,
                "result": f"错误: {e}",
                "provider": "openai"
            }


class CustomClient(BaseLLMClient):
    """自定义 API 客户端（适配任何兼容接口）"""

    def __init__(
        self,
        model: str = "custom-model",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        request_format: str = "openai"
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout = timeout
        self.request_format = request_format
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        if not HAS_REQUESTS:
            print("⚠️ 请安装 requests: pip install requests")
            self._available = False
            return False

        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                self._available = True
                return True
            resp = requests.get(self.base_url, timeout=5)
            if resp.status_code < 500:
                self._available = True
                return True
            self._available = False
            return False
        except:
            self._available = False
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs
    ) -> Dict[str, Any]:
        if not HAS_REQUESTS:
            return {
                "success": False,
                "result": "请安装 requests: pip install requests",
                "provider": "custom"
            }

        if not self.is_available():
            return {
                "success": False,
                "result": "自定义 API 不可用",
                "provider": "custom"
            }

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if context:
                messages.append({"role": "user", "content": f"上下文信息：\n{context}"})
            messages.append({"role": "user", "content": prompt})

            headers = self.headers.copy()
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers["Content-Type"] = "application/json"

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", self.timeout)
            )
            if resp.status_code == 200:
                result = resp.json()
                return {
                    "success": True,
                    "result": result["choices"][0]["message"]["content"],
                    "provider": "custom",
                    "model": self.model
                }
            else:
                return {
                    "success": False,
                    "result": f"HTTP {resp.status_code}: {resp.text}",
                    "provider": "custom"
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "result": f"请求超时 ({self.timeout}s)",
                "provider": "custom"
            }
        except Exception as e:
            return {
                "success": False,
                "result": f"错误: {e}",
                "provider": "custom"
            }


# ============================================================
# 工厂函数
# ============================================================

def create_client(
    provider: str = "ollama",
    model: str = None,
    **kwargs
) -> BaseLLMClient:
    """
    创建大模型客户端

    参数：
        provider: ollama / openai / custom
        model: 模型名称
        **kwargs: 各客户端特定参数

    示例：
        # 本地 Ollama
        client = create_client("ollama", model="qwen2.5:1.5b")

        # 云端 OpenAI
        client = create_client("openai", model="gpt-3.5-turbo", api_key="sk-xxx")

        # 自定义 API
        client = create_client("custom", model="my-model", base_url="http://localhost:8000/v1")
    """
    if provider == "ollama":
        return OllamaClient(model=model or "qwen2.5:1.5b", **kwargs)
    elif provider == "openai":
        return OpenAIClient(model=model or "gpt-3.5-turbo", **kwargs)
    elif provider == "custom":
        return CustomClient(model=model or "custom-model", **kwargs)
    else:
        raise ValueError(f"不支持的 provider: {provider}")


def get_default_client() -> BaseLLMClient:
    """获取默认客户端（优先 Ollama）"""
    try:
        client = OllamaClient()
        if client.is_available():
            return client
    except:
        pass
    return OpenAIClient()


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 大模型客户端测试")
    print("=" * 60)

    client = create_client("ollama", model="qwen2.5:1.5b")
    print(f"Ollama 状态: {'✅ 可用' if client.is_available() else '❌ 不可用'}")

    if client.is_available():
        resp = client.generate("你好，请简单介绍一下自己")
        print(f"Ollama 回复: {resp.get('result', '')[:100]}...")
    else:
        print("⚠️ Ollama 未运行，跳过测试")