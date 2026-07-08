#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI 项目核心模块
================
统一导出所有子模块

使用方式：
    from ai_project import Agent, Memory, create_agent, create_client
    from ai_project import AI_list, AI_train, AI_chat
"""

from .hunter_memory import HunterMemory as Memory, MemoryBlock
from .agent import Agent, create_agent
from .llm_client import (
    BaseLLMClient,
    OllamaClient,
    OpenAIClient,
    CustomClient,
    create_client,
    get_default_client
)
from .management import (
    AI_list,
    AI_train,
    AI_chat,
    AI_show,
    AI_delete,
    AI_clear,
    AI_info,
    AI_alias
)
from .model import UntrainedSuperModel
from .config import (
    MODEL_CONFIG,
    MEMORY_CONFIG,
    AGENT_CONFIG,
    LLM_CONFIG,
    FACTORY_CONFIG,
    DATA_DIR,
    BRAIN_DIR,
    MEMORY_DIR,
    REGISTRY_FILE,
    get_openai_api_key,
    get_data_dir
)

__version__ = "2.0.0"

__all__ = [
    # 记忆池
    "Memory",
    "MemoryBlock",
    # 本体AI
    "Agent",
    "create_agent",
    # 大模型客户端
    "BaseLLMClient",
    "OllamaClient",
    "OpenAIClient",
    "CustomClient",
    "create_client",
    "get_default_client",
    # AI工厂
    "AI_list",
    "AI_train",
    "AI_chat",
    "AI_show",
    "AI_delete",
    "AI_clear",
    "AI_info",
    "AI_alias",
    # 模型
    "UntrainedSuperModel",
    # 配置
    "MODEL_CONFIG",
    "MEMORY_CONFIG",
    "AGENT_CONFIG",
    "LLM_CONFIG",
    "FACTORY_CONFIG",
    "DATA_DIR",
    "BRAIN_DIR",
    "MEMORY_DIR",
    "REGISTRY_FILE",
    "get_openai_api_key",
    "get_data_dir",
]