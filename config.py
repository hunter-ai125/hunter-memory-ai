#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全局配置文件
============
集中管理所有配置参数，供其他模块导入使用。
"""

import os
from pathlib import Path

# ============================================================
# 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).parent

# 数据目录（自动创建）
DATA_DIR = PROJECT_ROOT / "data"
BRAIN_DIR = DATA_DIR / "brains"
MEMORY_DIR = DATA_DIR / "memory"

# 注册表文件
REGISTRY_FILE = DATA_DIR / "registry.json"

# ============================================================
# 模型配置（自建Transformer）
# ============================================================

MODEL_CONFIG = {
    "dim": 64,           # 模型维度
    "layers": 2,         # Transformer层数
    "heads": 2,          # 注意力头数
    "lr": 0.01,          # 学习率
    "max_seq_len": 512,  # 最大序列长度
}

# ============================================================
# 记忆池配置
# ============================================================

MEMORY_CONFIG = {
    "use_vector": True,                          # 是否启用向量检索
    "vector_model": "./models/all-MiniLM-L6-v2",          # 向量嵌入模型
    "top_k_default": 5,                          # 默认检索数量
    "keyword_weight": 0.3,                       # 关键词权重
    "truth_weight": 0.4,                         # 真/假参数权重
    "persist_dir": "./memory",                   # 持久化目录
}

# ============================================================
# 本体AI（Agent）配置
# ============================================================

AGENT_CONFIG = {
    "system_prompt": "你是用户专属的AI助手，拥有记忆能力。请基于上下文信息自然回答用户问题，保持简洁、友好。如果不知道就说不知道，不要编造。",
    "auto_save_memory": True,
    "top_k_memories": 3,
    "memory_weight": 0.6,
}

# ============================================================
# 大模型客户端配置
# ============================================================

LLM_CONFIG = {
    "default_provider": "ollama",                # 默认提供商
    "ollama": {
        "model": "qwen2.5:1.5b",
        "base_url": "http://localhost:11434",
        "timeout": 60,
    },
    "openai": {
        "model": "gpt-3.5-turbo",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout": 60,
    },
    "custom": {
        "model": "custom-model",
        "base_url": "http://localhost:8000/v1",
        "timeout": 60,
        "request_format": "openai",
    },
}

# ============================================================
# AI工厂配置
# ============================================================

FACTORY_CONFIG = {
    "default_dim": 64,
    "default_layers": 2,
    "default_heads": 2,
    "default_lr": 0.01,
}

# ============================================================
# 环境变量辅助
# ============================================================

def get_openai_api_key() -> str:
    """获取OpenAI API Key（从环境变量）"""
    return os.environ.get(LLM_CONFIG["openai"]["api_key_env"], "")

def get_data_dir() -> Path:
    """获取数据目录（自动创建）"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR