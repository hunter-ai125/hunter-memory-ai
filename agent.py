#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本体AI模块
===========
本体AI（Agent）是整个系统的核心调度器：
- 对内：调用记忆池进行检索、存储、更新
- 对外：调用大模型客户端生成回答
- 核心流程：输入 → 检索记忆 → 构建上下文 → 调用LLM → 存储记忆

配置来源：config.py
"""

import time
from typing import Optional, List, Dict, Any

# 导入配置
from config import AGENT_CONFIG, MEMORY_CONFIG

# 导入依赖模块（顶层导入，避免循环导入）
from hunter_memory import HunterMemory as Memory, MemoryBlock
from llm_client import BaseLLMClient, get_default_client, create_client


class Agent:
    """
    本体AI（智能体）
    - 对内管理记忆池
    - 对外调用大模型
    - 核心思维流程：检索 → 增强 → 生成 → 记忆
    """

    def __init__(
            self,
            memory: Optional[Memory] = None,
            llm_client: Optional[BaseLLMClient] = None,
            system_prompt: str = None,
            auto_save_memory: bool = None,
            top_k_memories: int = None,
            memory_weight: float = None,
            session_id: str = None
    ):
        """
        参数：
            memory: 记忆池实例
            llm_client: 大模型客户端
            system_prompt: 系统提示词
            auto_save_memory: 是否自动保存对话到记忆池
            top_k_memories: 检索记忆数量
            memory_weight: 记忆在提示中的权重（0-1）
            session_id: 会话ID
        """
        self.memory = memory or Memory()
        self.llm = llm_client or get_default_client()
        self._session_id = session_id

        self.system_prompt = system_prompt or AGENT_CONFIG.get("system_prompt",
                                                               "你是一个智能助手。请基于提供的上下文信息回答用户问题。"
                                                               "如果上下文信息不足，请基于你的常识回答。"
                                                               )

        self.auto_save_memory = auto_save_memory if auto_save_memory is not None else AGENT_CONFIG.get(
            "auto_save_memory", True)
        self.top_k = top_k_memories or AGENT_CONFIG.get("top_k_memories", 3)
        self.memory_weight = memory_weight or AGENT_CONFIG.get("memory_weight", 0.6)

        self.conversation_history = []
        self._last_memories = []
        self._last_llm_response = None

    def set_session(self, session_id: str):
        """设置会话ID"""
        self._session_id = session_id

    # ========== 核心思维方法 ==========
    def _calc_dynamic_weights(self, user_input: str):
        """
        根据对话历史动态计算检索权重
        返回: (keyword_weight, truth_weight)
        """
        # 基础权重从配置获取
        base_kw = MEMORY_CONFIG.get("keyword_weight", 0.4)
        base_truth = MEMORY_CONFIG.get("truth_weight", 0.2)

        # 1. 对话越长，越重视记忆（假定记忆池更丰富）
        history_len = len(self.conversation_history)
        memory_boost = min(0.3, history_len * 0.02)  # 最多增加0.3

        # 2. 判断是否是新话题（与上一条对话比较）
        is_new_topic = True
        if self.conversation_history:
            last_user = self.conversation_history[-1].get('user', '')
            if last_user and any(w in user_input for w in last_user.split()[:3]):
                is_new_topic = False

        # 新话题更依赖关键词，延续话题更依赖向量（降低关键词权重，提高真/假权重）
        if is_new_topic:
            keyword_weight = base_kw + 0.2  # 关键词权重提高
            truth_weight = base_truth  # 真理权重不变
        else:
            keyword_weight = max(0.1, base_kw - 0.2)  # 关键词降低
            truth_weight = min(0.6, base_truth + 0.2)  # 真理权重提高

        # 归一化保证总和 <= 1 (剩余为向量权重)
        total = keyword_weight + truth_weight
        if total > 1.0:
            scale = 1.0 / total
            keyword_weight *= scale
            truth_weight *= scale

        return keyword_weight, truth_weight

    def think(self, user_input: str, top_k: int = None) -> Dict[str, Any]:
        """
        核心思维流程：
        1. 从记忆池检索相关记忆
        2. 构建上下文
        3. 调用大模型生成回答
        4. 记录历史
        5. 可选：存储新记忆
        """
        start_time = time.time()
        top_k = top_k or self.top_k

        # 动态计算权重
        keyword_weight, truth_weight = self._calc_dynamic_weights(user_input)
        # 打印动态权重（调试用）
        print(f"🔧 动态权重: keyword={keyword_weight:.2f}, truth={truth_weight:.2f}")
        memories = self.memory.query(
            user_input,
            top_k=top_k,
            keyword_weight=keyword_weight,
            truth_weight=truth_weight
        )

        # 2. 构建上下文
        context = self._build_context(memories)

        # 3. 调用大模型
        llm_result = self.llm.generate(
            prompt=user_input,
            system_prompt=self.system_prompt,
            context=context,
            temperature=0.7,
            max_tokens=256
        )

        self._last_llm_response = llm_result

        # 4. 记录对话历史
        self.conversation_history.append({
            "user": user_input,
            "assistant": llm_result.get("result", ""),
            "timestamp": time.time(),
            "memories_used": [m.get("id") for m in memories],
            "success": llm_result.get("success", False)
        })

        # 5. 自动存储到记忆池（使用对话专用方法）
        if self.auto_save_memory and llm_result.get("success"):
            self._store_conversation_memory(user_input, llm_result.get("result", ""))

        return {
            "success": llm_result.get("success", False),
            "result": llm_result.get("result", ""),
            "memories_used": memories,
            "context": context,
            "elapsed": time.time() - start_time,
            "provider": llm_result.get("provider", "unknown"),
            "model": llm_result.get("model", "unknown"),
            "history_length": len(self.conversation_history)
        }

    def _build_context(self, memories: List[Dict]) -> str:
        """构建上下文文本（增强版）"""
        if not memories:
            return ""

        context_parts = []
        for i, m in enumerate(memories, 1):
            truth_marker = ""
            if m.get("truth_score", 0.5) > 0.7:
                truth_marker = " [✓ 可信]"
            elif m.get("truth_score", 0.5) < 0.3:
                truth_marker = " [⚠️ 需验证]"
            content = m.get('content', '')
            context_parts.append(f"{i}. {content}{truth_marker}")

        if len(context_parts) > 1:
            return "以下是已知的相关信息：\n" + "\n".join(context_parts) + "\n请基于以上信息回答用户问题。回答要自然、简洁。"
        else:
            return "相关信息：\n" + "\n".join(context_parts) + "\n请基于以上信息回答用户问题。"

    def _store_conversation_memory(self, user_input: str, response: str):
        """
        将对话存入记忆池（使用对话专用方法，锁定，不合并）
        """
        if len(response) < 10:
            print(f"⚠️ 对话太短，不存储 (长度: {len(response)})")
            return

        content = f"问：{user_input}\n答：{response[:200]}"
        if len(response) > 200:
            content += "..."

        keywords = self._extract_keywords(user_input + " " + response)

        print(f"📝 准备存储对话: {content[:30]}...")  # 调试

        # 使用对话专用添加方法
        if hasattr(self.memory, 'add_conversation'):
            block_id = self.memory.add_conversation(
                content=content,
                keywords=keywords,
                session_id=self._session_id,
                truth_score=0.9
            )
            print(f"✅ 对话已存储，块ID: {block_id[:8]}")  # 调试
        else:
            print("⚠️ memory 不支持 add_conversation")
            self.memory.add(content, keywords=keywords, source="conversation", truth_score=0.9)

    def _extract_keywords(self, text: str, top_k: int = 8) -> Dict[str, float]:
        words = text.replace("，", " ").replace("。", " ").replace("？", " ").replace("！", " ").split()
        if not words:
            return {}
        word_counts = {}
        for w in words:
            if len(w) >= 2:
                word_counts[w] = word_counts.get(w, 0) + 1
        total = sum(word_counts.values())
        if total == 0:
            return {}
        result = {}
        for w, c in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            result[w] = c / total
        return result

    # ========== 记忆池操作（对内） ==========

    def add_memory(self, content: str, keywords: Dict[str, float] = None,
                   source: str = "user", **kwargs) -> str:
        """添加普通知识记忆（可合并）"""
        return self.memory.add_knowledge(content, keywords=keywords, source=source, **kwargs)

    def add_conversation_memory(self, content: str, keywords: Dict[str, float] = None,
                                session_id: str = None) -> str:
        """添加对话记忆（锁定，不合并）"""
        return self.memory.add_conversation(content, keywords=keywords, session_id=session_id or self._session_id)

    def query_memory(self, query: str, top_k: int = None) -> List[Dict]:
        return self.memory.query(query, top_k=top_k or self.top_k)

    def get_memory_block(self, block_id: str) -> Optional[MemoryBlock]:
        return self.memory.get(block_id)

    def update_truth(self, block_id: str, delta: float, reason: str = ""):
        self.memory.update_truth(block_id, delta, reason)

    def merge_memories(self, block_id1: str, block_id2: str) -> Optional[str]:
        return self.memory.merge(block_id1, block_id2)

    def get_memory_stats(self) -> Dict:
        return self.memory.stats()

    def print_memory_stats(self):
        self.memory.print_stats()

    def end_session(self, session_id: str = None):
        """结束会话"""
        sid = session_id or self._session_id
        if sid and hasattr(self.memory, 'end_session'):
            self.memory.end_session(sid)

    # ========== LLM 操作（对外） ==========

    def get_llm_status(self) -> bool:
        return self.llm.is_available()

    def set_llm_client(self, client: BaseLLMClient):
        self.llm = client

    # ========== 历史记录 ==========

    def get_history(self, limit: int = None) -> List[Dict]:
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []

    # ========== 重置 ==========

    def reset(self):
        self.conversation_history = []
        self._last_memories = []
        self._last_llm_response = None


# ============================================================
# 工厂函数
# ============================================================

def create_agent(
        memory: Optional[Memory] = None,
        memory_dir: str = None,
        llm_provider: str = None,
        llm_model: str = None,
        system_prompt: str = None,
        auto_save_memory: bool = None,
        session_id: str = None,
        **kwargs
) -> Agent:
    """
    快速创建Agent

    示例：
        agent = create_agent()  # 使用默认配置
        # 使用自定义记忆池
        agent = create_agent(memory=my_memory_instance)
        # 使用 OpenAI
        agent = create_agent(
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            api_key="sk-xxx"
        )
    """
    # 如果提供了 memory 实例，直接使用；否则根据 memory_dir 创建
    if memory is None:
        memory = Memory(persist_dir=memory_dir or MEMORY_CONFIG.get("persist_dir", "./memory"))

    # 创建LLM客户端
    llm_provider = llm_provider or "ollama"
    if not llm_model:
        if llm_provider == "ollama":
            llm_model = "qwen2.5:1.5b"
        elif llm_provider == "openai":
            llm_model = "gpt-3.5-turbo"

    llm_kwargs = {k: v for k, v in kwargs.items() if k not in ['top_k_memories', 'memory_weight']}
    llm = create_client(provider=llm_provider, model=llm_model, **llm_kwargs)

    return Agent(
        memory=memory,
        llm_client=llm,
        system_prompt=system_prompt,
        auto_save_memory=auto_save_memory,
        top_k_memories=kwargs.get('top_k_memories'),
        memory_weight=kwargs.get('memory_weight'),
        session_id=session_id
    )


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 本体AI（Agent）测试")
    print("=" * 60)

    agent = create_agent()
    print(f"LLM 状态: {'✅ 可用' if agent.get_llm_status() else '❌ 不可用'}")

    agent.add_memory("北京是中国的首都", keywords={"北京": 0.9, "中国": 0.8, "首都": 0.7})
    agent.add_memory("残响城是一座被时间包裹的虚拟城市", keywords={"残响城": 0.9, "时间": 0.7})
    agent.add_memory("时崎狂三拥有刻刻帝，可以操控时间", keywords={"时崎狂三": 0.9, "刻刻帝": 0.8})

    print("📚 已添加 3 条记忆")

    questions = [
        "北京在哪里？",
        "什么是残响城？",
        "时崎狂三的能力是什么？",
        "刻刻帝是什么？",
        "残响城和时间有什么关系？",
        "时崎狂三为什么要回到过去？",
        "职阶体系是什么？",
        "北京是哪个国家的首都？",
        "时崎狂三和残响城有什么关系？"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        result = agent.think(q)
        if result["success"]:
            print(f"🤖 {result['result']}")
            print(f"   📚 使用 {len(result['memories_used'])} 条记忆")
            print(f"   ⏱️ 耗时 {result['elapsed']:.2f}s")
        else:
            print(f"⚠️ {result.get('result', '未知错误')}")

    agent.print_memory_stats()