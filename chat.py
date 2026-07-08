#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🤖 捕猎AI对话系统
=================
- 自由实时对话
- 自动记忆生长
- 3秒回忆机制
- 捕猎AI生态可视化
- 会话管理（对话块锁定/解锁）
"""

import time
import sys
import threading
from hunter_memory import HunterMemory as Memory
from agent import Agent, create_agent
from config import MEMORY_CONFIG, AGENT_CONFIG


class Spinner:
    def __init__(self, message: str = "处理中"):
        self.message = message
        self._running = False
        self._thread = None
        self._chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._idx = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin)
        self._thread.daemon = True
        self._thread.start()
        return self

    def _spin(self):
        while self._running:
            char = self._chars[self._idx % len(self._chars)]
            sys.stdout.write(f"\r{char} {self.message}")
            sys.stdout.flush()
            self._idx += 1
            time.sleep(0.08)

    def stop(self, result: str = ""):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        sys.stdout.write(f"\r✅ {self.message} {result}\n")
        sys.stdout.flush()


class ChatSystem:
    def __init__(self):
        print("=" * 60)
        print("🦅 捕猎AI对话系统")
        print("=" * 60)
        print("\n初始化中...")

        # 使用捕猎AI记忆池
        self.memory = Memory(persist_dir="./hunter_memory")

        # 启动会话（重要！）
        self.session_id = self.memory.sessions.start_session()
        print(f"📋 会话ID: {self.session_id[:8]}")

        # 创建Agent并传入会话ID
        self.agent = create_agent(memory=self.memory, session_id=self.session_id)

        # 加载种子数据
        self._load_seeds()

        # 3秒预热
        self._warmup()

        print("\n💡 命令:")
        print("   quit/exit - 退出（对话数据将转为知识）")
        print("   stats     - 查看生态统计")
        print("   clear     - 清除所有记忆")
        print("=" * 60 + "\n")

    def _load_seeds(self):
        seeds = [
            "残响城是一座被时间包裹的虚拟城市，源于时序混乱事件。",
            "时崎狂三拥有刻刻帝，可以操控时间。",
            "职阶体系允许角色加载特殊能力。",
            "记忆池应该原子化，数据块不可拆分。",
            "真/假参数可以帮助AI识别可信信息。",
            "神经AI用于记录查询通路。",
            "快入快出机制让系统保持小型化。"
        ]
        for seed in seeds:
            self.memory.add_knowledge(seed, source="seed")
        print(f"🌱 加载 {len(seeds)} 条种子记忆")

    def _warmup(self):
        print("🧠 预热记忆通路...")
        all_keywords = set()
        # 从 keyword_index 中提取关键词
        for kw_list in self.memory.keyword_index.values():
            all_keywords.update(kw_list)
        # 如果 keyword_index 为空，则从所有块内容提取关键词（备选）
        if not all_keywords:
            for block in self.memory.blocks.values():
                if block.content and not block.is_empty:
                    words = self.memory._extract_keywords(block.content)
                    all_keywords.update(words.keys())
        for kw in list(all_keywords)[:15]:
            self.memory.query(kw, top_k=2)
            time.sleep(0.01)
        print("   ✅ 预热完成")

    def run(self):
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    # 结束会话，对话块转为知识
                    print("\n📦 结束会话，对话数据转为知识...")
                    self.memory.end_session(self.session_id)
                    print("👋 再见！")
                    break

                if user_input.lower() == 'stats':
                    self.memory.print_stats()
                    continue

                if user_input.lower() == 'clear':
                    confirm = input("⚠️ 确定清除所有记忆？(y/n): ")
                    if confirm.lower() == 'y':
                        self.memory.clear()
                        self._load_seeds()
                        self._warmup()
                        print("✅ 记忆已重置")
                    continue

                self._process(user_input)

            except KeyboardInterrupt:
                print("\n👋 再见！")
                # 中断时也结束会话
                self.memory.end_session(self.session_id)
                break
            except Exception as e:
                print(f"❌ 错误: {e}")

    def _process(self, user_input: str):
        # 回忆
        spinner = Spinner("🧠 回忆中...").start()
        start = time.time()
        memories = self.memory.query(user_input, top_k=3)
        if len(memories) < 2:
            time.sleep(0.3)
            memories = self.memory.query(user_input, top_k=5)
        recall_time = time.time() - start
        if recall_time < 3.0:
            time.sleep(max(0, 3.0 - recall_time))
        spinner.stop(f"完成 ({max(recall_time, 3.0):.1f}s)")

        # 思考
        spinner = Spinner("💭 思考中...").start()
        try:
            start = time.time()
            result = self.agent.think(user_input)
            elapsed = time.time() - start
            spinner.stop(f"完成 ({elapsed:.1f}s)")
        except Exception as e:
            spinner.stop("⚠️ 失败")
            print(f"❌ {e}")
            return

        # 回答
        print(f"\n🤖 {result.get('result', '')}")
        if result.get("memories_used"):
            print(f"   📚 使用 {len(result['memories_used'])} 条记忆")


def main():
    chat = ChatSystem()
    chat.run()


if __name__ == "__main__":
    main()