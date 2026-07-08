#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

"""
捕猎AI记忆池系统 v5.0 — 轻量级参数化AI + 冻结机制
===================================================
每个捕猎者AI拥有真正可训练的神经网络参数（轻量级MLP）
- 合并 = 预测对战（胜者获得全部参数，败者消亡）
- 喂养 = 一步梯度下降微调
- 冻结 = 3分钟无活动或出现同级AI时自动冻结
- 解冻 = 查询命中或出现新同级AI时自动唤醒
"""

import json
import time
import uuid
import threading
import random
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ==================== 1. 轻量级可训练AI模型 ====================

class LightweightAI(nn.Module):
    """
    轻量级神经网络，每个捕猎者AI独立拥有
    包含主模型 + 策略网络，参数量约 3000-15000
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # 主模型（理解数据）
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.1)

        # 策略网络（自主决策）
        self.strategy_fc1 = nn.Linear(6, 32)
        self.strategy_fc2 = nn.Linear(32, 16)
        self.strategy_fc3 = nn.Linear(16, 3)  # 吸收率、融合程度、创造子块

    def forward(self, x):
        """主模型前向传播"""
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

    def strategy_forward(self, state):
        """策略网络前向传播"""
        x = torch.relu(self.strategy_fc1(state))
        x = torch.relu(self.strategy_fc2(x))
        return torch.sigmoid(self.strategy_fc3(x))

    def decide_fusion_strategy(self, self_score: float, opponent_score: float,
                               self_level: int, opponent_level: int,
                               self_norm: float, opponent_norm: float) -> Dict:
        """自主决策：根据状态决定如何处置对手"""
        state = torch.tensor([
            self_score,
            opponent_score,
            min(self_norm / 10.0, 1.0),
            min(opponent_norm / 10.0, 1.0),
            self_level / 5.0,
            opponent_level / 5.0
        ], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            strategy = self.strategy_forward(state)

        return {
            "absorption_rate": strategy[0][0].item(),
            "fusion_mix": strategy[0][1].item(),
            "create_new": strategy[0][2].item() > 0.5,
            "self_score": self_score,
            "opponent_score": opponent_score
        }

    def get_params(self) -> Dict[str, List[float]]:
        """导出所有参数"""
        return {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()}

    def load_params(self, params: Dict[str, List[float]]):
        """从字典加载参数"""
        state_dict = {k: torch.tensor(v) for k, v in params.items()}
        self.load_state_dict(state_dict)

    def clone(self) -> 'LightweightAI':
        new_model = LightweightAI()
        new_model.load_state_dict(self.state_dict())
        return new_model

    @classmethod
    def random(cls, input_dim: int = 128, hidden_dim: int = 64):
        return cls(input_dim, hidden_dim)


# ==================== 2. 数据块定义 ====================

@dataclass
class MemoryBlock:
    """原子化数据块 - 每个块拥有一个轻量级AI模型"""
    id: str
    content: str
    source: str
    timestamp: float
    truth_score: float = 0.5
    falsity_score: float = 0.5
    authority_score: float = 0.0

    # 轻量级模型参数
    model_params: Optional[Dict[str, List[float]]] = field(default_factory=dict)
    input_dim: int = 128
    hidden_dim: int = 64

    # 层级与捕猎
    level: int = 0
    is_hunter: bool = False
    hunter_for: List[str] = field(default_factory=list)
    eaten_by: Optional[str] = None

    # 从属关系
    children: List[str] = field(default_factory=list)
    spared: List[str] = field(default_factory=list)
    parent: Optional[str] = None

    # 数据承载
    last_bear_time: float = 0.0
    last_active_time: float = 0.0
    is_frozen: bool = False
    parameters_to_send: Dict[str, float] = field(default_factory=dict)

    # 抽象标记
    is_abstract: bool = False
    is_empty: bool = False
    absorbed_from: List[Dict] = field(default_factory=list)
    vector: Optional[List[float]] = None

    # 会话相关
    is_conversation: bool = False
    is_active: bool = False
    is_locked: bool = False
    session_id: Optional[str] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['vector'] = self.vector
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryBlock':
        return cls(
            id=data['id'],
            content=data.get('content', ''),
            source=data.get('source', 'user'),
            timestamp=data.get('timestamp', time.time()),
            truth_score=data.get('truth_score', 0.5),
            falsity_score=data.get('falsity_score', 0.5),
            authority_score=data.get('authority_score', 0.0),
            model_params=data.get('model_params', {}),
            input_dim=data.get('input_dim', 128),
            hidden_dim=data.get('hidden_dim', 64),
            level=data.get('level', 0),
            is_hunter=data.get('is_hunter', False),
            hunter_for=data.get('hunter_for', []),
            eaten_by=data.get('eaten_by', None),
            children=data.get('children', []),
            spared=data.get('spared', []),
            parent=data.get('parent', None),
            last_bear_time=data.get('last_bear_time', 0.0),
            last_active_time=data.get('last_active_time', 0.0),
            is_frozen=data.get('is_frozen', False),
            parameters_to_send=data.get('parameters_to_send', {}),
            is_abstract=data.get('is_abstract', False),
            is_empty=data.get('is_empty', False),
            absorbed_from=data.get('absorbed_from', []),
            vector=data.get('vector', None),
            is_conversation=data.get('is_conversation', False),
            is_active=data.get('is_active', False),
            is_locked=data.get('is_locked', False),
            session_id=data.get('session_id', None)
        )

    def get_model(self) -> LightweightAI:
        model = LightweightAI(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        if self.model_params:
            model.load_params(self.model_params)
        return model

    def set_model(self, model: LightweightAI):
        self.model_params = model.get_params()

    def touch(self):
        self.last_active_time = time.time()
        self.is_frozen = False

    def should_freeze(self, current_time: float, freeze_timeout: int = 180) -> bool:
        if self.is_frozen:
            return True
        if self.level == 0:
            return False
        return (current_time - self.last_active_time) > freeze_timeout


# ==================== 3. 会话管理器 ====================

class SessionManager:
    def __init__(self, memory: 'HunterMemory', timeout: int = 1800):
        self.memory = memory
        self.timeout = timeout
        self.active_sessions: Dict[str, Dict] = {}
        self._start_cleanup()

    def start_session(self, user_id: str = None) -> str:
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "user_id": user_id or "anonymous",
            "start_time": time.time(),
            "last_activity": time.time(),
            "conversation_blocks": []
        }
        return session_id

    def add_conversation(self, session_id: str, content: str,
                         keywords: Dict[str, float] = None) -> str:
        if session_id not in self.active_sessions:
            session_id = self.start_session()
        block_id = self.memory.add_conversation(content, keywords, session_id)
        self.active_sessions[session_id]["conversation_blocks"].append(block_id)
        self.active_sessions[session_id]["last_activity"] = time.time()
        return block_id

    def end_session(self, session_id: str):
        if session_id not in self.active_sessions:
            return
        count = 0
        for block_id in self.active_sessions[session_id]["conversation_blocks"]:
            block = self.memory.get(block_id)
            if block:
                block.is_conversation = False
                block.is_active = False
                block.is_locked = False
                block.source = "knowledge"
                block.session_id = None
                self.memory.blocks[block_id] = block
                count += 1
        self.memory._save()
        print(f"✅ 会话 {session_id[:8]} 结束，{count} 条对话转为知识")
        self.memory._check_force_merge()
        del self.active_sessions[session_id]

    def get_active_session(self, session_id: str) -> Optional[Dict]:
        return self.active_sessions.get(session_id)

    def is_active(self, session_id: str) -> bool:
        return session_id in self.active_sessions

    def _cleanup_timeout(self):
        now = time.time()
        to_end = []
        for sid, session in self.active_sessions.items():
            if now - session["last_activity"] > self.timeout:
                to_end.append(sid)
        for sid in to_end:
            print(f"⏰ 会话 {sid[:8]} 超时自动结束")
            self.end_session(sid)

    def _start_cleanup(self):
        def cleanup_loop():
            while True:
                time.sleep(60)
                try:
                    self._cleanup_timeout()
                except Exception:
                    pass

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()


# ==================== 4. 核心记忆池 ====================

class HunterMemory:
    """
    捕猎AI记忆池 v5.0 — 参数化AI + 冻结机制
    """

    # 阈值配置
    MERGE_SIMILARITY = 0.6
    MAX_BEAR_TIME = 10.0
    LEVEL_3_ACTIVATE = 3
    LEVEL_4_ACTIVATE = 4
    MAX_BLOCKS = 50
    FORCE_MERGE_THRESHOLD = 10
    FREEZE_TIMEOUT = 180  # 3分钟
    FREEZE_CHECK_INTERVAL = 30  # 每30秒检查一次

    def __init__(self, persist_dir: str = "./hunter_memory", use_vector: bool = True):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.use_vector = use_vector and HAS_SENTENCE_TRANSFORMERS
        self.vector_encoder = None
        if self.use_vector:
            try:
                self.vector_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"⚠️ 向量编码器加载失败: {e}")
                self.use_vector = False

        self.blocks: Dict[str, MemoryBlock] = {}
        self.keyword_index: Dict[str, List[str]] = {}
        self._lock = threading.Lock()  # 线程锁

        self._load()
        self._start_cleanup_thread()
        self._start_freeze_manager()
        self.sessions = SessionManager(self)

    # ==================== 持久化 ====================

    def _get_blocks_path(self) -> Path:
        return self.persist_dir / "blocks.json"

    def _get_index_path(self) -> Path:
        return self.persist_dir / "index.json"

    def _save(self):
        with self._lock:
            blocks_data = {bid: block.to_dict() for bid, block in self.blocks.items()}
            with open(self._get_blocks_path(), "w", encoding="utf-8") as f:
                json.dump(blocks_data, f, ensure_ascii=False, indent=2)
            with open(self._get_index_path(), "w", encoding="utf-8") as f:
                json.dump(self.keyword_index, f, ensure_ascii=False, indent=2)

    def _load(self):
        with self._lock:
            if self._get_blocks_path().exists():
                try:
                    with open(self._get_blocks_path(), "r", encoding="utf-8") as f:
                        blocks_data = json.load(f)
                        for bid, data in blocks_data.items():
                            self.blocks[bid] = MemoryBlock.from_dict(data)
                except Exception as e:
                    print(f"⚠️ 加载失败: {e}")

            if self._get_index_path().exists():
                try:
                    with open(self._get_index_path(), "r", encoding="utf-8") as f:
                        self.keyword_index = json.load(f)
                except Exception as e:
                    print(f"⚠️ 加载索引失败: {e}")

    # ==================== 添加方法 ====================

    def _text_to_vector(self, text: str) -> List[float]:
        """将文本转换为向量（用于模型输入）"""
        if self.use_vector and self.vector_encoder:
            return self.vector_encoder.encode(text).tolist()
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i:i + 2], 16) / 255.0 for i in range(0, 64, 2)]

    def _create_model_from_text(self, content: str) -> LightweightAI:
        model = LightweightAI.random()
        return model

    def add_conversation(self, content: str, keywords: Dict[str, float] = None,
                         session_id: str = None, truth_score: float = 0.9) -> str:
        if keywords is None:
            keywords = self._extract_keywords(content)

        with self._lock:
            if len(self.blocks) >= self.MAX_BLOCKS:
                self._fast_exit_locked()

            model = self._create_model_from_text(content)
            block = MemoryBlock(
                id=str(uuid.uuid4()),
                content=content,
                source="conversation",
                timestamp=time.time(),
                truth_score=truth_score,
                falsity_score=0.1,
                level=0,
                is_conversation=True,
                is_active=True,
                is_locked=True,
                session_id=session_id or str(uuid.uuid4()),
                last_bear_time=time.time(),
                last_active_time=time.time(),
                is_empty=False
            )
            block.set_model(model)
            if self.use_vector and self.vector_encoder:
                block.vector = self.vector_encoder.encode(content).tolist()

            self.blocks[block.id] = block
            for kw in keywords:
                self.keyword_index.setdefault(kw, []).append(block.id)

            if session_id and session_id in self.sessions.active_sessions:
                self.sessions.active_sessions[session_id]["conversation_blocks"].append(block.id)
                self.sessions.active_sessions[session_id]["last_activity"] = time.time()

        self._save()
        return block.id

    def add_knowledge(self, content: str, keywords: Dict[str, float] = None,
                      source: str = "knowledge", truth_score: float = 0.5) -> str:
        if keywords is None:
            keywords = self._extract_keywords(content)

        with self._lock:
            if len(self.blocks) >= self.MAX_BLOCKS:
                self._fast_exit_locked()

            model = self._create_model_from_text(content)
            block = MemoryBlock(
                id=str(uuid.uuid4()),
                content=content,
                source=source,
                timestamp=time.time(),
                truth_score=truth_score,
                falsity_score=0.5,
                level=0,
                is_conversation=False,
                is_active=False,
                is_locked=False,
                last_bear_time=time.time(),
                last_active_time=time.time(),
                is_empty=False
            )
            block.set_model(model)
            if self.use_vector and self.vector_encoder:
                block.vector = self.vector_encoder.encode(content).tolist()

            self.blocks[block.id] = block
            for kw in keywords:
                self.keyword_index.setdefault(kw, []).append(block.id)

        self._save()
        self._check_force_merge()
        return block.id

    def add(self, content: str, keywords: Dict[str, float] = None,
            source: str = "user", truth_score: float = 0.5,
            falsity_score: float = 0.5) -> str:
        return self.add_knowledge(content, keywords, source, truth_score)

    # ==================== 模型辅助方法 ====================

    def _compute_model_similarity(self, block1: MemoryBlock, block2: MemoryBlock) -> float:
        if not block1.model_params or not block2.model_params:
            return 0.0
        model1 = block1.get_model()
        model2 = block2.get_model()
        total_diff = 0.0
        total_norm = 0.0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            diff = (p1 - p2).norm().item()
            norm = (p1.norm().item() + p2.norm().item()) / 2
            total_diff += diff
            total_norm += norm
        if total_norm == 0:
            return 0.0
        return max(0.0, 1.0 - (total_diff / total_norm))

    def _compute_model_norm(self, model: LightweightAI) -> float:
        total = 0.0
        for p in model.parameters():
            total += p.norm().item() ** 2
        return total ** 0.5

    def _compute_self_score(self, model: LightweightAI) -> float:
        test_input = torch.randn(1, 128)
        with torch.no_grad():
            output = model(test_input)
        return output.item()

    def _split_params(self, model: LightweightAI) -> Tuple[List, List]:
        params = []
        for name, param in model.named_parameters():
            params.append((name, param.data.clone()))
        half = max(1, len(params) // 2)
        indices = list(range(len(params)))
        random.shuffle(indices)
        half1 = [params[i] for i in indices[:half]]
        half2 = [params[i] for i in indices[half:]]
        return half1, half2

    def _predict_params(self, half_params: List[Tuple], model: LightweightAI) -> float:
        if not half_params:
            return 0.0
        temp_model = LightweightAI(
            input_dim=model.fc1.in_features,
            hidden_dim=model.fc1.out_features
        )
        temp_state_dict = temp_model.state_dict()
        for name, param in half_params:
            if name in temp_state_dict:
                temp_state_dict[name] = param.clone()
        temp_model.load_state_dict(temp_state_dict)

        total_error = 0.0
        total_norm = 0.0
        for (name1, param1), (name2, param2) in zip(
                temp_model.named_parameters(),
                model.named_parameters()
        ):
            if name1 == name2:
                diff = (param1 - param2).norm().item()
                norm = param2.norm().item()
                total_error += diff
                total_norm += norm
        if total_norm == 0:
            return 0.0
        return max(0.0, 1.0 - (total_error / total_norm))

    def _execute_winner_strategy(self, winner: MemoryBlock, loser: MemoryBlock,
                                 winning_decision: Dict):
        winner_model = winner.get_model()
        loser_model = loser.get_model()

        absorption_rate = winning_decision['absorption_rate']
        fusion_mix = winning_decision['fusion_mix']
        create_new = winning_decision['create_new']

        if absorption_rate > 0.7:
            for w_param, l_param in zip(winner_model.parameters(), loser_model.parameters()):
                w_param.data = w_param.data * (1 - fusion_mix) + l_param.data * fusion_mix
            print(f"   🔗 深度融合: 吸收率={absorption_rate:.2f}")
        elif absorption_rate > 0.3:
            for w_param, l_param in zip(winner_model.parameters(), loser_model.parameters()):
                diff = (w_param - l_param).norm().item()
                if diff < 0.5:
                    w_param.data = w_param.data * 0.7 + l_param.data * 0.3
            print(f"   📊 选择性吸收: 吸收率={absorption_rate:.2f}")
        else:
            print(f"   🗑️ 丢弃对手参数: 吸收率={absorption_rate:.2f}")

        if create_new:
            new_model = LightweightAI()
            for p_new, p_winner, p_loser in zip(
                    new_model.parameters(),
                    winner_model.parameters(),
                    loser_model.parameters()
            ):
                p_new.data = (p_winner.data + p_loser.data) / 2

            new_block = MemoryBlock(
                id=str(uuid.uuid4()),
                content=f"[子块] 由 {winner.id[:8]} 创造",
                source="created",
                timestamp=time.time(),
                level=0,
                is_hunter=False,
                is_empty=False,
                is_abstract=False,
                last_active_time=time.time()
            )
            new_block.set_model(new_model)
            with self._lock:
                self.blocks[new_block.id] = new_block
            print(f"   🌱 创造新子块: {new_block.id[:8]}")

        winner.set_model(winner_model)
        with self._lock:
            self.blocks[winner.id] = winner

    def _destroy_block(self, block_id: str):
        with self._lock:
            block = self.blocks.get(block_id)
            if not block:
                return
            block.content = "[已消亡]"
            block.model_params = {}
            block.is_empty = True
            block.is_abstract = True
            block.level = -1
            block.last_bear_time = time.time()

            for kw in list(self.keyword_index.keys()):
                if block_id in self.keyword_index[kw]:
                    self.keyword_index[kw].remove(block_id)

            self.blocks[block_id] = block
        self._save()

    # ==================== 标准合并 ====================

    def merge(self, block_id1: str, block_id2: str) -> Optional[str]:
        with self._lock:
            b1 = self.blocks.get(block_id1)
            b2 = self.blocks.get(block_id2)
            if not b1 or not b2:
                return None
            if b1.is_locked or b2.is_locked:
                print(f"⚠️ 无法合并: 块被锁定")
                return None
            if not b1.model_params or not b2.model_params:
                print(f"⚠️ 无法合并: 块缺少模型参数")
                return None

        print(f"\n⚔️ 预测对战合并: {b1.id[:8]} (L{b1.level}) vs {b2.id[:8]} (L{b2.level})")

        model1 = b1.get_model()
        model2 = b2.get_model()
        b1.touch()
        b2.touch()

        score1 = self._compute_self_score(model1)
        score2 = self._compute_self_score(model2)
        norm1 = self._compute_model_norm(model1)
        norm2 = self._compute_model_norm(model2)

        decision1 = model1.decide_fusion_strategy(
            self_score=score1, opponent_score=score2,
            self_level=b1.level, opponent_level=b2.level,
            self_norm=norm1, opponent_norm=norm2
        )
        decision2 = model2.decide_fusion_strategy(
            self_score=score2, opponent_score=score1,
            self_level=b2.level, opponent_level=b1.level,
            self_norm=norm2, opponent_norm=norm1
        )

        print(f"   🧠 {b1.id[:8]} 策略: 吸收={decision1['absorption_rate']:.2f}, 融合={decision1['fusion_mix']:.2f}, 创造={decision1['create_new']}")
        print(f"   🧠 {b2.id[:8]} 策略: 吸收={decision2['absorption_rate']:.2f}, 融合={decision2['fusion_mix']:.2f}, 创造={decision2['create_new']}")

        score_combined1 = score1 * 0.4 + decision1['absorption_rate'] * 0.3 + decision1['fusion_mix'] * 0.3
        score_combined2 = score2 * 0.4 + decision2['absorption_rate'] * 0.3 + decision2['fusion_mix'] * 0.3

        if score_combined1 > score_combined2:
            winner, loser = b1, b2
            winning_decision = decision1
            print(f"   🏆 {winner.id[:8]} 胜出!")
        elif score_combined2 > score_combined1:
            winner, loser = b2, b1
            winning_decision = decision2
            print(f"   🏆 {winner.id[:8]} 胜出!")
        else:
            print(f"   🤝 平局! 双方各吸收一半")
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p1.data = (p1.data + p2.data) / 2
            b1.set_model(model1)
            b2.set_model(model2)
            b1.level = 0
            b2.level = 0
            with self._lock:
                self.blocks[b1.id] = b1
                self.blocks[b2.id] = b2
            self._save()
            return b1.id

        self._execute_winner_strategy(winner, loser, winning_decision)
        self._destroy_block(loser.id)

        winner.level += 1
        winner.is_hunter = True
        with self._lock:
            self.blocks[winner.id] = winner

        if winner.level >= 3:
            self._activate_level_3(winner)

        self._save()
        print(f"   ✅ 合并完成! 胜者: {winner.id[:8]} (L{winner.level})")
        return winner.id

    # ==================== 主动合并 ====================

    def active_merge(self, hunter_id: str, target_id: str) -> Dict[str, Any]:
        with self._lock:
            hunter = self.blocks.get(hunter_id)
            target = self.blocks.get(target_id)
            if not hunter or not target:
                return {"status": "error", "message": "AI不存在"}
            if hunter.is_locked or target.is_locked:
                return {"status": "error", "message": "AI被锁定"}
            if hunter.level < self.LEVEL_4_ACTIVATE or target.level < self.LEVEL_4_ACTIVATE:
                return {"status": "error", "message": "只有4级以上AI才能主动合并"}
            if hunter_id == target_id:
                return {"status": "error", "message": "不能自己合并自己"}
            hunter.touch()
            target.touch()

        result = self.merge(hunter_id, target_id)
        if result:
            return {"status": "success", "message": f"合并成功，胜者: {result[:8]}", "winner_id": result}
        else:
            return {"status": "failed", "message": "合并失败"}

    # ==================== 3级激活 ====================

    def _activate_level_3(self, hunter: MemoryBlock):
        print(f"🔥 L3激活: {hunter.id[:8]}")
        to_remove = []
        with self._lock:
            for child_id in hunter.children:
                child = self.blocks.get(child_id)
                if child and not child.is_locked:
                    similarity = self._compute_model_similarity(child, hunter)
                    if similarity > 0.85:
                        to_remove.append(child_id)

            for child_id in to_remove:
                if child_id in hunter.children:
                    hunter.children.remove(child_id)
                    hunter.spared.append(child_id)
                child = self.blocks.get(child_id)
                if child:
                    child.content = "[被剔除]"
                    child.model_params = {}
                    child.level = 0
                    child.is_abstract = True
                    child.parent = None
                    child.is_empty = True
                    child.last_bear_time = time.time()
                    self.blocks[child_id] = child
                    print(f"   📉 剔除空块: {child_id[:8]}")
        self._save()

    # ==================== 强制合并 ====================

    def _check_force_merge(self):
        with self._lock:
            knowledge_blocks = [b for b in self.blocks.values()
                                if not b.is_conversation and b.level == 0
                                and not b.is_locked and not b.is_empty
                                and b.source != "seed"]
        if len(knowledge_blocks) >= self.FORCE_MERGE_THRESHOLD:
            print(f"🔥 触发强制合并：{len(knowledge_blocks)} 个知识块")
            self._force_merge(knowledge_blocks)

    def _force_merge(self, blocks: List[MemoryBlock]):
        sorted_blocks = sorted(blocks, key=lambda x: x.timestamp)
        for i in range(0, len(sorted_blocks) - 1, 2):
            if i + 1 < len(sorted_blocks):
                result = self.merge(sorted_blocks[i].id, sorted_blocks[i + 1].id)
                if result:
                    print(f"   ✅ 强制合并成功")

    # ==================== 冻结管理 ====================

    def _start_freeze_manager(self):
        def freeze_loop():
            while True:
                time.sleep(self.FREEZE_CHECK_INTERVAL)
                try:
                    self._manage_freezing()
                except Exception:
                    pass
        thread = threading.Thread(target=freeze_loop, daemon=True)
        thread.start()

    def _manage_freezing(self):
        current_time = time.time()
        with self._lock:
            for bid, block in self.blocks.items():
                if block.level > 0 and not block.is_locked and not block.is_frozen:
                    if block.should_freeze(current_time, self.FREEZE_TIMEOUT):
                        block.is_frozen = True
                        self.blocks[bid] = block
                        print(f"❄️ 冻结AI: {bid[:8]} (L{block.level})")

            layers = {}
            for bid, block in self.blocks.items():
                if block.level > 0 and not block.is_locked:
                    layers.setdefault(block.level, []).append(bid)

            for level, bids in layers.items():
                if len(bids) >= 2:
                    for bid in bids:
                        block = self.blocks.get(bid)
                        if block and block.is_frozen:
                            block.is_frozen = False
                            block.last_active_time = current_time
                            print(f"🔓 解冻AI: {bid[:8]} (L{level}) 因出现同级AI")

    def _thaw_on_query(self, block_id: str):
        with self._lock:
            block = self.blocks.get(block_id)
            if block and block.is_frozen:
                block.is_frozen = False
                block.last_active_time = time.time()
                self.blocks[block_id] = block
                print(f"🔓 查询命中解冻: {block_id[:8]}")

    # ==================== 查询 ====================

    def query(self, query: str, top_k: int = 5, keyword_weight: float = 0.3,
              truth_weight: float = 0.4) -> List[Dict[str, Any]]:
        if not self.blocks:
            return []

        query_words = set(query.replace("，", " ").replace("。", " ").split())
        kw_scores = {}
        with self._lock:
            for word in query_words:
                if word in self.keyword_index:
                    for bid in self.keyword_index[word]:
                        block = self.blocks.get(bid)
                        if not block or block.is_empty or block.level < 0:
                            continue
                        weight = self._extract_keywords(block.content).get(word, 0.5)
                        kw_scores[bid] = kw_scores.get(bid, 0) + weight

            vec_scores = {}
            if self.use_vector and self.vector_encoder:
                try:
                    query_vec = self.vector_encoder.encode(query)
                    for bid, block in self.blocks.items():
                        if block.vector is not None and not block.is_empty and block.level >= 0:
                            vec_scores[bid] = self._cosine_similarity(query_vec, block.vector)
                except Exception:
                    pass

            all_bids = set(kw_scores.keys()) | set(vec_scores.keys())
            max_kw = max(kw_scores.values()) if kw_scores else 1.0
            max_vec = max(vec_scores.values()) if vec_scores else 1.0

            final_scores = {}
            for bid in all_bids:
                block = self.blocks[bid]
                if block.is_empty or block.level < 0:
                    continue
                kw_score = kw_scores.get(bid, 0) / max_kw if max_kw > 0 else 0
                vec_score = vec_scores.get(bid, 0) / max_vec if max_vec > 0 else 0
                truth_boost = block.truth_score * truth_weight
                conv_boost = 0.3 if block.is_conversation and block.is_active else 0
                freeze_penalty = 0.2 if block.is_frozen else 0

                final_scores[bid] = (keyword_weight * kw_score +
                                     (1 - keyword_weight - truth_weight) * vec_score +
                                     truth_boost + conv_boost - freeze_penalty)

            sorted_bids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 解冻命中的AI（在锁外执行，避免长时间持有锁）
        for bid, _ in sorted_bids:
            self._thaw_on_query(bid)

        results = []
        with self._lock:
            for bid, score in sorted_bids:
                block = self.blocks[bid]
                results.append({
                    "id": bid,
                    "content": block.content,
                    "source": block.source,
                    "truth_score": block.truth_score,
                    "level": block.level,
                    "is_hunter": block.is_hunter,
                    "is_conversation": block.is_conversation,
                    "is_active": block.is_active,
                    "is_empty": block.is_empty,
                    "is_frozen": block.is_frozen,
                    "has_model": bool(block.model_params),
                    "score": score
                })
        return results

    # ==================== 工具方法 ====================

    def get(self, block_id: str) -> Optional[MemoryBlock]:
        with self._lock:
            block = self.blocks.get(block_id)
        if block:
            self._thaw_on_query(block_id)
        return block

    def delete(self, block_id: str):
        with self._lock:
            if block_id not in self.blocks:
                return
            block = self.blocks[block_id]
            if block.is_locked:
                print(f"⚠️ 无法删除锁定块: {block_id[:8]}")
                return
            del self.blocks[block_id]
            for kw in self.keyword_index:
                if block_id in self.keyword_index[kw]:
                    self.keyword_index[kw].remove(block_id)
        self._save()

    def _extract_keywords(self, text: str, top_k: int = 10) -> Dict[str, float]:
        words = text.replace("，", " ").replace("。", " ").replace("？", " ").replace("！", " ").split()
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

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        if hasattr(vec1, 'tolist'):
            vec1 = vec1.tolist()
        if hasattr(vec2, 'tolist'):
            vec2 = vec2.tolist()
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _fast_exit(self):
        # 此方法仅在持有锁时调用，故命名为 _fast_exit_locked
        pass

    def _fast_exit_locked(self):
        sorted_blocks = sorted(self.blocks.items(), key=lambda x: x[1].last_bear_time)
        if sorted_blocks:
            to_delete = sorted_blocks[0][0]
            if not self.blocks[to_delete].is_locked:
                # 延迟删除，避免在迭代中修改
                pass
        # 实际删除在外部处理，此处调用 delete
        if sorted_blocks:
            self.delete(sorted_blocks[0][0])

    def _start_cleanup_thread(self):
        def cleanup_loop():
            while True:
                time.sleep(2)
                try:
                    self._check_3s_timeout()
                except Exception:
                    pass
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def _check_3s_timeout(self):
        current_time = time.time()
        to_delete = []
        with self._lock:
            for bid, block in self.blocks.items():
                if block.level == 0 and block.is_empty and not block.is_locked:
                    if current_time - block.last_bear_time > self.MAX_BEAR_TIME:
                        to_delete.append(bid)
        for bid in to_delete:
            print(f"⏰ 淘汰空块: {bid[:8]}")
            self.delete(bid)

    def end_session(self, session_id: str):
        self.sessions.end_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get_active_session(session_id)

    # ==================== 更新真假值 ====================

    def update_truth(self, block_id: str, delta: float, reason: str = ""):
        with self._lock:
            block = self.blocks.get(block_id)
            if block:
                block.truth_score = max(0.0, min(1.0, block.truth_score + delta))
        self._save()

    # ==================== 统计 ====================

    def stats(self) -> Dict:
        with self._lock:
            levels = {}
            empty_count = 0
            conv_count = 0
            model_count = 0
            frozen_count = 0
            for block in self.blocks.values():
                levels[block.level] = levels.get(block.level, 0) + 1
                if block.is_empty:
                    empty_count += 1
                if block.is_conversation:
                    conv_count += 1
                if block.model_params:
                    model_count += 1
                if block.is_frozen:
                    frozen_count += 1
            return {
                "total": len(self.blocks),
                "by_level": levels,
                "hunters": len([b for b in self.blocks.values() if b.is_hunter]),
                "bottom_ais": len([b for b in self.blocks.values() if b.level == 0]),
                "empty_blocks": empty_count,
                "conversation_blocks": conv_count,
                "active_sessions": len(self.sessions.active_sessions),
                "models_with_params": model_count,
                "frozen_ais": frozen_count
            }

    def print_stats(self):
        stats = self.stats()
        print("\n" + "=" * 50)
        print("🦅 捕猎AI生态统计（参数化版本 + 冻结机制）")
        print("=" * 50)
        print(f"总AI数: {stats['total']}")
        print(f"按层级: {stats['by_level']}")
        print(f"捕猎者: {stats['hunters']}")
        print(f"底层AI: {stats['bottom_ais']}")
        print(f"空块: {stats['empty_blocks']}")
        print(f"对话块: {stats['conversation_blocks']}")
        print(f"活跃会话: {stats['active_sessions']}")
        print(f"带有模型参数的块: {stats['models_with_params']}")
        print(f"冻结的AI: {stats['frozen_ais']}")
        print("=" * 50)

    def clear(self):
        with self._lock:
            self.blocks = {}
            self.keyword_index = {}
            self.sessions.active_sessions = {}
        self._save()


# 兼容旧接口
Memory = HunterMemory
MemoryBlock = MemoryBlock