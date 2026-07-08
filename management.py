"""
AI 工厂模块
===========
功能：
- 创建AI（自动分配系统名 AI_01, AI_02, ...）
- 训练AI（生成 .pth 权重文件）
- 管理AI（删除、重命名、查看）
- 支持别名
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
from pathlib import Path

# 导入配置
from config import FACTORY_CONFIG, DATA_DIR

# 导入模型
from model import UntrainedSuperModel

# ========== 配置 ==========
DEFAULT_DIM = FACTORY_CONFIG.get("default_dim", 64)
DEFAULT_LAYERS = FACTORY_CONFIG.get("default_layers", 2)
DEFAULT_HEADS = FACTORY_CONFIG.get("default_heads", 2)
DEFAULT_LR = FACTORY_CONFIG.get("default_lr", 0.01)

WORK_DIR = Path(__file__).parent
BRAIN_DIR = DATA_DIR / "brains"
REGISTRY_FILE = DATA_DIR / "registry.json"

BRAIN_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ========== 内部工具 ==========

def _load_registry():
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_registry(registry):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

def _get_brain_path(sys_name):
    return BRAIN_DIR / f"{sys_name}.pth"

def _generate_system_name(registry):
    existing = [k for k in registry.keys() if k.startswith("AI_")]
    if not existing:
        return "AI_01"
    max_num = 0
    for name in existing:
        if '_' not in name[3:]:
            try:
                num = int(name[3:])
                if num > max_num:
                    max_num = num
            except:
                continue
        else:
            parts = name[3:].split('_')
            if len(parts) == 2:
                try:
                    group = int(parts[0])
                    seq = int(parts[1])
                    num = (group - 1) * 99 + seq
                    if num > max_num:
                        max_num = num
                except:
                    continue
    next_num = max_num + 1
    if next_num <= 99:
        return f"AI_{next_num:02d}"
    else:
        group = (next_num - 100) // 99 + 1
        seq = (next_num - 100) % 99 + 1
        return f"AI_{group:02d}_{seq:02d}"

def _resolve_name(name, registry):
    if name in registry:
        return name
    for sys_name, cfg in registry.items():
        if cfg.get("alias") == name:
            return sys_name
    return None

def _generate_unique_alias(base_alias, registry):
    if base_alias is None or base_alias.startswith("AI_"):
        return None
    used_aliases = [cfg.get("alias") for cfg in registry.values() if cfg.get("alias")]
    used_sys_names = list(registry.keys())
    if base_alias not in used_aliases and base_alias not in used_sys_names:
        return base_alias
    suffix = 1
    while True:
        new_alias = f"{base_alias}_{suffix}"
        if new_alias not in used_aliases and new_alias not in used_sys_names:
            return new_alias
        suffix += 1


# ========== 核心函数 ==========

def AI_list(number=None, data=None, epochs=None, custom_func=None):
    """批量创建AI"""
    if number is None or data is None or epochs is None:
        return {"success": False, "message": "number, data, epochs 都不能为空"}

    if isinstance(data, str):
        data_list = [s.strip() for s in data.split() if s.strip()]
    else:
        data_list = data
    if not data_list:
        return {"success": False, "message": "训练数据为空"}

    registry = _load_registry()
    created = []
    created_aliases = []

    for i in range(number):
        sys_name = _generate_system_name(registry)
        alias = None
        block_data = data_list
        block_epochs = epochs

        if custom_func and callable(custom_func):
            try:
                ret = custom_func(sys_name, i, data_list, epochs)
                if isinstance(ret, dict):
                    alias = ret.get("alias")
                    block_data = ret.get("data", block_data)
                    block_epochs = ret.get("epochs", block_epochs)
                    if alias:
                        alias = _generate_unique_alias(alias, registry)
            except Exception as e:
                print(f"⚠️ 自定义函数错误 (第{i+1}个): {e}")

        registry[sys_name] = {
            "data": block_data,
            "epochs": block_epochs,
            "dim": DEFAULT_DIM,
            "layers": DEFAULT_LAYERS,
            "heads": DEFAULT_HEADS,
            "lr": DEFAULT_LR,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trained": False,
            "final_loss": None,
            "data_size": len(block_data),
            "alias": alias,
            "brain_path": None
        }
        created.append(sys_name)
        created_aliases.append(alias)

    _save_registry(registry)
    return {
        "success": True,
        "message": f"成功创建 {len(created)} 个AI",
        "created": created,
        "created_aliases": created_aliases,
        "total": len(created)
    }


def AI_train(name=None):
    """训练AI"""
    registry = _load_registry()
    if not registry:
        return {"success": False, "message": "没有AI可训练"}

    if name:
        sys_name = _resolve_name(name, registry)
        if not sys_name:
            return {"success": False, "message": f"AI '{name}' 不存在"}
        targets = [sys_name]
    else:
        targets = [n for n, cfg in registry.items() if not cfg.get("trained")]
        if not targets:
            return {"success": True, "message": "所有AI均已训练", "results": []}

    results = []
    for sys_name in targets:
        config = registry[sys_name]
        loss = None  # 👈 初始化 loss，防止未定义
        try:
            data = config["data"]
            chars = sorted(set(''.join(data)))
            stoi = {ch: i + 1 for i, ch in enumerate(chars)}
            vocab_size = len(stoi) + 1
            pad_idx = 0

            max_len = max(len(t) for t in data)
            X_list = []
            for t in data:
                ids = [stoi[ch] for ch in t] + [pad_idx] * (max_len - len(t))
                X_list.append(torch.tensor(ids))
            X = torch.stack(X_list)
            Y = X[:, 1:]
            X = X[:, :-1]

            model = UntrainedSuperModel(
                vocab_size=vocab_size,
                dim=config["dim"],
                n_layers=config["layers"],
                n_heads=config["heads"]
            )
            optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
            loss_fn = nn.CrossEntropyLoss()

            model.train()
            for epoch in range(config["epochs"]):
                logits = model(X)
                loss = loss_fn(logits.reshape(-1, vocab_size), Y.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            brain_path = _get_brain_path(sys_name)
            torch.save(model.state_dict(), brain_path)

            config["trained"] = True
            config["final_loss"] = loss.item() if loss is not None else float('inf')
            config["brain_path"] = str(brain_path)
            config["trained_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

            results.append({
                "sys_name": sys_name,
                "success": True,
                "loss": config["final_loss"],
                "brain_path": str(brain_path)
            })
        except Exception as e:
            results.append({"sys_name": sys_name, "success": False, "error": str(e)})

    _save_registry(registry)
    return {"success": True, "results": results, "total": len(results)}


def AI_chat(name, seed_char=None, max_tokens=10):
    """与AI对话"""
    registry = _load_registry()
    sys_name = _resolve_name(name, registry)
    if not sys_name:
        return {"success": False, "message": f"AI '{name}' 不存在", "result": ""}

    config = registry[sys_name]
    brain_path = _get_brain_path(sys_name)
    if not brain_path.exists():
        return {"success": False, "message": f"AI '{sys_name}' 尚未训练", "result": ""}

    data = config["data"]
    chars = sorted(set(''.join(data)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    itos = {i + 1: ch for i, ch in enumerate(chars)}
    pad_idx = 0
    vocab_size = len(stoi) + 1

    model = UntrainedSuperModel(
        vocab_size=vocab_size,
        dim=config["dim"],
        n_layers=config["layers"],
        n_heads=config["heads"]
    )
    model.load_state_dict(torch.load(brain_path, weights_only=True))
    model.eval()

    if seed_char is None:
        seed_char = data[0][0]
    if seed_char not in stoi:
        return {"success": False, "message": f"字符 '{seed_char}' 不在词表中", "result": ""}

    start = torch.tensor([[stoi[seed_char]]])
    with torch.no_grad():
        generated = model.generate(start, max_new_tokens=max_tokens)

    result = ''.join([itos.get(int(id), '') for id in generated[0] if int(id) != pad_idx])
    return {"success": True, "result": result, "sys_name": sys_name}


def AI_show():
    """查看所有AI"""
    registry = _load_registry()
    if not registry:
        return {"success": True, "ais": [], "message": "工作区为空"}

    ais = []
    for sys_name, config in registry.items():
        ais.append({
            "sys_name": sys_name,
            "alias": config.get("alias"),
            "data_size": config.get("data_size"),
            "trained": config.get("trained", False),
            "final_loss": config.get("final_loss"),
            "brain_path": config.get("brain_path"),
            "created": config.get("created"),
            "epochs": config.get("epochs")
        })
    return {"success": True, "ais": ais, "total": len(ais)}


def AI_delete(name):
    """删除AI"""
    registry = _load_registry()
    sys_name = _resolve_name(name, registry)
    if not sys_name:
        return {"success": False, "message": f"AI '{name}' 不存在"}

    brain_path = _get_brain_path(sys_name)
    if brain_path.exists():
        brain_path.unlink()

    del registry[sys_name]
    _save_registry(registry)
    return {"success": True, "message": f"AI '{sys_name}' 已删除"}


def AI_clear():
    """清空所有AI"""
    registry = _load_registry()
    if not registry:
        return {"success": True, "message": "工作区已为空"}

    for sys_name in list(registry.keys()):
        brain_path = _get_brain_path(sys_name)
        if brain_path.exists():
            brain_path.unlink()

    _save_registry({})
    return {"success": True, "message": f"已清空所有AI"}


def AI_info(name):
    """查看AI详情"""
    registry = _load_registry()
    sys_name = _resolve_name(name, registry)
    if not sys_name:
        return {"success": False, "message": f"AI '{name}' 不存在"}

    config = registry[sys_name]
    brain_path = _get_brain_path(sys_name)
    return {
        "success": True,
        "sys_name": sys_name,
        "config": config,
        "brain_exists": brain_path.exists(),
        "brain_size": brain_path.stat().st_size if brain_path.exists() else 0
    }


def AI_alias(name, new_alias):
    """设置别名"""
    registry = _load_registry()
    sys_name = _resolve_name(name, registry)
    if not sys_name:
        return {"success": False, "message": f"AI '{name}' 不存在"}

    if new_alias.startswith("AI_"):
        return {"success": False, "message": "别名不能以 'AI_' 开头"}

    final_alias = _generate_unique_alias(new_alias, registry)
    if not final_alias:
        return {"success": False, "message": "别名无效"}

    registry[sys_name]["alias"] = final_alias
    _save_registry(registry)
    return {"success": True, "message": f"AI '{sys_name}' 别名已设为 '{final_alias}'"}


# ========== 导出 ==========

__all__ = [
    "AI_list",
    "AI_train",
    "AI_chat",
    "AI_show",
    "AI_delete",
    "AI_clear",
    "AI_info",
    "AI_alias"
]