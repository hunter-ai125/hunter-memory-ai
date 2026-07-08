#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🧪 捕猎AI功能测试
"""

from hunter_memory import HunterMemory


def test_hunter():
    print("=" * 60)
    print("🧪 捕猎AI测试")
    print("=" * 60)

    memory = HunterMemory(persist_dir="./test_memory")

    # 添加原始数据
    print("\n📝 添加原始数据...")
    ids = []
    data = [
        ("残响城是一座被时间包裹的虚拟城市", {"残响城": 0.9, "时间": 0.7, "虚拟": 0.6}),
        ("残响城源于时序混乱事件", {"残响城": 0.8, "时序": 0.7, "事件": 0.5}),
        ("时崎狂三拥有刻刻帝", {"时崎狂三": 0.9, "刻刻帝": 0.8, "能力": 0.6}),
        ("刻刻帝可以操控时间", {"刻刻帝": 0.8, "时间": 0.7, "操控": 0.5}),
        ("职阶体系允许角色加载特殊能力", {"职阶": 0.8, "角色": 0.6, "能力": 0.7}),
        ("每个角色只能拥有一个职阶", {"角色": 0.7, "职阶": 0.6, "限制": 0.5}),
    ]
    for content, keywords in data:
        bid = memory.add(content, keywords=keywords)
        ids.append(bid)
        print(f"   ✅ {bid[:8]}: {content[:20]}...")

    memory.print_stats()

    # L1合并
    print("\n🔄 L1合并...")
    h1 = memory.merge(ids[0], ids[1])
    h2 = memory.merge(ids[2], ids[3])
    print(f"   ✅ 合并完成: {h1[:8]}, {h2[:8]}")

    # L2合并
    print("\n🔄 L2合并...")
    h3 = memory.merge(h1, h2)
    print(f"   ✅ 合并完成: {h3[:8]}")

    # L3合并（触发清理）
    print("\n🔄 L3合并（触发清理）...")
    h4 = memory.merge(h3, memory.add("新数据: 残响城和时间的关系"))
    print(f"   ✅ 合并完成: {h4[:8]}")

    # 查询测试
    print("\n🔍 查询测试...")
    results = memory.query("残响城", top_k=3)
    for r in results:
        print(f"   [{r['score']:.2f}] L{r['level']} {r['id'][:8]}: {r['content'][:30]}...")

    memory.print_stats()

    # 主动合并测试
    print("\n⚔️ 主动合并测试...")
    # 创建两个L4 AI
    h5 = memory.merge(h4, memory.add("更多残响城数据..."))
    h6 = memory.merge(h5, memory.add("更多时间数据..."))

    print(f"   h6: {h6[:8]} (L{memory.get(h6).level})")
    result = memory.active_merge(h6, memory.add("完全不同的话题..."))
    print(f"   结果: {result['status']}")

    memory.print_stats()
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    test_hunter()