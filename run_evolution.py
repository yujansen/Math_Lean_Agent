"""
全分支数学演化脚本 — 驱动 Turing 在 Lean4/Mathlib 全部数学领域完成定理证明。

覆盖 Mathlib 全部 24 个纯数学分支（26 期任务、130 个定理），包括：
  数论、代数、线性代数、群论、环论、域论、分析、测度论、概率论、
  拓扑、代数拓扑、几何、代数几何、范畴论、组合数学、序与格、
  逻辑、集合论、可计算性、动力系统、模型论、信息论、凝聚数学、表示论。

每个成功的证明自动命名、分类并存入定理库。演化结果保存为 JSON 供后续分析。

用法::

    python run_evolution.py            # 运行全部 26 期、130 个定理
    TOKENIZERS_PARALLELISM=false python run_evolution.py  # 消除 HF 警告
"""

import asyncio
import json
import time

from turing.agents.turing_agent import TuringAgent

# ======================================================================
#  全分支任务库 — 从基础到进阶，覆盖 Lean/Mathlib 的主要数学分支
# ======================================================================

MATH_BRANCHES = [
    # ------------------------------------------------------------------
    #  Phase 1: 自然数论 (Number Theory - Natural Numbers)
    # ------------------------------------------------------------------
    {
        "name": "自然数论",
        "area": "number_theory",
        "tasks": [
            "证明: 对任意自然数 n，n + 0 = n",
            "证明: 对任意自然数 a 和 b，a + b = b + a（加法交换律）",
            "证明: 对任意自然数 a, b, c，(a + b) + c = a + (b + c)（加法结合律）",
            "证明: 对任意自然数 a 和 b，a * b = b * a（乘法交换律）",
            "证明: 对任意自然数 a, b, c，a * (b + c) = a * b + a * c（左分配律）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 2: 整数与整除性 (Integer Theory & Divisibility)
    # ------------------------------------------------------------------
    {
        "name": "整数与整除性",
        "area": "number_theory",
        "tasks": [
            "证明: 对任意整数 a，a + 0 = a",
            "证明: 对任意整数 a，a + (-a) = 0",
            "证明: 对任意整数 a 和 b，a + b = b + a",
            "证明: 对任意整数 a，a * 1 = a",
            "证明: 对任意整数 a，a * 0 = 0",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 3: 命题逻辑 (Propositional Logic)
    # ------------------------------------------------------------------
    {
        "name": "命题逻辑",
        "area": "logic",
        "tasks": [
            "证明: 对任意命题 P，P → P（同一律）",
            "证明: 对任意命题 P 和 Q，P → Q → P（弱化）",
            "证明: 对任意命题 P 和 Q，P ∧ Q → Q ∧ P（合取交换律）",
            "证明: 对任意命题 P 和 Q，P ∨ Q → Q ∨ P（析取交换律）",
            "证明: 对任意命题 P，¬¬P → P（双重否定消去）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 4: 集合论基础 (Set Theory)
    # ------------------------------------------------------------------
    {
        "name": "集合论",
        "area": "set_theory",
        "tasks": [
            "证明: 对任意集合 S，S ⊆ S（自反性）",
            "证明: 对任意集合 A 和 B，A ∩ B ⊆ A",
            "证明: 对任意集合 A 和 B，A ∩ B = B ∩ A（交集交换律）",
            "证明: 对任意集合 A 和 B，A ∪ B = B ∪ A（并集交换律）",
            "证明: 对任意类型 α 的集合 S，S ∩ ∅ = ∅",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 5: 代数结构 (Abstract Algebra)
    # ------------------------------------------------------------------
    {
        "name": "代数结构",
        "area": "algebra",
        "tasks": [
            "证明: 在任意群 G 中，单位元是唯一的",
            "证明: 在任意群 G 中，对任意元素 a，(a⁻¹)⁻¹ = a",
            "证明: 对任意环 R 和元素 a，0 * a = 0",
            "证明: 对任意环 R 和元素 a 和 b，(-a) * b = -(a * b)",
            "证明: 在任意交换群 G 中，对任意 a 和 b，(a * b)⁻¹ = a⁻¹ * b⁻¹",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 6: 序与格 (Order Theory & Lattices)
    # ------------------------------------------------------------------
    {
        "name": "序与格",
        "area": "order_theory",
        "tasks": [
            "证明: 任意偏序集中的最小元素（如果存在）是唯一的",
            "证明: 对任意全序集中的元素 a 和 b，a ≤ b 或 b ≤ a",
            "证明: 在任意格中，a ⊓ b ≤ a",
            "证明: 在任意格中，a ≤ a ⊔ b",
            "证明: 对任意偏序集中的元素 a，a ≤ a（自反性）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 7: 实分析基础 (Real Analysis)
    # ------------------------------------------------------------------
    {
        "name": "实分析",
        "area": "analysis",
        "tasks": [
            "证明: 对任意实数 x，x + 0 = x",
            "证明: 对任意实数 x 和 y，x + y = y + x",
            "证明: 对任意实数 x，|x| ≥ 0（绝对值非负性）",
            "证明: 对任意实数 x 和 y，|x + y| ≤ |x| + |y|（三角不等式）",
            "证明: 对任意实数 x，x^2 ≥ 0",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 8: 函数与映射 (Functions & Mappings)
    # ------------------------------------------------------------------
    {
        "name": "函数与映射",
        "area": "functions",
        "tasks": [
            "证明: 恒等函数是单射",
            "证明: 恒等函数是满射",
            "证明: 两个单射函数的复合仍是单射",
            "证明: 两个满射函数的复合仍是满射",
            "证明: 对任意函数 f，f ∘ id = f",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 9: 组合数学 (Combinatorics)
    # ------------------------------------------------------------------
    {
        "name": "组合数学",
        "area": "combinatorics",
        "tasks": [
            "证明: 对任意有限集 A 和 B，若 A ⊆ B，则 |A| ≤ |B|（基数单调性）",
            "证明: 空集的基数为 0",
            "证明: 对任意自然数 n，0 ≤ n（自然数非负性）",
            "证明: 对任意自然数 n，n ≤ n + 1",
            "证明: 对任意自然数 n 和 m，若 n ≤ m 且 m ≤ n，则 n = m（反对称性）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 10: 拓扑学基础 (Topology)
    # ------------------------------------------------------------------
    {
        "name": "拓扑学",
        "area": "topology",
        "tasks": [
            "证明: 在任意拓扑空间中，全集是开集",
            "证明: 在任意拓扑空间中，空集是开集",
            "证明: 在任意拓扑空间中，全集是闭集",
            "证明: 在任意度量空间中，距离函数 d(x, x) = 0",
            "证明: 在任意度量空间中，d(x, y) = d(y, x)（对称性）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 11: 线性代数 (Linear Algebra)
    # ------------------------------------------------------------------
    {
        "name": "线性代数",
        "area": "linear_algebra",
        "tasks": [
            "证明: 线性映射保持零向量，即对任意线性映射 f，f(0) = 0",
            "证明: 对任意向量空间中的向量 v，1 • v = v（单位标量乘法）",
            "证明: 对任意向量空间中的向量 v，0 • v = 0（零标量乘法）",
            "证明: 两个线性映射的复合仍是线性映射",
            "证明: 对任意矩阵 A，转置的转置等于 A，即 Aᵀᵀ = A",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 12: 群论 (Group Theory)
    # ------------------------------------------------------------------
    {
        "name": "群论",
        "area": "group_theory",
        "tasks": [
            "证明: 群同态保持单位元，即若 φ: G → H 是群同态，则 φ(1) = 1",
            "证明: 群同态保持逆元，即 φ(a⁻¹) = φ(a)⁻¹",
            "证明: 任意群的子群包含单位元",
            "证明: 交换群的每个子群都是正规子群",
            "证明: 对任意群元素 g，g 的阶等于 g⁻¹ 的阶",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 13: 环论 (Ring Theory)
    # ------------------------------------------------------------------
    {
        "name": "环论",
        "area": "ring_theory",
        "tasks": [
            "证明: 在整环中，若 a * b = 0 则 a = 0 或 b = 0",
            "证明: 在交换环中，(a + b)² = a² + 2 * a * b + b²",
            "证明: 在任意环中，(-1) * (-1) = 1",
            "证明: 在交换环中，极大理想是素理想",
            "证明: 在交换环中，1 加幂零元是可逆的",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 14: 域论 (Field Theory)
    # ------------------------------------------------------------------
    {
        "name": "域论",
        "area": "field_theory",
        "tasks": [
            "证明: 域的特征为 0 或素数",
            "证明: 有理数域 ℚ 的特征为 0",
            "证明: 域中每个非零元素都有乘法逆元",
            "证明: 任意域都是整环",
            "证明: 有限域的元素个数是素数的幂次",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 15: 测度论 (Measure Theory)
    # ------------------------------------------------------------------
    {
        "name": "测度论",
        "area": "measure_theory",
        "tasks": [
            "证明: 空集的测度为 0",
            "证明: 在任意可测空间中，全集是可测集",
            "证明: 在任意可测空间中，空集是可测集",
            "证明: 可测集的补集仍是可测集",
            "证明: 两个可测集的并集仍是可测集",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 16: 概率论 (Probability)
    # ------------------------------------------------------------------
    {
        "name": "概率论",
        "area": "probability",
        "tasks": [
            "证明: 概率测度中全空间的测度等于 1",
            "证明: 空事件的概率为 0",
            "证明: 任意事件的概率不超过 1",
            "证明: 概率是非负的，即对任意事件 A，P(A) ≥ 0",
            "证明: 若 A ⊆ B，则 P(A) ≤ P(B)（概率的单调性）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 17: 几何学 (Geometry)
    # ------------------------------------------------------------------
    {
        "name": "几何学",
        "area": "geometry",
        "tasks": [
            "证明: 两个凸集的交集仍是凸集",
            "证明: 空集是凸集",
            "证明: 全空间是凸集",
            "证明: 凸集中任意两点的中点仍在该凸集中",
            "证明: 内积空间中柯西-施瓦茨不等式 |⟨u, v⟩| ≤ ‖u‖ * ‖v‖",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 18: 范畴论 (Category Theory)
    # ------------------------------------------------------------------
    {
        "name": "范畴论",
        "area": "category_theory",
        "tasks": [
            "证明: 在任意范畴中，恒等态射是左单位元，即 id ≫ f = f",
            "证明: 在任意范畴中，恒等态射是右单位元，即 f ≫ id = f",
            "证明: 态射复合满足结合律，即 (f ≫ g) ≫ h = f ≫ (g ≫ h)",
            "证明: 函子保持恒等态射，即 F.map (𝟙 X) = 𝟙 (F.obj X)",
            "证明: 函子保持态射复合，即 F.map (f ≫ g) = F.map f ≫ F.map g",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 19: 代数几何 (Algebraic Geometry)
    # ------------------------------------------------------------------
    {
        "name": "代数几何",
        "area": "algebraic_geometry",
        "tasks": [
            "证明: 局部环有唯一的极大理想",
            "证明: 域只有两个理想：零理想和全环",
            "证明: 在整环中，零理想是素理想",
            "证明: 交换环中，素理想的补集对乘法封闭",
            "证明: 环同态的核是理想",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 20: 代数拓扑 (Algebraic Topology)
    # ------------------------------------------------------------------
    {
        "name": "代数拓扑",
        "area": "algebraic_topology",
        "tasks": [
            "证明: 同伦等价关系是自反的，即任意连续映射与自身同伦",
            "证明: 同伦等价关系是对称的",
            "证明: 路径连通空间是连通空间",
            "证明: 同胚的空间具有相同的基本群",
            "证明: 可缩空间是路径连通的",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 21: 可计算性理论 (Computability)
    # ------------------------------------------------------------------
    {
        "name": "可计算性",
        "area": "computability",
        "tasks": [
            "证明: 零函数是原始递归的",
            "证明: 后继函数是原始递归的",
            "证明: 原始递归函数的复合仍是原始递归的",
            "证明: 确定性有限自动机(DFA)接受的语言在补运算下封闭",
            "证明: 每个 DFA 都可以转化为等价的 NFA",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 22: 动力系统 (Dynamics)
    # ------------------------------------------------------------------
    {
        "name": "动力系统",
        "area": "dynamics",
        "tasks": [
            "证明: 函数的不动点是周期为 1 的周期点",
            "证明: 恒等映射的每个点都是不动点",
            "证明: 若 x 是 f 的不动点，则 x 也是 f 的 n 次迭代的不动点",
            "证明: 不动点集是周期点集的子集",
            "证明: 对任意函数 f，f 的不动点集关于 f 不变",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 23: 模型论 (Model Theory)
    # ------------------------------------------------------------------
    {
        "name": "模型论",
        "area": "model_theory",
        "tasks": [
            "证明: 任意一阶结构是自身的子结构",
            "证明: 结构嵌入保持原子公式的真值",
            "证明: 同构的结构满足相同的一阶句子",
            "证明: 两个结构的同构关系是对称的",
            "证明: 结构同构的复合仍是结构同构",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 24: 信息论 (Information Theory)
    # ------------------------------------------------------------------
    {
        "name": "信息论",
        "area": "information_theory",
        "tasks": [
            "证明: 汉明距离满足 d(x, x) = 0（自反性）",
            "证明: 汉明距离满足对称性 d(x, y) = d(y, x)",
            "证明: 汉明距离满足三角不等式",
            "证明: 汉明距离是非负的",
            "证明: 对任意两个码字 x 和 y，若汉明距离 d(x, y) = 0 则 x = y",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 25: 凝聚数学 (Condensed Mathematics)
    # ------------------------------------------------------------------
    {
        "name": "凝聚数学",
        "area": "condensed",
        "tasks": [
            "证明: 离散集合可以自然地视为凝聚集合",
            "证明: 凝聚阿贝尔群的直和仍是凝聚阿贝尔群",
            "证明: 凝聚集合的范畴具有有限极限",
            "证明: 凝聚阿贝尔群构成阿贝尔范畴",
            "证明: 离散化函子保持有限乘积",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 26: 表示论 (Representation Theory)
    # ------------------------------------------------------------------
    {
        "name": "表示论",
        "area": "representation_theory",
        "tasks": [
            "证明: 平凡表示是群表示",
            "证明: 两个表示的直和仍是群表示",
            "证明: 有限群在特征为零的域上的表示是完全可约的（Maschke 定理）",
            "证明: 群表示的子表示的补空间也是子表示（在半单情况下）",
            "证明: Schur 引理：不可约表示之间的非零同态是同构",
        ],
    },
]


async def main():
    turing = TuringAgent()
    await turing.initialize()

    all_results = []
    total_success = 0
    total_tasks = 0
    named_theorems = []

    t_start = time.time()

    for phase_idx, branch in enumerate(MATH_BRANCHES, 1):
        print(f"\n{'='*70}")
        print(f"  🔬 第 {phase_idx}/{len(MATH_BRANCHES)} 期: {branch['name']} [{branch['area']}]")
        print(f"{'='*70}")

        phase_results = []
        t0 = time.time()

        for task in branch["tasks"]:
            print(f"\n  📝 {task[:70]}...")
            try:
                result = await turing.process_task(task, area=branch["area"])
            except Exception as e:
                print(f"  → [✗] 异常: {str(e)[:80]}")
                result = {"success": False, "error": str(e), "task": task}
            phase_results.append(result)

            status = "✓" if result.get("success") else "✗"
            naming = result.get("theorem_naming", {})
            thm_name = naming.get("theorem_name", "")
            is_novel = naming.get("is_novel", False)
            novel_tag = " 🆕" if is_novel else ""

            if result.get("success"):
                code = result.get("lean_code", "")
                lines = len(code.strip().splitlines()) if code else 0
                print(f"  → [{status}] {thm_name}{novel_tag} ({lines} 行 Lean)")
                named_theorems.append({
                    "phase": branch["name"],
                    "area": branch["area"],
                    "task": task,
                    "theorem_name": thm_name,
                    "chinese_name": naming.get("chinese_name", ""),
                    "is_novel": is_novel,
                    "description": naming.get("description", ""),
                    "lean_lines": lines,
                    "lean_code": code,
                })
            else:
                err = result.get("last_error", result.get("error", ""))[:60]
                print(f"  → [{status}] {err}")

        elapsed = time.time() - t0
        successes = sum(1 for r in phase_results if r.get("success"))
        total_success += successes
        total_tasks += len(branch["tasks"])

        # 分期报告
        print(f"\n  {'─'*50}")
        print(f"  📊 {branch['name']}报告: {successes}/{len(branch['tasks'])} ({100*successes//len(branch['tasks'])}%) | {elapsed:.0f}s")
        for i, (task, result) in enumerate(zip(branch["tasks"], phase_results), 1):
            s = "✓" if result.get("success") else "✗"
            nm = result.get("theorem_naming", {}).get("theorem_name", "—")
            nv = " 🆕" if result.get("theorem_naming", {}).get("is_novel") else ""
            print(f"    {i}. [{s}] {nm}{nv}")

        all_results.append({
            "phase": branch["name"],
            "area": branch["area"],
            "results": phase_results,
            "successes": successes,
            "total": len(branch["tasks"]),
        })

        # 每个分支后触发演化
        print(f"\n  🔄 触发演化...")
        await turing.evaluate_and_evolve()

    total_elapsed = time.time() - t_start

    # ==================================================================
    #  最终报告
    # ==================================================================
    print(f"\n{'🏆'*35}")
    print(f"\n  📋 全分支数学演化 — 最终报告")
    print(f"  {'='*60}")
    print(f"  总任务: {total_tasks}")
    print(f"  总成功: {total_success} ({100*total_success//total_tasks}%)")
    print(f"  总用时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print()

    for ar in all_results:
        pct = 100 * ar["successes"] // ar["total"]
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {ar['phase']:10s} [{ar['area']:15s}] {bar} {ar['successes']}/{ar['total']} ({pct}%)")

    # 原创定理报告
    novel_list = [t for t in named_theorems if t.get("is_novel")]
    print(f"\n  {'─'*60}")
    print(f"  📚 已命名定理: {len(named_theorems)}")
    print(f"  🆕 原创定理 (非 Mathlib): {len(novel_list)}")

    if novel_list:
        print(f"\n  🆕 原创定理列表:")
        for t in novel_list:
            print(f"    • {t['theorem_name']} ({t.get('chinese_name', '')})")
            print(f"      [{t['area']}] {t['description'][:60]}")
            print(f"      Lean: {t['lean_lines']} 行")

    print(f"\n  📚 全部命名定理:")
    for t in named_theorems:
        nv = " 🆕" if t.get("is_novel") else ""
        print(f"    • [{t['area']}] {t['theorem_name']}{nv} — {t['description'][:50]}")

    # 保存定理库到 JSON
    output_path = "./data/theorem_library.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_tasks": total_tasks,
                "total_success": total_success,
                "success_rate": f"{100*total_success//total_tasks}%",
                "total_elapsed_seconds": total_elapsed,
                "branches": [
                    {
                        "name": ar["phase"],
                        "area": ar["area"],
                        "success": ar["successes"],
                        "total": ar["total"],
                    }
                    for ar in all_results
                ],
                "named_theorems": named_theorems,
                "novel_theorems": novel_list,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 定理库已保存到: {output_path}")
    except Exception as e:
        print(f"\n  ⚠️ 保存失败: {e}")

    print(f"\n{'🏆'*35}")

    await turing.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
