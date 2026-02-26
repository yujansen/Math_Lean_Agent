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
    #  使用 LinearAlgebra, LinearMap, Matrix 模块
    # ------------------------------------------------------------------
    {
        "name": "线性代数",
        "area": "linear_algebra",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，对 Module R M 和线性映射 f : M →ₗ[R] M，f 0 = 0（使用 LinearMap.map_zero 或 map_zero）",
            "证明: 在 Lean4/Mathlib 中，对 Module R M 和 v : M，(1 : R) • v = v（使用 one_smul）",
            "证明: 在 Lean4/Mathlib 中，对 Module R M 和 v : M，(0 : R) • v = 0（使用 zero_smul）",
            "证明: 在 Lean4/Mathlib 中，对 Matrix m n R，(Aᵀ)ᵀ = A，即 Matrix.transpose_transpose A（使用 Matrix.transpose_transpose）",
            "证明: 在 Lean4/Mathlib 中，对线性映射 f : M →ₗ[R] N 和 x y : M，f (x + y) = f x + f y（使用 map_add）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 12: 群论 (Group Theory)
    #  使用 GroupTheory, MulAction 模块
    # ------------------------------------------------------------------
    {
        "name": "群论",
        "area": "group_theory",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，群同态保持单位元，即对 MonoidHom φ : G →* H，φ 1 = 1（使用 map_one）",
            "证明: 在 Lean4/Mathlib 中，群同态保持逆元，即对 MonoidHom φ : G →* H 和 [Group G] [Group H]，φ a⁻¹ = (φ a)⁻¹（使用 map_inv）",
            "证明: 在 Lean4/Mathlib 中，对 Group G 和 Subgroup H，(1 : G) ∈ H（子群包含单位元，使用 Subgroup.one_mem 或 one_mem）",
            "证明: 在 Lean4/Mathlib 中，对 CommGroup G 和 a b : G，(a * b)⁻¹ = a⁻¹ * b⁻¹（使用 mul_inv 或 CommGroup 实例）",
            "证明: 在 Lean4/Mathlib 中，对 Group G 和 a : G，a * a⁻¹ = 1（使用 mul_inv_cancel）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 13: 环论 (Ring Theory)
    #  使用 Algebra.Ring, Ideal 模块
    # ------------------------------------------------------------------
    {
        "name": "环论",
        "area": "ring_theory",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，在整环 [IsDomain R] 中，若 a * b = 0 则 a = 0 或 b = 0（使用 mul_eq_zero.mp 或 eq_zero_or_eq_zero_of_mul_eq_zero）",
            "证明: 在 Lean4/Mathlib 中，在任意环中，(-1 : R) * (-1 : R) = 1（使用 neg_one_mul_neg_one 或 neg_mul_neg）",
            "证明: 在 Lean4/Mathlib 中，在交换环中，对 a b : R，(a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2（使用 add_pow_two 或 ring）",
            "证明: 在 Lean4/Mathlib 中，对 CommRing R 和 Ideal I，极大理想是素理想，即 Ideal.IsMaximal I → Ideal.IsPrime I",
            "证明: 在 Lean4/Mathlib 中，对任意环 R 和 a : R，a * 0 = 0（使用 mul_zero）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 14: 域论 (Field Theory)
    #  使用 FieldTheory, CharP 模块
    # ------------------------------------------------------------------
    {
        "name": "域论",
        "area": "field_theory",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，域中每个非零元素都有乘法逆元，即对 Field k 和 a : k，a ≠ 0 → a * a⁻¹ = 1（使用 mul_inv_cancel₀）",
            "证明: 在 Lean4/Mathlib 中，任意域都是整环，即 Field k → IsDomain k（使用 Field.isDomain 实例）",
            "证明: 在 Lean4/Mathlib 中，有理数域的特征为 0，即 CharZero ℚ（这是 Mathlib 中的自动实例）",
            "证明: 在 Lean4/Mathlib 中，对 Field k 和 a b : k，b ≠ 0 → a / b * b = a（使用 div_mul_cancel₀）",
            "证明: 在 Lean4/Mathlib 中，对 Field k 和 a : k，a / 1 = a（使用 div_one）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 15: 测度论 (Measure Theory)
    #  使用 MeasureTheory 命名空间，import Mathlib.MeasureTheory
    # ------------------------------------------------------------------
    {
        "name": "测度论",
        "area": "measure_theory",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，对任意 MeasurableSpace α 和 Measure α，空集的测度为 0，即 μ ∅ = 0（使用 measure_empty）",
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，Set.univ 是可测集，即 MeasurableSet (Set.univ : Set α)",
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，空集是可测集，即 MeasurableSet (∅ : Set α)",
            "证明: 在 Lean4/Mathlib 中，对任意可测函数 f g : α → β，f + g 仍是可测的（Measurable f → Measurable g → Measurable (f + g)，需要 Add β 和 MeasurableAdd₂）",
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，测度是单调的，若 s ⊆ t 且 MeasurableSet s，则 μ s ≤ μ t（使用 measure_mono）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 16: 概率论 (Probability)
    #  使用 MeasureTheory + ProbabilityTheory 命名空间
    # ------------------------------------------------------------------
    {
        "name": "概率论",
        "area": "probability",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，若 μ 是 IsProbabilityMeasure，则 μ Set.univ = 1（使用 measure_univ）",
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，若 μ 是 IsProbabilityMeasure，则 μ ∅ = 0（ProbabilityMeasure 下空集的概率为 0）",
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，对任意测度 μ 和可测集 s，μ s ≤ μ Set.univ（单个事件测度不超过全空间）",
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，测度是非负的，即对任意 s，0 ≤ μ s（使用 ENNReal 自动成立，zero_le）",
            "证明: 在 Lean4/Mathlib 中，open MeasureTheory，若 s ⊆ t，则 μ s ≤ μ t（测度的单调性，使用 measure_mono）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 17: 几何学 (Geometry / Convex Analysis)
    #  使用 Analysis.InnerProductSpace 和 Convex 模块
    # ------------------------------------------------------------------
    {
        "name": "几何学",
        "area": "geometry",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，open Convex，空集是凸集，即 Convex ℝ (∅ : Set ℝ)（使用 convex_empty）",
            "证明: 在 Lean4/Mathlib 中，open Convex，全空间是凸集，即 Convex ℝ (Set.univ : Set ℝ)（使用 convex_univ）",
            "证明: 在 Lean4/Mathlib 中，对任意内积空间 E 和 x : E，‖x‖ ≥ 0（范数非负性，使用 norm_nonneg）",
            "证明: 在 Lean4/Mathlib 中，对任意内积空间 E 和 x : E，‖0‖ = 0（零向量范数为 0，使用 norm_zero）",
            "证明: 在 Lean4/Mathlib 中，对任意赋范空间中的 x y : E，‖x + y‖ ≤ ‖x‖ + ‖y‖（三角不等式，使用 norm_add_le）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 18: 范畴论 (Category Theory)
    #  使用 CategoryTheory 命名空间，open CategoryTheory
    # ------------------------------------------------------------------
    {
        "name": "范畴论",
        "area": "category_theory",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，恒等态射是左单位元，即 𝟙 X ≫ f = f（使用 Category.id_comp）",
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，恒等态射是右单位元，即 f ≫ 𝟙 Y = f（使用 Category.comp_id）",
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，态射复合满足结合律，即 (f ≫ g) ≫ h = f ≫ (g ≫ h)（使用 Category.assoc）",
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，函子保持恒等态射，即 F.map (𝟙 X) = 𝟙 (F.obj X)（使用 Functor.map_id）",
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，函子保持态射复合，即 F.map (f ≫ g) = F.map f ≫ F.map g（使用 Functor.map_comp）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 19: 代数几何 (Algebraic Geometry / Commutative Algebra)
    #  使用 RingTheory, LocalRing, Ideal 模块
    # ------------------------------------------------------------------
    {
        "name": "代数几何",
        "area": "algebraic_geometry",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，域中零理想是素理想，即对 Field k，Ideal.IsPrime (⊥ : Ideal k)（使用 Ideal.bot_prime）",
            "证明: 在 Lean4/Mathlib 中，环同态的核是理想，即 RingHom.ker f 是 Ideal（这在 Mathlib 中是定义自动成立）",
            "证明: 在 Lean4/Mathlib 中，域只有两个理想：对 Field k 和 Ideal k，该 Ideal 要么是 ⊥ 要么是 ⊤（使用 Ideal.eq_bot_or_top）",
            "证明: 在 Lean4/Mathlib 中，对 CommRing R，极大理想是素理想，即 Ideal.IsMaximal I → Ideal.IsPrime I（使用 Ideal.IsMaximal.isPrime）",
            "证明: 在 Lean4/Mathlib 中，对 CommRing R 和 Ideal I J，若 I ≤ J 则 R ⧸ J 是 R ⧸ I 的商（使用 Ideal.quotientMap）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 20: 代数拓扑 (Algebraic Topology)
    #  使用 Topology.Homotopy, Topology.Connected 模块
    # ------------------------------------------------------------------
    {
        "name": "代数拓扑",
        "area": "algebraic_topology",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，路径连通空间是连通空间，即 PathConnectedSpace α → ConnectedSpace α（使用 PathConnectedSpace.connectedSpace）",
            "证明: 在 Lean4/Mathlib 中，对任意拓扑空间 α 和 x : α，x 与自身路径连通，即 Joined x x（使用 Joined.refl）",
            "证明: 在 Lean4/Mathlib 中，若 Joined x y 则 Joined y x（路径连通是对称的，使用 Joined.symm）",
            "证明: 在 Lean4/Mathlib 中，open TopologicalSpace，连通空间中 connectedComponents 的唯一性：对连通空间，connectedComponent x = Set.univ",
            "证明: 在 Lean4/Mathlib 中，恒等映射是连续的，即 Continuous (id : α → α)（使用 continuous_id）",
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
    #  使用 ModelTheory.Basic 和 FirstOrder 模块
    # ------------------------------------------------------------------
    {
        "name": "模型论",
        "area": "model_theory",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，open FirstOrder Language，对任意一阶语言 L 和结构 M [L.Structure M]，恒等嵌入是嵌入（Embedding.refl L M）",
            "证明: 在 Lean4/Mathlib 中，一阶理论中空理论的模型就是所有结构（Theory.Model.of_trivial）",
            "证明: 在 Lean4/Mathlib 中，对任意类型 α，α 上的等价关系 Eq 是等价关系（使用 eq_equivalence）",
            "证明: 在 Lean4/Mathlib 中，对任意 DecidableEq α 和 a : α，({a} : Finset α).card = 1（使用 Finset.card_singleton）",
            "证明: 在 Lean4/Mathlib 中，对任意类型 α 和 BEq α 实例，∀ a : α, a == a（使用 BEq.refl 或 beq_self_eq_true）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 24: 信息论 (Information Theory)
    #  使用 Combinatorics.SimpleGraph, Nat.dist 等模块
    # ------------------------------------------------------------------
    {
        "name": "信息论",
        "area": "information_theory",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，对 α [DecidableEq α] 和 x : α → Fin n，hammingDist x x = 0（汉明距离自反性）",
            "证明: 在 Lean4/Mathlib 中，对 α [DecidableEq α] 和 x y : α → Fin n，hammingDist x y = hammingDist y x（汉明距离对称性）",
            "证明: 在 Lean4/Mathlib 中，Nat.dist 满足 Nat.dist n n = 0（自然数距离自反性）",
            "证明: 在 Lean4/Mathlib 中，Nat.dist 满足对称性 Nat.dist m n = Nat.dist n m（使用 Nat.dist_comm）",
            "证明: 在 Lean4/Mathlib 中，对有限类型 α [Fintype α] 和 s : Finset α，s.card ≤ Fintype.card α（使用 Finset.card_le_univ）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 25: 凝聚数学 (Condensed Mathematics)
    #  使用 Condensed, Sheaf 模块
    # ------------------------------------------------------------------
    {
        "name": "凝聚数学",
        "area": "condensed",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，对任意范畴 C 的对象 X，𝟙 X ≫ 𝟙 X = 𝟙 X（恒等态射幂等，使用 Category.id_comp）",
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，对任意函子 F : C ⥤ D，F.map (𝟙 X) = 𝟙 (F.obj X)（函子保持恒等）",
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，自然变换的恒等存在：NatTrans.id F 是从 F 到 F 的自然变换",
            "证明: 在 Lean4/Mathlib 中，对 AddCommGroup G，G 的直和 G ⊕ G 仍是 AddCommGroup（使用 instAddCommGroupProd 或 Prod.instAddCommGroup）",
            "证明: 在 Lean4/Mathlib 中，open CategoryTheory，对任意范畴 C 的对象 X Y 和态射 f : X ⟶ Y，f ≫ 𝟙 Y = f（使用 Category.comp_id）",
        ],
    },
    # ------------------------------------------------------------------
    #  Phase 26: 表示论 (Representation Theory)
    #  使用 Representation, Module 模块
    # ------------------------------------------------------------------
    {
        "name": "表示论",
        "area": "representation_theory",
        "tasks": [
            "证明: 在 Lean4/Mathlib 中，对任意 Module R M，零子模块 ⊥ 是子模块（使用 Submodule.bot_mem 相关引理）",
            "证明: 在 Lean4/Mathlib 中，对任意 R-Module M 和 Submodule R M，子模块对加法封闭（使用 Submodule.add_mem）",
            "证明: 在 Lean4/Mathlib 中，对任意 R-Module M，Module.End R M 构成环（使用 Ring (Module.End R M)，这是自动实例）",
            "证明: 在 Lean4/Mathlib 中，对 Module R M 和线性映射 f g : M →ₗ[R] M，f + g 仍是线性映射（使用 LinearMap.add）",
            "证明: 在 Lean4/Mathlib 中，对 Module R M，恒等线性映射 LinearMap.id 满足 LinearMap.id x = x（使用 LinearMap.id_apply）",
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
