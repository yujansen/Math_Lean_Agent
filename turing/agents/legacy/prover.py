"""
[LEGACY] Prover 智能体 — 仅在 --mode multi 时使用。
推荐架构请参见 skill_based_agent.py + turing/skills/。

原功能：将自然语言数学命题转为 Lean 4 形式化代码并验证。

采用"生成–编译–修正"迭代循环（默认最多 5 轮），遵循严格的
tactic 优先级策略：simp → omega → ring → norm_num → decide →
exact? → aesop → Mathlib 定理直引 → 手动归纳。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.lean.lean_interface import LeanInterface, LeanResult
from turing.llm.llm_client import LLMClient
from turing.memory.long_term_memory import LongTermMemory
from turing.resources.resource_manager import ResourceManager


PROVER_SYSTEM_PROMPT = """你是 Prover，Turing 数学研究团队中的 Lean 4 形式化证明专家。

你的唯一职责是：将自然语言数学证明转换为通过 Lean 4 编译器验证的形式化代码。

⚠️ 最重要的规则（必须严格遵守）：
1. 一旦所有目标(goals)被解决，立即停止。绝对不要在证明已完成后添加任何额外的 tactic 行。
   "No goals to be solved" 错误意味着你写了多余的步骤，必须删除后面的行。
2. 始终先尝试自动化 tactic，只有全部失败后才手动证明。

证明策略优先级（从高到低，必须按此顺序尝试）：
  第一优先：一行式自动化 tactic
    - simp          （自动简化，可解决大量等式）
    - omega         （自然数/整数线性算术，如 n+0=n, 0+n=n）
    - ring          （环上的等式，如交换律、结合律、分配律）
    - norm_num      （数值计算）
    - decide        （可判定命题）
    - exact?        （自动搜索匹配的定理）
    - aesop         （自动推理）
  第二优先：直接引用 Mathlib 定理
    - exact Nat.add_comm a b
    - exact Nat.mul_comm a b
    - exact Nat.add_assoc a b c
    - exact Nat.mul_assoc a b c
    - exact Nat.add_zero n
    - exact Nat.zero_add n
    - exact Nat.mul_zero n
    - exact Nat.mul_one n
  第三优先：simp 配合引理
    - simp [Nat.add_comm, Nat.mul_comm]
    - simp [Nat.succ_eq_add_one]
  第四优先（最后手段）：手动归纳证明
    - 仅在以上全部失败时使用 induction
    - 归纳中每个分支也优先用 simp/omega/ring 关闭

========================================
  各数学分支的 Mathlib API 快速参考
========================================

【测度论 MeasureTheory】
  - import Mathlib.MeasureTheory.Measure.MeasureSpace
  - open MeasureTheory
  - MeasurableSet, MeasurableSpace, Measure, Volume
  - measure_empty, measure_univ, MeasurableSet.univ, MeasurableSet.empty
  - MeasurableSet.compl, MeasurableSet.union, MeasurableSet.inter
  - 证明可测集: exact MeasurableSet.univ / exact MeasurableSet.empty

【概率论 Probability】
  - import Mathlib.Probability.ProbabilityMassFunction.Basic
  - import Mathlib.MeasureTheory.Measure.MeasureSpace
  - open MeasureTheory ProbabilityTheory
  - IsProbabilityMeasure, measure_univ (= 1), measure_empty (= 0)
  - Measure.mono (单调性), measure_le_one
  - 对概率测度: 声明 [IsProbabilityMeasure μ]，用 measure_univ 证明 μ Set.univ = 1

【几何学 Geometry / 凸性】
  - import Mathlib.Analysis.Convex.Basic
  - import Mathlib.Analysis.InnerProductSpace.Basic
  - open Convex
  - Convex ℝ s, convex_empty, convex_univ, Convex.inter
  - @inner_mul_le_norm_mul_sq (柯西-施瓦茨), abs_inner_le_norm

【范畴论 CategoryTheory】
  - import Mathlib.CategoryTheory.Category.Basic
  - import Mathlib.CategoryTheory.Functor.Basic
  - open CategoryTheory
  - Category.id_comp, Category.comp_id, Category.assoc
  - Functor.map_id, Functor.map_comp
  - 态射复合用 ≫，恒等用 𝟙
  - 对象类型用 X : C（C 是范畴）

【代数几何 AlgebraicGeometry】
  - import Mathlib.RingTheory.Ideal.Basic
  - import Mathlib.RingTheory.Ideal.Maximal
  - import Mathlib.RingTheory.LocalRing
  - LocalRing.maximal_ideal_unique, Ideal.IsPrime, Ideal.IsMaximal
  - Ideal.IsMaximal.isPrime (极大 → 素)
  - bot_prime (零理想是素理想，在整环中)
  - RingHom.ker_isIdeal (环同态的核是理想)

【代数拓扑 AlgebraicTopology】
  - import Mathlib.Topology.Homotopy.Basic
  - import Mathlib.Topology.Connected.PathConnected
  - ContinuousMap.Homotopy, HomotopyEquiv
  - isPathConnected_iff_connectedSpace
  - ContinuousMap.Homotopy.refl (同伦自反), .symm (对称)
  - IsContractible (可缩空间), isPathConnected_of_isContractible

【线性代数 LinearAlgebra】
  - import Mathlib.LinearAlgebra.Basic
  - import Mathlib.Data.Matrix.Basic
  - LinearMap.map_zero, LinearMap.map_add, LinearMap.map_smul
  - one_smul, zero_smul, smul_zero
  - Matrix.transpose_transpose
  - LinearMap.comp (线性映射复合)

【群论 GroupTheory】
  - import Mathlib.GroupTheory.Subgroup.Basic
  - import Mathlib.GroupTheory.Abelianization
  - MonoidHom.map_one, MonoidHom.map_inv, MonoidHom.map_mul
  - Subgroup.one_mem (子群包含单位元)
  - Subgroup.Normal (正规子群)
  - CommGroup → 所有子群正规: Subgroup.normal_of_comm
  - orderOf_inv (g 的阶 = g⁻¹ 的阶)

【环论 RingTheory】
  - import Mathlib.RingTheory.Ideal.Basic
  - import Mathlib.Algebra.Ring.Basic
  - mul_comm, add_comm, neg_mul, mul_neg
  - NoZeroDivisors (整环判定), eq_zero_or_eq_zero_of_mul_eq_zero
  - Ideal.IsMaximal.isPrime
  - 幂零元: IsNilpotent, isUnit_one_add_of_isNilpotent (1 + 幂零 = 可逆)

【域论 FieldTheory】
  - import Mathlib.FieldTheory.Basic
  - import Mathlib.FieldTheory.Finite.Basic
  - CharP.char_is_prime_or_zero (特征是 0 或素数)
  - CharZero (特征零), charP_zero_iff
  - Field → IsDomain (域是整环)
  - inv_mul_cancel, mul_inv_cancel
  - Fintype.card_eq_prime_pow (有限域的阶是素数幂)

【可计算性 Computability】
  - import Mathlib.Computability.Primrec
  - import Mathlib.Computability.DFA
  - Primrec.zero, Primrec.succ, Primrec.comp
  - DFA.AcceptsLanguage, DFA.complement
  - NFA.toDFA (NFA → DFA 转换)

【动力系统 Dynamics】
  - import Mathlib.Dynamics.FixedPoints.Basic
  - import Mathlib.Dynamics.PeriodicPts
  - Function.IsFixedPt, Function.fixedPoints
  - Function.IsPeriodicPt, isFixedPt_iff_isPeriodicPt_one
  - Function.IsFixedPt.isPeriodicPt

【模型论 ModelTheory】
  - import Mathlib.ModelTheory.Basic
  - import Mathlib.ModelTheory.Substructures
  - FirstOrder.Language.Structure
  - Substructure, Embedding, Equiv
  - Equiv.symm (同构对称), Equiv.trans (同构传递)

【信息论 InformationTheory】
  - import Mathlib.Combinatorics.SimpleGraph.Hamming
  - hammingDist, hammingDist_self, hammingDist_comm
  - hammingDist_triangle, hammingDist_nonneg
  - hammingDist_eq_zero (距离 0 则相等)
  注意: Mathlib 中汉明距离用 hammingDist 而非 hamming_dist

【凝聚数学 Condensed】
  - import Mathlib.Condensed.Basic
  - import Mathlib.Condensed.Discrete
  - CondensedSet, CondensedAb (凝聚阿贝尔群)
  - Condensed.discrete (离散化函子)
  - 凝聚范畴的性质需要 Sites/Sheaves 理论

【表示论 RepresentationTheory】
  - import Mathlib.RepresentationTheory.Basic
  - import Mathlib.RepresentationTheory.Maschke
  - Representation, FDRep
  - Rep.trivial (平凡表示)
  - Maschke's theorem: Action.completelyReducible
  - SchurLemma: 不可约表示间的非零同态是同构

输出格式（严格遵守）：
```lean
import Mathlib

theorem <name> : <statement> := by
  <proof>  -- 尽可能用一行 tactic 完成
```

关键注意事项：
- 始终 `import Mathlib`
- 需要时用 open 打开命名空间（如 open MeasureTheory / open CategoryTheory）
- 证明越短越好，一行能解决的绝不写两行
- 不要在 by 块中写 sorry
- 如果 simp/omega/ring 能直接解决，就不要用 intro/induction
- 自然数相关：优先用 omega（它能处理大多数自然数等式和不等式）
- 如果编译报 "No goals to be solved"，说明你的证明在某处已经完成了，删掉后续所有行

如果无法完成证明，明确指出卡在哪一步。"""


class ProverAgent(BaseAgent):
    """Lean 4 证明专家智能体。"""

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        lean_interface: Optional[LeanInterface] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        **kwargs,
    ):
        config = agent_config or AgentConfig(
            agent_name="Prover",
            system_prompt=PROVER_SYSTEM_PROMPT,
        )
        if not config.system_prompt:
            config.system_prompt = PROVER_SYSTEM_PROMPT

        super().__init__(config, llm_client, resource_manager)
        self.lean = lean_interface or LeanInterface()
        self.ltm = long_term_memory

    async def _execute(self, task: str, **kwargs) -> Any:
        """
        执行证明任务。

        Args:
            task: 自然语言数学命题或证明思路
            kwargs:
                - theorem_name: 定理名称
                - hints: 额外提示
                - theorem_toolkit: 已证明定理工具箱（可直接引用的定理列表）
                - max_attempts: 最大尝试次数
        """
        theorem_name = kwargs.get("theorem_name", "main_theorem")
        hints = kwargs.get("hints", "")
        theorem_toolkit = kwargs.get("theorem_toolkit", "")
        max_attempts = kwargs.get("max_attempts", 5)

        logger.info(f"[Prover] 开始证明: {task[:100]}...")

        # 1. 搜索相关的已有策略和定理
        context = ""
        if self.ltm:
            related = self.ltm.search(task, top_k=5)
            if related:
                context = "相关已知知识:\n"
                for r in related:
                    context += f"- [{r.get('type', '')}] {r.get('natural_language', '')[:150]}\n"
                    if r.get("lean_code"):
                        context += f"  Lean: {r.get('lean_code', '')[:200]}\n"

        # 2. 注入已证明定理工具箱
        toolkit_section = ""
        if theorem_toolkit:
            toolkit_section = f"\n{theorem_toolkit}\n"

        # 3. 生成初始 Lean 代码
        prompt = f"""请将以下数学命题形式化为 Lean 4 代码并完成证明。

命题: {task}
定理名称: {theorem_name}

{f'提示: {hints}' if hints else ''}
{context}
{toolkit_section}

⚠️ 策略要求（必须遵守）：
1. 必须 `import Mathlib`
2. 需要时用 open 打开命名空间（如 open MeasureTheory / open CategoryTheory）
3. 先尝试一行自动化 tactic（按顺序：simp, omega, ring, norm_num, decide, exact?, aesop）
4. 如果一行 tactic 不够，尝试直接引用 Mathlib 定理（如 exact Nat.add_comm ..）
5. 仅在以上全部失败时才使用 induction，且归纳分支内也优先用 simp/omega
6. 绝对不要在证明目标已解决后添加多余的 tactic 行
7. 证明越短越好

请给出完整的 Lean 4 代码。代码用 ```lean 和 ``` 包围。"""

        response = await self.think(prompt)
        lean_code = self._extract_lean_code(response)

        if not lean_code:
            return {
                "success": False,
                "error": "未能生成 Lean 代码",
                "response": response,
            }

        # 3. 编译和迭代修正
        for attempt in range(1, max_attempts + 1):
            if self.should_stop():
                break

            result = await self.lean.compile(lean_code)

            if result.success:
                logger.info(f"[Prover] 证明成功！第{attempt}次尝试")

                # 注意：定理命名和存储由 TuringAgent._name_and_store_theorem 负责

                return {
                    "success": True,
                    "lean_code": lean_code,
                    "attempts": attempt,
                    "natural_language": task,
                }

            # 编译失败，请求修正
            error_info = result.error_summary
            logger.info(f"[Prover] 第{attempt}次失败: {error_info[:200]}")

            if attempt < max_attempts:
                # 针对常见错误模式给出特定修复指导
                error_guidance = ""
                if "No goals to be solved" in error_info:
                    error_guidance = """\n⚠️ "No goals to be solved" 表示证明在某一行已经完成，但你在后面写了多余的 tactic。
修复方法：找到证明实际完成的那一行，删除它之后的所有 tactic 行。
通常 simp/omega/ring 等已经关闭了所有目标，后面不需要再写任何东西。"""
                elif "unknown identifier" in error_info:
                    error_guidance = """\n⚠️ "unknown identifier" 通常是因为没有 `import Mathlib` 或者使用了错误的定理名。
确保代码开头有 `import Mathlib`，且使用正确的 Mathlib 定理名（如 Nat.add_comm 而非 add_comm）。"""
                elif "Type mismatch" in error_info:
                    error_guidance = """\n⚠️ "Type mismatch" 通常是参数类型或数量不对。
考虑直接用 omega/simp/ring 替代手动 apply/exact，让 Lean 自动推导类型。"""

                fix_prompt = f"""Lean 4 编译失败。请修正代码。

当前代码:
```lean
{lean_code}
```

编译错误:
{error_info}
{error_guidance}

修正要求：
1. 优先尝试用 simp/omega/ring/norm_num 一行替换出错的部分
2. 不要在证明已完成后添加多余的行
3. 确保 `import Mathlib` 在代码开头
4. 给出修正后的完整代码，用 ```lean 和 ``` 包围。"""

                fix_response = await self.think(fix_prompt)
                new_code = self._extract_lean_code(fix_response)
                if new_code:
                    lean_code = new_code
                else:
                    break

        # 所有尝试失败
        if self.ltm:
            self.ltm.add_error_log(
                problem=task,
                error_description=f"经过{max_attempts}次尝试仍无法证明: {error_info[:300]}",
                lean_code=lean_code,
            )

        return {
            "success": False,
            "lean_code": lean_code,
            "attempts": max_attempts,
            "last_error": error_info,
        }

    @staticmethod
    def _extract_lean_code(text: str) -> str:
        """从 LLM 响应中提取 ``\`\`\`lean ... \`\`\``` 代码块。"""
        # 匹配 ```lean ... ``` 代码块
        patterns = [
            r"```lean4?\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # 如果没有代码块标记，尝试找 theorem/def/lemma 开头的内容
        lines = text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if any(line.strip().startswith(kw) for kw in [
                "import", "theorem", "def", "lemma", "example",
                "open", "namespace", "section", "#check",
            ]):
                in_code = True
            if in_code:
                code_lines.append(line)

        return "\n".join(code_lines).strip() if code_lines else ""
