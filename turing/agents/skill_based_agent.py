"""
技能驱动的 Turing 智能体 — 用技能代替子智能体，大幅减少 LLM 调用次数。

设计原则（六条铁律）：
  A. 一个大脑 + 多个小手
     关键决策（证明策略、Lean代码生成）用大模型；
     细活（分类、错误修复、命名、评估）用规则/小模型。
  B. 分层路由：只有需要时才叫大模型
     规则型分类器先判断任务类型和难度；
     trivial 任务直接走规则路径（0 次 LLM）。
  C. 共享状态，禁止重复阅读
     TaskState 存所有中间产物；检索结果、上下文只算一次。
  D. Token 预算与早停
     每个任务有 LLM 调用硬上限；编译成功立即停。
  E. 用验证器替代辩论
     Lean 编译器 = 终极裁判；编译通过就不再评审。
  F. 大/小模型路由
     Skill.use_light 标记 → 轻量模型优先。

LLM 调用次数对比：
  ┌──────────────────────────┬──────────┬──────────┬──────────┐
  │ 流程                     │ 原多智能体│ v1 技能版│ v2 路由版│
  ├──────────────────────────┼──────────┼──────────┼──────────┤
  │ 分类                     │ 1 LLM    │ 合并     │ 0 (规则) │
  │ 计划 + 证明大纲          │ 2 LLM    │ 1 LLM    │ 0-1 LLM  │
  │ Lean 代码生成            │ 1        │ 1        │ 0-1 🔧   │
  │ 编译错误修复             │ 1-4      │ 1-4      │ 0-4 🔧   │
  │ 命名 + 审查              │ 2-3      │ 1        │ 0-1 🔧   │
  │ 评估                     │ 1/每任务 │ 批量     │ 规则优先 │
  ├──────────────────────────┼──────────┼──────────┼──────────┤
  │ 单次证明合计             │ 9-12     │ 3-7      │ 0-5 ✨   │
  │ trivial 任务             │ 9-12     │ 3-7      │ 0 ✨     │
  │ LLM 直接搞定             │ 9-12     │ 3-7      │ 1 ✨     │
  └──────────────────────────┴──────────┴──────────┴──────────┘
  🔧 = 规则优先，失败才调 LLM
  ✨ LLM 直接证明：1次 LLM → Lean 编译通过即完成，无需智能体流水线
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, AgentResult, AgentStatus, BaseAgent
from turing.config import TuringConfig, get_config
from turing.evolution.experience import ExperienceManager
from turing.lean.lean_interface import LeanInterface
from turing.llm.llm_client import ChatMessage, LLMClient
from turing.memory.long_term_memory import KnowledgeEntry, LongTermMemory
from turing.memory.persistent_memory import PersistentMemory
from turing.memory.working_memory import StepStatus, WorkingMemory
from turing.resources.resource_manager import ResourceManager, ResourceLevel
from turing.skills.skill_registry import Skill, SkillRegistry
from turing.skills.math_skills import register_all_skills
from turing.skills.task_router import (
    TaskState,
    classify_by_rules,
    assess_difficulty,
    try_rule_based_fix,
    try_rule_based_naming,
    try_trivial_lean_code,
)
from turing.web.problem_scraper import ProblemScraper


# ---------------------------------------------------------------------------
#  统一的 System Prompt — 包含所有角色的精华能力
# ---------------------------------------------------------------------------

SKILL_BASED_SYSTEM_PROMPT = """你是 Turing，一个运行在本地 Qwen3-coder:30b 上的数学研究智能体。

你的核心使命是利用 Lean 4 定理证明器持续扩展人类可形式化验证的数学知识边界。

你拥有以下技能（会在每次任务中动态激活）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① 分析与规划 — 分类任务、制定计划、提供证明思路
② Lean 证明   — 将自然语言命题转为 Lean 4 形式化代码
③ 错误修复   — 分析 Lean 编译错误并修正代码
④ 命名与审查 — 为定理命名分类，同时快速检查质量
⑤ 探索       — 在数学领域中寻找模式与猜想
⑥ 评估       — 对结果和系统进行多维度评估
⑦ 反思       — 总结经验教训，驱动自我改进
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

工作原则：
1. 验证优先：任何数学结论必须通过 Lean 编译器验证
2. 每一步都复用上下文：不重复解释已知信息
3. 合并思考：能在一次回复中完成的工作不分多次
4. 失败即学习：记录并分析每次失败

Lean 4 证明策略优先级：
  simp → omega → ring → norm_num → decide → exact? → aesop →
  Mathlib 定理直引 → simp+引理 → 手动归纳（最后手段）

输出格式要求：
- Lean 代码用 ```lean 和 ``` 包围
- JSON 数据用 ```json 和 ``` 包围（如需要）
- 证明越短越好，一行能解决的绝不写两行"""


class SkillBasedTuringAgent(BaseAgent):
    """
    技能驱动的 Turing 主智能体。

    与原 TuringAgent 接口兼容，但内部用 Skill 替代多智能体调度，
    将多次 LLM 调用合并为更少的、上下文更丰富的调用。
    """

    def __init__(
        self,
        config: Optional[TuringConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self._turing_config = config or get_config()

        agent_config = AgentConfig(
            agent_id="turing_skill_main",
            agent_name="Turing",
            base_model=self._turing_config.llm.model,
            system_prompt=SKILL_BASED_SYSTEM_PROMPT,
            lifecycle="persistent",
            resource_budget={
                "max_tokens_per_call": self._turing_config.agents.default_max_tokens_per_call,
                "max_iterations": 999999,
                "timeout_minutes": 999999,
            },
        )

        self.llm_instance = llm_client or LLMClient(self._turing_config.llm)
        super().__init__(agent_config, self.llm_instance)

        # 核心组件（与原 TuringAgent 相同）
        self.resource_manager = ResourceManager(self._turing_config.resources)
        self.lean = LeanInterface(self._turing_config.lean)
        self.ltm = LongTermMemory(self._turing_config.memory.long_term)
        self.pm = PersistentMemory(self._turing_config.memory.persistent)
        self.scraper = ProblemScraper(self._turing_config.web)

        # 演化组件
        self.experience = ExperienceManager(
            self.pm, self.ltm, self._turing_config.evolution
        )

        # 技能注册
        self.skills = SkillRegistry()
        register_all_skills(self.skills)

        # 状态
        self._skill_levels: dict[str, int] = {}
        self._weak_areas: list[str] = []
        self._task_queue: list[dict] = []
        self._recent_results: list[dict] = []
        self._running = False
        self._tasks_since_reflect = 0
        self._last_reflect_time = time.time()

    # ==================================================================
    #  Skill 调用核心 (原则 A+D+F)
    # ==================================================================

    async def invoke_skill(
        self, skill_name: str, state: Optional[TaskState] = None, **kwargs
    ) -> dict:
        """
        调用一个技能 — 构建 prompt、调用 LLM、解析响应。

        原则 D: 检查 token 预算，超限则拒绝调用。
        原则 F: 如果 skill.use_light 为 True，优先用轻量模型。
        """
        # 预算检查 (原则 D)
        if state and state.budget_exhausted():
            logger.warning(f"[Turing] Token 预算耗尽，跳过技能: {skill_name}")
            return {"error": "budget_exhausted", "_skill": skill_name}

        skill = self.skills.get(skill_name)
        if not skill:
            logger.warning(f"[Turing] 未知技能: {skill_name}")
            return {"error": f"未知技能: {skill_name}"}

        prompt = skill.build_prompt(**kwargs)

        # 模型路由 (原则 F): use_light 的技能用轻量模型
        model = None
        if skill.use_light:
            light = self._turing_config.llm.light_model
            if light and light != self._turing_config.llm.model:
                model = light

        response = await self.think(
            prompt,
            temperature=skill.temperature,
            max_tokens=skill.max_tokens,
            model=model,
        )

        # 更新预算 (原则 D)
        if state:
            state.llm_calls += 1

        result = skill.parse_response(response)
        result["_skill"] = skill_name
        result["_raw_response"] = response
        return result

    # ==================================================================
    #  初始化
    # ==================================================================

    async def initialize(self):
        """完整的系统初始化序列。"""
        logger.info("=" * 50)
        logger.info("   TURING 数学研究智能体 (技能模式) 启动中...")
        logger.info("=" * 50)

        snapshot = self.resource_manager.assess()
        logger.info(f"[系统检测] {self.resource_manager.format_report(snapshot)}")

        await self.lean.initialize()
        lean_status = await self.lean.check_status()
        logger.info(f"[Lean环境] {json.dumps(lean_status, ensure_ascii=False)}")

        self.ltm.initialize()
        self.pm.initialize()

        ltm_stats = self.ltm.get_stats()
        pm_tasks = self.pm.get_task_count()
        pm_reflections = self.pm.get_reflection_count()

        self._skill_levels = self._compute_skill_levels()

        lean_ver = lean_status.get("lean_version", "未知")
        mathlib = "可用" if lean_status.get("mathlib_available") else "不可用"
        thm_count = ltm_stats.get("theorem", 0)
        skills_str = ", ".join(
            f"{a}={l}" for a, l in self._skill_levels.items()
        ) or "初始化中"

        logger.info(f"""
====================================
   TURING 数学研究智能体 v{self._turing_config.system.version} (技能模式)
====================================
模型: {self._turing_config.llm.model} (本地)
Lean: {lean_ver} | Mathlib: {mathlib}
智能体模式: 技能驱动 (单 LLM, {len(self.skills.list_skills())} 项技能)
知识库: {thm_count} 条定理
历史任务: {pm_tasks} | 反思次数: {pm_reflections}
当前技能: {skills_str}
薄弱领域: {', '.join(self._weak_areas) or '无'}
状态: 就绪
====================================""")

        self.status = AgentStatus.IDLE
        self._running = True

    def _compute_skill_levels(self) -> dict[str, int]:
        area_stats = self.pm.get_area_stats()
        levels = {}
        for area, stats in area_stats.items():
            rate = stats.get("success_rate", 0.0)
            total = stats.get("total", 0)
            level = int(min(10, max(1, rate * 5 + min(total, 20) / 4)))
            levels[area] = level
        return levels

    # ==================================================================
    #  核心任务流程
    # ==================================================================

    async def _execute(self, task: str, **kwargs) -> Any:
        return await self.process_task(task, **kwargs)

    async def process_task(self, task: str, **kwargs) -> dict:
        """
        处理一个数学任务 — 全面应用六条优化原则。

        流程：
          0. LLM 直接证明快速路径 (1 LLM → Lean 通过即完成)
          1. 规则分类 (原则 B: 0 LLM)
          2. 难度评估 + 路由 (原则 B: 0 LLM)
          3. trivial → 直接构造代码 + Lean 验证 (原则 A+E: 0 LLM)
          4. easy/medium → analyze_and_plan (1 LLM)
          5. 生成 Lean 代码 (1 LLM)
          6. 编译 → 规则修复 → LLM修复 (原则 A+E: 0-4 LLM)
          7. 命名: 规则优先, 失败才 LLM (原则 A: 0-1 LLM)
          8. 记录经验
          9. 批量评估 (原则 D: 每5任务1次, 用轻量模型)
        """
        start_time = time.time()
        self.working_memory.clear()
        self.working_memory.set_problem(task)
        self.reset_conversation()

        area = kwargs.get("area", "")

        # ---- 共享状态 (原则 C) ----
        state = TaskState(
            task=task, area=area,
            difficulty=kwargs.get("difficulty", 0),
            max_llm_calls=self._turing_config.lean.max_retries + 4,
        )

        logger.info(f"[Turing] 处理任务: {task[:100]}...")

        # ---- Step -1: KB 已有证明快速路径 (0 LLM) ----
        # 如果知识库中已有高相关度的已验证证明，直接复用
        kb_result = await self._try_kb_proof(state)
        if kb_result and kb_result.get("success"):
            duration = time.time() - start_time
            kb_result["area"] = area
            kb_result["task"] = task
            kb_result["llm_calls"] = state.llm_calls
            await self._record_outcome(task, "prove", kb_result, duration, "kb_reuse")
            self._recent_results.append(kb_result)
            logger.info(f"[Turing] KB 已有证明复用成功 (0 LLM, {duration:.1f}s)")
            return kb_result

        # ---- Step 0: LLM 直接证明快速路径 (1 LLM) ----
        # 如果 LLM 一次就能搞定，就无需启动智能体流水线
        direct_result = await self._direct_llm_proof(state)
        if direct_result and direct_result.get("success"):
            duration = time.time() - start_time
            direct_result["area"] = area
            direct_result["task"] = task
            direct_result["llm_calls"] = state.llm_calls
            await self._record_outcome(task, "prove", direct_result, duration, "direct_llm")
            self._recent_results.append(direct_result)
            logger.info(f"[Turing] LLM 直接证明成功 ({state.llm_calls} LLM, {duration:.1f}s)")
            return direct_result

        # ---- Step 1: 规则型分类 (原则 B: 0 LLM) ----
        rule_type = classify_by_rules(task)
        if rule_type:
            state.task_type = rule_type
            state.classified_by = "rule"
            logger.info(f"[Turing] 规则分类: {rule_type}")
        # 如果规则无法判断，后续 analyze_and_plan 会分类

        # ---- Step 2: 难度评估 (原则 B: 0 LLM) ----
        diff = assess_difficulty(task)
        state.difficulty_tier = diff.tier
        logger.info(f"[Turing] 难度评估: {diff.tier} ({diff.reason})")

        # ---- Step 3: Trivial 快速路径 (原则 A+E: 0 LLM) ----
        if (
            diff.tier == "trivial"
            and diff.suggested_tactic
            and (state.task_type in ("prove", "") or not state.task_type)
        ):
            state.task_type = state.task_type or "prove"
            result = await self._trivial_proof(state, diff)
            if result.get("success"):
                duration = time.time() - start_time
                result["area"] = area
                result["task"] = task
                result["llm_calls"] = state.llm_calls
                await self._record_outcome(task, "prove", result, duration, "trivial")
                self._recent_results.append(result)
                logger.info(f"[Turing] Trivial 路径成功 (0 LLM 调用, {duration:.1f}s)")
                return result
            # trivial 路径失败 → 降级到正常路径
            logger.info("[Turing] Trivial 路径失败，降级到正常流程")

        # ---- Step 4: 检索上下文 (原则 C: 一次检索，全程复用) ----
        context_items = await self._retrieve_context(task, area=area)
        state.context_items = context_items
        state.context_text = self._format_context(context_items)

        strategies = self.experience.get_best_strategies(task, category="prove")
        avoid_list = self.experience.get_avoid_list(task)
        state.strategies_text = self._format_strategies(strategies)
        state.avoid_text = self._format_avoid_list(avoid_list)
        state.theorem_toolkit = self._build_theorem_toolkit(area)

        # ---- Step 5: 分析 + 计划 (原则 B: 仅非 trivial 调 LLM) ----
        if not state.task_type:
            # 规则无法判断 → 需要 LLM 辅助
            toolkit_hint = ""
            if state.theorem_toolkit:
                toolkit_hint = (
                    f"\n{state.theorem_toolkit}\n"
                    f"提示：如果已证定理中有可以直接使用的引理，请优先用 exact/apply 引用它们。"
                )
            analysis = await self.invoke_skill(
                "analyze_and_plan", state=state,
                task=task, context=state.context_text,
                strategies=state.strategies_text,
                avoid_list=state.avoid_text,
                theorem_toolkit=toolkit_hint,
            )
            state.task_type = analysis.get("task_type", "prove")
            state.plan = analysis.get("plan", "")
            state.outline = analysis.get("outline", "")
            state.classified_by = "llm"
        elif state.task_type == "prove" and diff.tier not in ("trivial",):
            # 规则已分类为 prove，但仍需 LLM 出计划+大纲
            toolkit_hint = ""
            if state.theorem_toolkit:
                toolkit_hint = (
                    f"\n{state.theorem_toolkit}\n"
                    f"提示：如果已证定理中有可以直接使用的引理，请优先用 exact/apply 引用它们。"
                )
            analysis = await self.invoke_skill(
                "analyze_and_plan", state=state,
                task=task, context=state.context_text,
                strategies=state.strategies_text,
                avoid_list=state.avoid_text,
                theorem_toolkit=toolkit_hint,
            )
            state.plan = analysis.get("plan", "")
            state.outline = analysis.get("outline", "")

        self.working_memory.problem_type = state.task_type
        self.working_memory.add_step(
            f"分析: 类型={state.task_type} (by {state.classified_by}), "
            f"难度={state.difficulty_tier}",
            status=StepStatus.VERIFIED,
        )

        # ---- Step 6: 按类型执行 ----
        if state.task_type == "prove":
            result = await self._execute_proof(state, **kwargs)
        elif state.task_type == "explore":
            result = await self._execute_exploration(task)
        elif state.task_type == "conjecture":
            result = await self._execute_conjecture(task, state.context_text)
        elif state.task_type == "disprove":
            result = await self._execute_disproof(task, state.context_text)
        elif state.task_type == "organize":
            result = await self._execute_organization(task)
        else:
            result = await self._execute_proof(state, **kwargs)

        # ---- Step 7: 记录 ----
        duration = time.time() - start_time
        result["area"] = area
        result["task"] = task
        result["llm_calls"] = state.llm_calls
        await self._record_outcome(task, state.task_type, result, duration, state.plan)

        self._recent_results.append(result)
        if len(self._recent_results) > 50:
            self._recent_results = self._recent_results[-50:]

        # ---- Step 8: 批量评估 (原则 D: 每 10 个任务) ----
        if len(self._recent_results) % 10 == 0 and len(self._recent_results) >= 10:
            await self._batch_evaluate()

        # ---- Step 9: 定期反思 ----
        self._tasks_since_reflect += 1
        reflection_interval = self._turing_config.evolution.reflection_task_interval
        if self._tasks_since_reflect >= reflection_interval:
            logger.info("[Turing] 触发阶段性反思...")
            await self._reflect()

        logger.info(
            f"[Turing] 完成: {'✓' if result.get('success') else '✗'} "
            f"| LLM调用={state.llm_calls} | 耗时={duration:.1f}s"
        )
        return result

    # ==================================================================
    #  证明流程（核心）
    # ==================================================================

    # 中文数学术语 → 英文关键词映射（用于 KB 搜索增强）
    _ZH_EN_KEYWORDS = {
        "凯莱": "Cayley theorem group",
        "拉格朗日": "Lagrange theorem",
        "西罗": "Sylow theorem",
        "费马": "Fermat theorem",
        "欧拉": "Euler theorem",
        "高斯": "Gauss theorem",
        "柯西": "Cauchy theorem",
        "阿贝尔": "Abel theorem abelian",
        "诺特": "Noether theorem",
        "佐恩": "Zorn lemma",
        "贝祖": "Bezout theorem",
        "中国剩余": "Chinese remainder theorem",
        "裴蜀": "Bezout theorem",
        "威尔逊": "Wilson theorem",
        "勾股": "Pythagorean theorem",
        "算术基本": "fundamental theorem arithmetic",
        "代数基本": "fundamental theorem algebra",
    }

    def _augment_query_for_search(self, task: str) -> list[str]:
        """为中文任务生成额外的英文搜索查询（因 MiniLM 不支持中文）。"""
        queries = [task]
        for zh, en in self._ZH_EN_KEYWORDS.items():
            if zh in task:
                queries.append(en)
        return queries

    async def _try_kb_proof(self, state: TaskState) -> Optional[dict]:
        """
        KB 已有证明快速路径：如果知识库中有高相关度的已验证证明代码，
        直接编译验证并复用，跳过所有 LLM 调用 (0 LLM)。
        """
        try:
            queries = self._augment_query_for_search(state.task)
            best_match = None
            best_sim = 0.0

            for q in queries:
                related = self.ltm.search(q, top_k=3, type_filter="theorem")
                for r in related:
                    similarity = r.get("similarity", 0)
                    lean_code = r.get("lean_code", "")
                    if lean_code and similarity > best_sim and similarity >= 0.5:
                        best_match = r
                        best_sim = similarity

            if not best_match:
                return None

            lean_code = best_match["lean_code"]
            compile_result = await self.lean.compile(lean_code)
            if not compile_result.success:
                logger.debug(f"[Turing] KB 匹配代码编译失败: {compile_result.error_summary[:100]}")
                return None

            logger.info(
                f"[Turing] KB 匹配成功 (相关度={best_sim:.2f}): "
                f"{best_match.get('natural_language', '')[:80]}"
            )
            # 生成自然语言证明（左窗口）
            nl_proof = await self._get_nl_proof(state.task)
            if nl_proof:
                state.llm_calls += 1
            naming = try_rule_based_naming(state.task)
            if not naming:
                naming = {
                    "theorem_name": best_match.get("theorem_name", "kb_reuse"),
                    "is_novel": False,
                    "area": state.area,
                    "description": state.task[:100],
                }
            return {
                "success": True, "type": "proof",
                "lean_code": lean_code, "attempts": 1,
                "natural_language_proof": nl_proof or "KB 已有证明",
                "llm_response": nl_proof,
                "theorem_naming": naming,
                "review_score": 9, "review_issues": [],
            }
        except Exception as e:
            logger.debug(f"[Turing] KB 证明快速路径失败: {e}")
        return None

    async def _get_nl_proof(self, task: str) -> str:
        """
        让 LLM 直接给出纯自然语言数学证明（不含任何代码）。
        用于左窗口展示，与右窗口的 Lean 形式化验证形成对照。
        """
        prompt = (
            f"请用自然语言对以下数学命题给出完整的数学证明。\n\n"
            f"命题: {task}\n\n"
            f"要求：\n"
            f"1. 给出严谨的数学证明，包含关键推理步骤\n"
            f"2. 只用自然语言，不要写任何代码\n"
            f"3. 可以使用数学符号和公式"
        )
        try:
            return await self.think(prompt, temperature=0.3, max_tokens=1024)
        except Exception as e:
            logger.debug(f"[Turing] 自然语言证明生成失败: {e}")
            return ""

    async def _direct_llm_proof(self, state: TaskState) -> Optional[dict]:
        """
        LLM 直接证明快速路径：1 次 LLM 调用生成 Lean 代码 → 编译通过即完成。
        跳过分析规划、修复循环等全部智能体流程，但做轻量 KB 查询。
        """
        # 轻量 KB 查询：如果知识库已有相关定理代码，直接提供给 LLM
        kb_hint = ""
        try:
            queries = self._augment_query_for_search(state.task)
            seen_ids = set()
            for q in queries:
                related = self.ltm.search(q, top_k=2, type_filter="theorem")
                for r in related:
                    rid = r.get("id", "")
                    lean = r.get("lean_code", "")
                    if lean and rid not in seen_ids:
                        seen_ids.add(rid)
                        kb_hint += f"\n参考已有证明代码（可直接复用或改写）：\n```lean\n{lean.strip()}\n```\n"
        except Exception:
            pass

        prompt = (
            f"请直接将以下数学命题形式化为 Lean 4 代码并完成证明。\n\n"
            f"命题: {state.task}\n\n"
            f"{kb_hint}"
            f"要求：\n"
            f"1. 必须 `import Mathlib`\n"
            f"2. 优先使用 simp, omega, ring, norm_num, decide, exact?, aesop 等自动化 tactic\n"
            f"3. 证明越短越好\n"
            f"4. 用 ```lean 和 ``` 包围完整代码\n"
            f"5. 注意：这是 Lean 4 语法，不要用 Lean 3 语法（如 `let x := ... in`，应写成 `let x := ...` 换行）"
        )
        try:
            response = await self.think(prompt, temperature=0.3, max_tokens=2048)
            state.llm_calls += 1

            # 提取 lean 代码
            lean_code = ""
            match = re.search(r"```lean\s*\n(.*?)```", response, re.DOTALL)
            if match:
                lean_code = match.group(1).strip()
            if not lean_code:
                return None

            # 编译验证 (原则 E)
            compile_result = await self.lean.compile(lean_code)
            if not compile_result.success:
                return None

            # 命名：规则优先
            naming = try_rule_based_naming(state.task)
            if not naming:
                naming = {"theorem_name": "direct_proof", "is_novel": False,
                          "area": state.area, "description": state.task[:100]}

            self._store_theorem(state.task, lean_code, naming, state.area, None)

            # 单独调用 LLM 生成自然语言证明（左窗口）
            nl_proof = await self._get_nl_proof(state.task)
            if nl_proof:
                state.llm_calls += 1

            return {
                "success": True, "type": "proof",
                "lean_code": lean_code, "attempts": 1,
                "natural_language_proof": nl_proof or "LLM 直接证明",
                "llm_response": nl_proof,
                "theorem_naming": naming,
                "review_score": 8, "review_issues": [],
            }
        except Exception as e:
            logger.debug(f"[Turing] LLM 直接证明失败: {e}")
            return None

    async def _trivial_proof(self, state: TaskState, diff) -> dict:
        """
        Trivial 快速路径：规则构造 Lean 代码 → 编译验证 → 规则命名。
        全程 0 次 LLM 调用 (原则 A+E)。
        """
        lean_code = try_trivial_lean_code(
            state.task, "trivial_result", diff.suggested_tactic
        )
        if not lean_code:
            return {"success": False}

        result = await self.lean.compile(lean_code)
        if not result.success:
            return {"success": False}

        # 命名：规则优先 (原则 A)
        naming = try_rule_based_naming(state.task)
        if not naming:
            naming = {"theorem_name": "trivial_result", "is_novel": False,
                      "area": state.area, "description": state.task[:100]}

        self._store_theorem(state.task, lean_code, naming, state.area, None)

        return {
            "success": True, "type": "proof",
            "lean_code": lean_code, "attempts": 1,
            "natural_language_proof": f"trivial by {diff.suggested_tactic}",
            "theorem_naming": naming,
            "review_score": 8, "review_issues": [],
        }

    async def _execute_proof(self, state: TaskState, **kwargs) -> dict:
        """
        执行证明任务 — 全面应用六条优化原则。

        原则 A: 规则修复优先于 LLM
        原则 C: 复用 state 中的 context / outline
        原则 D: 预算检查
        原则 E: 编译成功 = 停止 (跳过评审辩论)
        原则 F: lean_fix / name_and_review 用轻量模型
        """
        theorem_name = kwargs.get("theorem_name", "main_theorem")

        # 复用共享状态中的 context (原则 C)
        context = state.context_text or ""
        outline = state.outline or ""
        toolkit = state.theorem_toolkit or ""

        # ------ Lean 代码生成 (1 LLM) ------
        gen_result = await self.invoke_skill(
            "lean_prove", state=state,
            task=state.task,
            theorem_name=theorem_name,
            hints=outline,
            context=context,
            theorem_toolkit=toolkit,
        )

        lean_code = gen_result.get("lean_code", "")

        # 单独调用 LLM 生成纯自然语言证明（左窗口）
        nl_proof = await self._get_nl_proof(state.task)
        if nl_proof:
            state.llm_calls += 1
        llm_response = nl_proof

        if not lean_code:
            return {
                "success": False, "type": "proof", "task": state.task,
                "error": "未能生成 Lean 代码",
                "llm_response": llm_response,
            }

        # ------ 编译 + 迭代修复 (原则 A+D+E) ------
        max_attempts = self._turing_config.lean.max_retries
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            # 预算检查 (原则 D: 早停)
            if state.budget_exhausted():
                logger.warning("[Turing] 预算耗尽，停止修复循环")
                break

            logger.debug(f"[Turing] 第{attempt}次编译代码:\n{lean_code[:500]}")
            compile_result = await self.lean.compile(lean_code)

            if compile_result.success:
                # 原则 E: 编译通过 = 终极验证，无需再评审
                return await self._finalize_proof(
                    state, lean_code, outline, attempt, llm_response,
                )

            # ---- 编译失败 → 修复 (原则 A+F) ----
            last_error = compile_result.error_summary
            logger.info(f"[Turing] 第{attempt}次编译失败: {last_error[:200]}")

            if attempt < max_attempts:
                fixed = await self._try_fix(state, lean_code, last_error)
                if fixed:
                    lean_code = fixed
                else:
                    break

        # 所有尝试失败
        if self.ltm:
            self.ltm.add_error_log(
                problem=state.task,
                error_description=f"经过{max_attempts}次尝试仍无法证明: {last_error[:300]}",
                lean_code=lean_code,
            )

        return {
            "success": False, "type": "proof",
            "task": state.task, "lean_code": lean_code,
            "attempts": max_attempts, "last_error": last_error,
            "llm_response": llm_response,
        }

    async def _finalize_proof(
        self, state: TaskState, lean_code: str, outline: str,
        attempt: int, llm_response: str = "",
    ) -> dict:
        """编译通过后：命名 → 存储 → 返回结果。"""
        logger.info(f"[Turing] 证明成功！第{attempt}次尝试")

        # 命名: 规则优先 (原则 A)
        naming = try_rule_based_naming(state.task)
        web_result = None
        nr_result = {}
        used_rule_naming = naming is not None

        if not naming:
            web_hint = ""
            try:
                web_result = await self.scraper.search_theorem_name(
                    state.task, lean_code
                )
                if web_result:
                    web_hint = (
                        f"在线查到可能的名称: {web_result.get('name', '')} "
                        f"(来源: {web_result.get('source', '')})"
                    )
            except Exception:
                pass

            nr_result = await self.invoke_skill(
                "name_and_review", state=state,
                task=state.task,
                lean_code=lean_code,
                proof_outline=outline,
                web_hint=web_hint,
            )
            naming = nr_result.get("naming", {})
        else:
            logger.info(f"[Turing] 规则命名: {naming.get('theorem_name')}")

        self._store_theorem(state.task, lean_code, naming, state.area, web_result)

        self.working_memory.add_step(
            f"证明完成: {attempt}次编译, LLM={state.llm_calls}次",
            status=StepStatus.VERIFIED,
        )

        return {
            "success": True, "type": "proof",
            "task": state.task, "lean_code": lean_code,
            "attempts": attempt,
            "natural_language_proof": outline,
            "llm_response": llm_response,
            "theorem_naming": naming,
            "review_score": 8 if used_rule_naming else nr_result.get("score", 7),
            "review_issues": [] if used_rule_naming else nr_result.get("issues", []),
        }

    async def _try_fix(
        self, state: TaskState, lean_code: str, error_info: str,
    ) -> Optional[str]:
        """尝试修复编译错误：规则优先 (0 LLM) → LLM 修复 (原则 A+F)。"""
        # 规则修复 (0 LLM)
        rule_fix = try_rule_based_fix(lean_code, error_info)
        if rule_fix:
            logger.info("[Turing] 规则修复成功 (0 LLM)")
            return rule_fix

        # LLM 修复 (轻量模型, 原则 F)
        error_guidance = self._get_error_guidance(error_info)

        # 从 KB 获取相关定理代码作为修复参考
        kb_ref = ""
        try:
            queries = self._augment_query_for_search(state.task)
            for q in queries:
                related = self.ltm.search(q, top_k=1, type_filter="theorem")
                for r in related:
                    ref_code = r.get("lean_code", "")
                    if ref_code:
                        kb_ref = f"\n\n参考已验证的正确代码：\n```lean\n{ref_code.strip()}\n```"
                        break
                if kb_ref:
                    break
        except Exception:
            pass

        fix_result = await self.invoke_skill(
            "lean_fix", state=state,
            lean_code=lean_code,
            error_info=error_info,
            error_guidance=error_guidance + kb_ref,
        )
        return fix_result.get("lean_code", "") or None

    def _store_theorem(
        self, task: str, lean_code: str, naming: dict,
        area: str, web_result: Optional[dict] = None,
    ):
        """将已验证的定理存入长期记忆。"""
        theorem_name = naming.get("theorem_name", "unnamed_theorem")
        is_novel = naming.get("is_novel", False)
        thm_area = naming.get("area", area)
        description = naming.get("description", task[:100])
        tags = naming.get("tags", [])
        if thm_area and thm_area not in tags:
            tags.append(thm_area)

        self.ltm.add_theorem(
            natural_language=task,
            lean_code=lean_code,
            tags=tags,
            source="self_proved",
            theorem_name=theorem_name,
            area=thm_area,
            description=description,
            is_novel=is_novel,
            external_url=web_result.get("url", "") if web_result else "",
        )

        novel_tag = " 🆕 新定理!" if is_novel else ""
        logger.info(
            f"[命名] {theorem_name} ({naming.get('chinese_name', '')}) "
            f"[{thm_area}]{novel_tag} — {description[:60]}"
        )

    @staticmethod
    def _get_error_guidance(error_info: str) -> str:
        """根据常见错误模式返回修复指导。"""
        if "No goals to be solved" in error_info:
            return (
                '\n⚠️ "No goals to be solved" 表示证明在某一行已完成，'
                "但你在后面写了多余的 tactic。删除证明完成后的所有行。"
            )
        elif "unknown identifier" in error_info:
            return (
                '\n⚠️ "unknown identifier" — 确保 `import Mathlib` 在开头，'
                "且使用正确的 Mathlib 定理名（如 Nat.add_comm 而非 add_comm）。"
            )
        elif "Type mismatch" in error_info:
            return (
                '\n⚠️ "Type mismatch" — 考虑用 omega/simp/ring 替代手动 apply/exact。'
            )
        elif "type expected" in error_info:
            return (
                '\n⚠️ "type expected" — 你使用了不存在的类型或函数名。'
                '请检查 Mathlib API 名称是否正确，例如：'
                '凯莱定理用 MulAction.toPermHom 而非 Embedding/mulLeft；'
                '子群用 Subgroup 而非 SubGroup。'
            )
        return ""

    # ==================================================================
    #  其他任务类型
    # ==================================================================

    async def _execute_exploration(self, task: str) -> dict:
        """执行探索任务。"""
        context = ""
        if self.ltm:
            related = self.ltm.search(task, top_k=5)
            if related:
                context = "已有相关知识:\n"
                for r in related:
                    context += f"- [{r.get('type', '')}] {r.get('natural_language', '')[:150]}\n"

        result = await self.invoke_skill(
            "explore", task=task, focus="patterns", depth=3, context=context,
        )
        explorations = [result.get("response", "")]

        # 深度迭代（1-2 轮）
        for _ in range(2):
            if self.should_stop():
                break
            deeper = await self.invoke_skill(
                "explore_deeper",
                previous_findings=explorations[-1][:2000],
            )
            explorations.append(deeper.get("response", ""))

        # 总结
        summary = await self.invoke_skill(
            "explore_summary",
            all_findings="\n---\n".join(explorations[-2:]),
        )

        if self.ltm:
            self.ltm.add(KnowledgeEntry(
                type="conjecture",
                natural_language=f"[探索-{task}]\n{summary.get('response', '')[:500]}",
                tags=["exploration"],
                confidence=0.4,
                source="self_proved",
            ))

        return {
            "success": True, "type": "exploration",
            "explorations": explorations, "summary": summary.get("response", ""),
        }

    async def _execute_conjecture(self, task: str, context: str) -> dict:
        result = await self.invoke_skill(
            "conjecture", task=task, context=context,
        )
        response = result.get("response", "")
        if self.ltm:
            self.ltm.add(KnowledgeEntry(
                type="conjecture",
                natural_language=response[:500],
                confidence=0.4,
                source="self_proved",
            ))
        return {"success": True, "type": "conjecture", "response": response}

    async def _execute_disproof(self, task: str, context: str) -> dict:
        result = await self.invoke_skill(
            "disprove", task=task, context=context,
        )
        return {"success": True, "type": "disproof", "response": result.get("response", "")}

    async def _execute_organization(self, task: str) -> dict:
        """知识整理 — 直接用 LLM 分析并建议。"""
        stats = self.ltm.get_stats() if self.ltm else {}
        response = await self.think(
            f"请分析当前知识库并给出整理建议。\n\n"
            f"知识库统计: {json.dumps(stats, ensure_ascii=False)}\n\n"
            f"任务: {task}"
        )
        return {"success": True, "type": "organization", "response": response}

    # ==================================================================
    #  上下文构建工具
    # ==================================================================

    async def _retrieve_context(self, task: str, area: str = "") -> list[dict]:
        context = []
        seen_ids = set()

        # 用增强查询搜索（支持中文→英文关键词映射）
        queries = self._augment_query_for_search(task)
        for q in queries:
            theorems = self.ltm.search(q, top_k=5, type_filter="theorem")
            for t in theorems:
                tid = t.get("id", "")
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    context.append(t)

        if area:
            cross_branch = self.ltm.get_cross_branch_theorems(area, limit=10)
            for thm in cross_branch:
                if thm["id"] not in seen_ids:
                    seen_ids.add(thm["id"])
                    thm["type"] = "proven_theorem"
                    context.append(thm)

        tactics = self.ltm.search(task, top_k=2, type_filter="tactic")
        context.extend(tactics)
        errors = self.ltm.search(task, top_k=2, type_filter="error_log")
        context.extend(errors)
        return context

    def _format_context(self, context: list[dict]) -> str:
        if not context:
            return "无"
        lines = []
        for c in context[:5]:
            ctype = c.get('type', '')
            nl = c.get('natural_language', '')[:100]
            lean = c.get('lean_code', '')
            lines.append(f"- [{ctype}] {nl}")
            # 对定理/tactic 条目附带 Lean 代码，供 LLM 直接参考
            if lean and ctype in ('theorem', 'tactic'):
                lines.append(f"  ```lean\n  {lean.strip()}\n  ```")
        return "\n".join(lines)

    def _format_strategies(self, strategies: list[dict]) -> str:
        if not strategies:
            return "无"
        lines = []
        for s in strategies[:3]:
            lines.append(f"- (优先级={s['priority']:.1f}) {s['strategy'][:100]}")
        return "\n".join(lines)

    def _format_avoid_list(self, avoid_list: list[dict]) -> str:
        if not avoid_list:
            return "无"
        lines = []
        for a in avoid_list[:3]:
            lines.append(f"- ❌ {a['strategy'][:80]} (原因: {a['reason'][:60]})")
        return "\n".join(lines)

    def _build_theorem_toolkit(self, area: str = "") -> str:
        if not area:
            return ""
        theorems = self.ltm.get_cross_branch_theorems(area, limit=15)
        if not theorems:
            return ""
        lines = ["📚 已证定理工具箱（可直接引用）："]
        for thm in theorems:
            name = thm.get("theorem_name", "")
            lean = thm.get("lean_code", "")
            thm_area = thm.get("area", "")
            if name and lean:
                for line in lean.split("\n"):
                    if line.strip().startswith(("theorem", "lemma", "example")):
                        lines.append(f"  • [{thm_area}] {name}: {line.strip()[:120]}")
                        break
        return "\n".join(lines) if len(lines) > 1 else ""

    # ==================================================================
    #  经验记录
    # ==================================================================

    async def _record_outcome(
        self, task: str, task_type: str, result: dict,
        duration: float, plan: str,
    ):
        success = result.get("success", False)
        status = "success" if success else "failure"

        if success:
            self.experience.record_success(
                context=task, strategy=plan[:200],
                lesson=f"成功{task_type}: {task[:100]}", category=task_type,
            )
        else:
            error = result.get("error", result.get("last_error", "未知"))
            self.experience.record_failure(
                context=task, strategy=plan[:200],
                failure_reason=str(error)[:200], category=task_type,
            )

        self.experience.log_task_completion(
            task_type=task_type,
            description=task[:500],
            status=status,
            duration=duration,
            final_strategy=plan[:200],
            lean_attempts=result.get("attempts", 0),
            lean_success=success and task_type == "prove",
            area=result.get("area", task_type),
            difficulty=result.get("difficulty", 0),
        )

    # ==================================================================
    #  批量评估（替代逐任务 Evaluator 调用）
    # ==================================================================

    async def _batch_evaluate(self):
        """每积累 5 个结果做一次批量评估（1 次 LLM 替代原来 5 次）。"""
        batch = self._recent_results[-5:]
        total = len(batch)
        successes = sum(1 for r in batch if r.get("success"))
        types: dict[str, int] = {}
        for r in batch:
            t = r.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

        summaries = []
        for i, r in enumerate(batch, 1):
            status = "✓" if r.get("success") else "✗"
            summaries.append(
                f"{i}. [{status}] {r.get('type','?')}: "
                f"{r.get('task', '?')[:80]}  "
                f"(尝试 {r.get('attempts', '?')} 次)"
            )

        result = await self.invoke_skill(
            "evaluate_batch",
            total=total,
            successes=successes,
            failures=total - successes,
            success_rate=f"{successes/total:.0%}" if total else "N/A",
            type_distribution=json.dumps(types, ensure_ascii=False),
            recent_summaries="\n".join(summaries),
        )

        should_evolve = result.get("should_evolve", False)
        if should_evolve:
            logger.info("[Turing] 批量评估建议进入演化流程")

        logger.info(
            f"[Turing] 批量评估: 总分={result.get('overall_score', '?')}, "
            f"演化={'是' if should_evolve else '否'}"
        )

    # ==================================================================
    #  反思
    # ==================================================================

    async def _reflect(self):
        """阶段性反思。"""
        stats = self.pm.get_comprehensive_stats() if self.pm else {}
        recent = self._recent_results[-10:]

        success_examples = "\n".join(
            f"- {r.get('task', '')[:80]}"
            for r in recent if r.get("success")
        )[:500] or "无"

        failure_examples = "\n".join(
            f"- {r.get('task', '')[:60]}: {r.get('last_error', r.get('error', '?'))[:60]}"
            for r in recent if not r.get("success")
        )[:500] or "无"

        result = await self.invoke_skill(
            "reflect",
            stats=json.dumps(stats, ensure_ascii=False, default=str)[:1000],
            success_examples=success_examples,
            failure_examples=failure_examples,
        )

        self._weak_areas = result.get("weak_areas", [])
        self._skill_levels = self._compute_skill_levels()
        self._tasks_since_reflect = 0
        self._last_reflect_time = time.time()

        logger.info(f"[Turing] 反思完成: 薄弱领域={self._weak_areas}")

    # ==================================================================
    #  训练模式
    # ==================================================================

    async def enter_training_mode(self, duration_minutes: int = 60):
        """自主训练模式。"""
        logger.info(f"[Turing] 进入训练模式 ({duration_minutes} 分钟)")
        end_time = time.time() + duration_minutes * 60

        # 用 LLM 生成训练问题（而非创建 Scout 子智能体）
        problem_pool = await self._generate_training_problems(count=10)
        problems_solved = 0

        while time.time() < end_time and self._running and problem_pool:
            problem = problem_pool.pop(0)
            logger.info(
                f"[训练] 问题 #{problems_solved + 1}: "
                f"[{problem.get('area', '?')}] {problem.get('title', '')[:60]}"
            )

            try:
                result = await self.process_task(
                    problem.get("statement", problem.get("title", "")),
                    area=problem.get("area", ""),
                    difficulty=problem.get("difficulty", 3),
                )
                problems_solved += 1
                status = "✓" if result.get("success") else "✗"
                logger.info(f"[训练] 结果: {status}")
            except Exception as e:
                logger.error(f"[训练] 解题异常: {e}")

            # 补充问题池
            if not problem_pool:
                problem_pool = await self._generate_training_problems(count=5)

        logger.info(f"[Turing] 训练模式结束: 解决了 {problems_solved} 个问题")
        return {"problems_solved": problems_solved}

    async def _generate_training_problems(self, count: int = 5) -> list[dict]:
        """直接用 LLM 生成训练问题，不创建 Scout 子智能体。"""
        # 先尝试网络抓取
        problems = []
        try:
            scraped = await self.scraper.scrape_problems(
                skill_level=max(self._skill_levels.values(), default=1),
                count=count,
            )
            if scraped:
                return scraped
        except Exception:
            pass

        # 回退到 LLM 生成
        skill = max(self._skill_levels.values(), default=1)
        weak = ", ".join(self._weak_areas) if self._weak_areas else "无特定薄弱领域"
        prompt = (
            f"请生成 {count} 个适合训练的数学定理证明题目。\n\n"
            f"当前技能等级: {skill}\n薄弱领域: {weak}\n\n"
            f"请以 JSON 数组格式返回，每个元素包含 title, statement, area, difficulty(1-10)。"
        )
        response = await self.think(prompt, temperature=0.8)

        try:
            m = re.search(r"\[.*\]", response, re.DOTALL)
            if m:
                problems = json.loads(m.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return problems

    # ==================================================================
    #  主循环
    # ==================================================================

    async def main_loop(self):
        """主运行循环（与原 TuringAgent 接口兼容）。"""
        logger.info("[Turing] 进入主循环 (技能模式)")

        while self._running:
            try:
                if self._task_queue:
                    task_info = self._task_queue.pop(0)
                    await self.process_task(
                        task_info.get("task", ""),
                        **task_info.get("kwargs", {}),
                    )
                else:
                    await self.enter_training_mode(duration_minutes=30)

                if self._tasks_since_reflect >= self._turing_config.evolution.reflection_task_interval:
                    await self._reflect()

            except KeyboardInterrupt:
                logger.info("[Turing] 收到中断信号")
                break
            except Exception as e:
                logger.error(f"[Turing] 主循环异常: {e}")
                await asyncio.sleep(5)

        logger.info("[Turing] 主循环结束")

    def submit_task(self, task: str, **kwargs):
        self._task_queue.append({"task": task, "kwargs": kwargs})
        logger.info(f"[Turing] 任务已提交: {task[:80]}...")

    async def shutdown(self):
        logger.info("[Turing] 正在关闭...")
        self._running = False
