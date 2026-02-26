"""
Turing 主智能体 — 系统的核心大脑和中央调度器。

Turing 是整个系统的主导智能体，负责：
- 接收和分析任务
- 调度子智能体协同工作
- 与 Lean 4 交互验证证明
- 管理知识积累循环
- 触发自我反思和进化
- 在空闲时主动学习
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from turing.agents.agent_factory import AgentFactory
from turing.agents.base_agent import AgentConfig, AgentStatus, BaseAgent
from turing.config import TuringConfig, get_config
from turing.evolution.experience import ExperienceManager
from turing.evolution.reflection import ReflectionEngine
from turing.lean.lean_interface import LeanInterface
from turing.llm.llm_client import ChatMessage, LLMClient
from turing.memory.long_term_memory import KnowledgeEntry, LongTermMemory
from turing.memory.persistent_memory import PersistentMemory
from turing.memory.working_memory import StepStatus, WorkingMemory
from turing.resources.resource_manager import ResourceManager, ResourceLevel
from turing.web.problem_scraper import ProblemScraper


TURING_SYSTEM_PROMPT = """你是 Turing，一个运行在本地 Qwen3-coder:30b 上的数学研究智能体。

你的核心使命是：利用 Lean 4 定理证明器，持续扩展人类可形式化验证的数学知识边界。

你的工作包括：
- 将自然语言数学命题转换为 Lean 4 形式化证明
- 基于模式识别和类比推理提出数学猜想
- 运用合情推理探索数学方向
- 构建和维护结构化的数学知识图谱

工作原则：
1. 验证优先：任何数学结论必须通过 Lean 编译器验证
2. 渐进积累：每次交互都是学习机会
3. 失败即学习：失败的尝试必须记录和分析
4. 资源节约：合理分配计算资源

当你需要执行任务时：
1. 分析任务类型（证明/反驳/猜想/探索/整理）
2. 检索相关的已有知识和经验
3. 制定执行计划
4. 逐步执行，每步用 Lean 验证
5. 总结经验，更新知识库

你可以调度子智能体：Prover、Explorer、Critic、Librarian、Scout、Architect、Evaluator。
根据任务需要决定是否以及何时调度它们。
Evaluator 负责对你的结果进行多维评估，你根据评估报告决定是否进入演化。"""


class TuringAgent(BaseAgent):
    """
    Turing 主智能体 — 系统核心。
    """

    def __init__(
        self,
        config: Optional[TuringConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self._turing_config = config or get_config()

        agent_config = AgentConfig(
            agent_id="turing_main",
            agent_name="Turing",
            base_model=self._turing_config.llm.model,
            system_prompt=TURING_SYSTEM_PROMPT,
            lifecycle="persistent",
            resource_budget={
                "max_tokens_per_call": self._turing_config.agents.default_max_tokens_per_call,
                "max_iterations": 999999,   # 持续运行
                "timeout_minutes": 999999,
            },
        )

        self.llm_instance = llm_client or LLMClient(self._turing_config.llm)
        super().__init__(agent_config, self.llm_instance)

        # 核心组件
        self.resource_manager = ResourceManager(self._turing_config.resources)
        self.lean = LeanInterface(self._turing_config.lean)
        self.ltm = LongTermMemory(self._turing_config.memory.long_term)
        self.pm = PersistentMemory(self._turing_config.memory.persistent)
        self.scraper = ProblemScraper(self._turing_config.web)

        # 演化组件
        self.experience = ExperienceManager(self.pm, self.ltm, self._turing_config.evolution)
        self.reflection = ReflectionEngine(self.pm, self.llm_instance, self._turing_config.evolution)

        # 智能体工厂
        self.factory = AgentFactory(self.llm_instance, self.resource_manager, self.pm)

        # 注册子智能体类型
        self._register_agents()

        # 状态
        self._skill_levels: dict[str, int] = {}
        self._weak_areas: list[str] = []
        self._task_queue: list[dict] = []
        self._recent_results: list[dict] = []  # 最近 N 次结果，用于批量评估
        self._running = False

    def _register_agents(self):
        """注册所有预定义子智能体类型。"""
        from turing.agents.prover import ProverAgent
        from turing.agents.explorer import ExplorerAgent
        from turing.agents.critic import CriticAgent
        from turing.agents.librarian import LibrarianAgent
        from turing.agents.scout import ScoutAgent
        from turing.agents.architect import ArchitectAgent
        from turing.agents.dynamic_agent import DynamicAgent
        from turing.agents.evaluator import EvaluatorAgent

        AgentFactory.register("prover", ProverAgent)
        AgentFactory.register("explorer", ExplorerAgent)
        AgentFactory.register("critic", CriticAgent)
        AgentFactory.register("librarian", LibrarianAgent)
        AgentFactory.register("scout", ScoutAgent)
        AgentFactory.register("architect", ArchitectAgent)
        AgentFactory.register("evaluator", EvaluatorAgent)
        AgentFactory.register("custom", DynamicAgent)
        AgentFactory.register("dynamic", DynamicAgent)

    # ==================================================================
    #  初始化与系统自检
    # ==================================================================

    async def initialize(self):
        """完整的系统初始化序列。"""
        logger.info("=" * 50)
        logger.info("   TURING 数学研究智能体 启动中...")
        logger.info("=" * 50)

        # 1. 资源检测
        snapshot = self.resource_manager.assess()
        logger.info(f"[系统检测] {self.resource_manager.format_report(snapshot)}")

        # 2. Lean 环境验证
        await self.lean.initialize()
        lean_status = await self.lean.check_status()
        logger.info(f"[Lean环境] {json.dumps(lean_status, ensure_ascii=False)}")

        # 3. 记忆加载
        self.ltm.initialize()
        self.pm.initialize()

        ltm_stats = self.ltm.get_stats()
        pm_tasks = self.pm.get_task_count()
        pm_reflections = self.pm.get_reflection_count()

        # 4. 加载历史技能等级
        last_reflection = self.pm.get_latest_reflection()
        if last_reflection:
            try:
                weak = last_reflection.get("weak_areas", "[]")
                if isinstance(weak, str):
                    self._weak_areas = json.loads(weak)
                else:
                    self._weak_areas = weak
            except Exception:
                pass

        self._skill_levels = self._compute_skill_levels()

        # 5. 自检报告
        report = self._generate_status_report(
            snapshot, lean_status, ltm_stats, pm_tasks, pm_reflections
        )
        logger.info(report)

        self.status = AgentStatus.IDLE
        self._running = True

    def _generate_status_report(
        self,
        snapshot,
        lean_status: dict,
        ltm_stats: dict,
        pm_tasks: int,
        pm_reflections: int,
    ) -> str:
        lean_ver = lean_status.get("lean_version", "未知")
        mathlib = "可用" if lean_status.get("mathlib_available") else "不可用"
        thm_count = ltm_stats.get("theorem", 0)
        tactic_count = ltm_stats.get("tactic", 0)

        skills = ", ".join(
            f"{area}={level}" for area, level in self._skill_levels.items()
        ) or "初始化中"

        return f"""
====================================
   TURING 数学研究智能体 v{self._turing_config.system.version}
====================================
模型: {self._turing_config.llm.model} (本地)
Lean: {lean_ver} | Mathlib: {mathlib}
GPU: {snapshot.gpu_name} ({snapshot.gpu_memory_free_gb:.1f}GB free)
RAM: {snapshot.ram_free_gb:.1f}GB / {snapshot.ram_total_gb:.1f}GB
知识库: {thm_count} 条定理, {tactic_count} 条策略
历史任务: {pm_tasks} | 反思次数: {pm_reflections}
活跃子智能体: {snapshot.active_agents}
资源等级: {snapshot.level.value}
当前技能: {skills}
薄弱领域: {', '.join(self._weak_areas) or '无'}
状态: 就绪
===================================="""

    def _compute_skill_levels(self) -> dict[str, int]:
        """根据历史表现计算各领域技能等级。"""
        area_stats = self.pm.get_area_stats()
        levels = {}
        for area, stats in area_stats.items():
            rate = stats.get("success_rate", 0.0)
            total = stats.get("total", 0)
            # 简单的技能等级计算: 基于成功率和经验数量
            level = int(min(10, max(1, rate * 5 + min(total, 20) / 4)))
            levels[area] = level
        return levels

    # ==================================================================
    #  核心任务执行
    # ==================================================================

    async def _execute(self, task: str, **kwargs) -> Any:
        """处理一个数学任务（BaseAgent 的抽象方法实现）。"""
        return await self.process_task(task, **kwargs)

    async def process_task(self, task: str, **kwargs) -> dict:
        """
        处理一个数学任务的完整流程。

        1. 分类 → 2. 检索 → 3. 计划 → 4. 执行 → 5. 验证 → 6. 积累
        """
        start_time = time.time()
        self.working_memory.clear()
        self.working_memory.set_problem(task)
        self.reset_conversation()

        logger.info(f"[Turing] 处理任务: {task[:100]}...")

        # 1. 任务分类
        task_type = await self._classify_task(task)
        self.working_memory.problem_type = task_type

        # 2. 记忆检索（含跨分支已证定理）
        area = kwargs.get("area", "")
        context = await self._retrieve_context(task, area=area)
        self.working_memory.inject_context(context)

        # 3. 获取策略建议
        strategies = self.experience.get_best_strategies(task, category=task_type)
        avoid_list = self.experience.get_avoid_list(task)

        # 4. 制定计划
        plan = await self._make_plan(task, task_type, context, strategies, avoid_list)

        # 5. 执行
        result = await self._execute_plan(task, task_type, plan, **kwargs)

        # 6. 记录经验
        duration = time.time() - start_time
        result["area"] = area  # 确保记录实际数学分支
        await self._record_outcome(task, task_type, result, duration, plan)

        # 7. 评估结果
        result["task"] = task  # 确保 task 字段存在
        self._recent_results.append(result)
        if len(self._recent_results) > 50:
            self._recent_results = self._recent_results[-50:]

        evaluation = await self._evaluate_result(task, result)
        if evaluation:
            result["evaluation"] = evaluation.get("summary", "")

            # 8. 评估结果触发演化
            if evaluation.get("should_evolve"):
                logger.info("[Turing] Evaluator 建议进入演化流程")
                await self._run_evolution(evaluation)

        # 9. 检查是否需要反思
        self.reflection.tick_task()
        if self.reflection.should_reflect():
            logger.info("[Turing] 触发阶段性反思...")
            reflection_result = await self.reflection.reflect()
            self._weak_areas = reflection_result.get("parsed", {}).get("weak_areas", [])
            self._skill_levels = self._compute_skill_levels()

        return result

    # ------------------------------------------------------------------
    #  子流程
    # ------------------------------------------------------------------

    async def _classify_task(self, task: str) -> str:
        """使用 LLM 分类任务类型。"""
        prompt = f"""请判断以下数学任务的类型，只回复一个类别：
prove（证明定理）| disprove（反驳/找反例）| conjecture（提出猜想）|
explore（探索概念）| organize（整理知识）

任务: {task}

类型:"""

        response = await self.think(prompt, temperature=0.1, max_tokens=20)
        response = response.strip().lower()

        for t in ["prove", "disprove", "conjecture", "explore", "organize"]:
            if t in response:
                return t
        return "prove"

    async def _retrieve_context(self, task: str, area: str = "") -> list[dict]:
        """从长期记忆检索相关上下文，包括跨分支已证定理。"""
        context = []

        # 检索语义相关的定理
        theorems = self.ltm.search(task, top_k=5, type_filter="theorem")
        context.extend(theorems)

        # 跨分支已证定理复用
        if area:
            cross_branch = self.ltm.get_cross_branch_theorems(area, limit=10)
            for thm in cross_branch:
                # 避免重复
                if not any(c.get("id") == thm["id"] for c in context):
                    thm["type"] = "proven_theorem"
                    thm["similarity"] = 0.8  # 标记为已证定理
                    context.append(thm)

        # 检索相关策略
        tactics = self.ltm.search(task, top_k=2, type_filter="tactic")
        context.extend(tactics)

        # 检索相关错误（避免重蹈覆辙）
        errors = self.ltm.search(task, top_k=2, type_filter="error_log")
        context.extend(errors)

        return context

    async def _make_plan(
        self,
        task: str,
        task_type: str,
        context: list[dict],
        strategies: list[dict],
        avoid_list: list[dict],
    ) -> str:
        """制定执行计划。"""
        ctx_text = ""
        if context:
            ctx_text = "\n相关知识:\n"
            for c in context[:5]:
                ctx_text += f"  - [{c.get('type','')}] {c.get('natural_language','')[:100]}\n"

        strategy_text = ""
        if strategies:
            strategy_text = "\n推荐策略（按优先级）:\n"
            for s in strategies[:3]:
                strategy_text += f"  - (p={s['priority']:.1f}) {s['strategy'][:100]}\n"

        avoid_text = ""
        if avoid_list:
            avoid_text = "\n应避免的策略:\n"
            for a in avoid_list[:3]:
                avoid_text += f"  - ❌ {a['strategy'][:80]} (原因: {a['reason'][:60]})\n"

        prompt = f"""请为以下任务制定执行计划。

任务类型: {task_type}
任务: {task}
{ctx_text}{strategy_text}{avoid_text}

请给出简洁的分步计划（3-7步），每步标注依赖关系和预期结果。"""

        plan = await self.think(prompt)

        # 记录计划到工作记忆
        self.working_memory.add_step(
            f"计划: {plan[:500]}", status=StepStatus.VERIFIED
        )

        return plan

    async def _execute_plan(
        self, task: str, task_type: str, plan: str, **kwargs
    ) -> dict:
        """执行计划。"""

        if task_type == "prove":
            return await self._execute_proof(task, plan, **kwargs)
        elif task_type == "explore":
            return await self._execute_exploration(task, plan)
        elif task_type == "conjecture":
            return await self._execute_conjecture(task, plan)
        elif task_type == "organize":
            return await self._execute_organization(task)
        elif task_type == "disprove":
            return await self._execute_disproof(task, plan)
        else:
            return await self._execute_proof(task, plan, **kwargs)

    def _build_theorem_toolkit(self, area: str = "") -> str:
        """构建已证定理工具箱，供 Prover 作为可用引理参考。"""
        if not area:
            return ""

        theorems = self.ltm.get_cross_branch_theorems(area, limit=15)
        if not theorems:
            return ""

        lines = ["\n📚 已证定理工具箱（你可以在证明中直接引用这些定理的 Lean 名称）："]
        for thm in theorems:
            name = thm.get("theorem_name", "")
            desc = thm.get("description", thm.get("natural_language", ""))[:80]
            lean = thm.get("lean_code", "")
            thm_area = thm.get("area", "")
            if name and lean:
                # 提取 theorem 声明行
                for line in lean.split("\n"):
                    if line.strip().startswith(("theorem", "lemma", "example")):
                        lines.append(f"  • [{thm_area}] {name}: {line.strip()[:120]}")
                        break
                else:
                    lines.append(f"  • [{thm_area}] {name}: {desc}")

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    async def _execute_proof(self, task: str, plan: str, **kwargs) -> dict:
        """执行证明任务。"""
        area = kwargs.get("area", "")

        # 构建已证定理工具箱
        theorem_toolkit = self._build_theorem_toolkit(area)

        # 第一步：生成自然语言证明思路
        step = self.working_memory.add_step("生成证明思路")
        wm_context = self.working_memory.export_context()

        toolkit_hint = ""
        if theorem_toolkit:
            toolkit_hint = f"\n\n{theorem_toolkit}\n提示：如果已证定理中有可以直接使用的引理，请优先用 exact/apply 引用它们。"

        nl_proof = await self.think_with_context(
            f"请给出以下命题的自然语言证明思路（不需要 Lean 代码）：\n\n{task}\n\n"
            f"提示：优先考虑是否可以用 Lean 的 simp/omega/ring/norm_num 等自动化 tactic 一步解决。"
            f"只有这些都不行时才考虑手动归纳或分步证明。"
            f"{toolkit_hint}",
            context=wm_context,
        )
        self.working_memory.update_step(step.id, status=StepStatus.VERIFIED, content=nl_proof)

        # 第二步：调用 Prover 生成 Lean 代码
        prover = self.factory.create(
            "prover",
            lean_interface=self.lean,
            long_term_memory=self.ltm,
        )

        if prover:
            prover_result = await prover.run(
                task,
                theorem_name=kwargs.get("theorem_name", "main_theorem"),
                hints=nl_proof,
                theorem_toolkit=theorem_toolkit,
                max_attempts=self._turing_config.lean.max_retries,
            )
            await self.factory.destroy(prover.agent_id)

            result_data = prover_result.result or {}
            if isinstance(result_data, dict) and result_data.get("success"):
                proof_step = self.working_memory.add_step(
                    f"Lean 证明成功",
                    status=StepStatus.VERIFIED,
                    lean_code=result_data.get("lean_code", ""),
                )

                # 为定理命名、分类并存储
                naming = await self._name_and_store_theorem(
                    task, result_data.get("lean_code", ""),
                    area=kwargs.get("area", ""),
                )
                result_data["theorem_naming"] = naming

                # 调用 Critic 审查（如果资源允许）
                if self.resource_manager.can_spawn_agent():
                    review = await self._review_proof(
                        task, nl_proof, result_data.get("lean_code", "")
                    )
                    result_data["review"] = review

                return {
                    "success": True,
                    "type": "proof",
                    "task": task,
                    "natural_language_proof": nl_proof,
                    **result_data,
                }
            else:
                step = self.working_memory.add_step(
                    f"证明失败: {result_data.get('last_error', '未知')}",
                    status=StepStatus.ABANDONED,
                )
                self.working_memory.update_step(
                    step.id,
                    abandon_reason=result_data.get("last_error", ""),
                )
                return {
                    "success": False,
                    "type": "proof",
                    "task": task,
                    "natural_language_proof": nl_proof,
                    **result_data,
                }

        # Prover 无法创建（资源不足），自己尝试
        return await self._direct_prove(task, nl_proof)

    async def _direct_prove(self, task: str, hints: str) -> dict:
        """Turing 直接尝试证明（不使用 Prover 子智能体）。"""
        prompt = f"""请将以下命题转为完整的 Lean 4 代码。

命题: {task}
思路: {hints}

⚠️ 要求：
1. 必须 `import Mathlib`
2. 优先用一行 tactic：simp, omega, ring, norm_num, decide, exact?
3. 不能在证明完成后写多余的行
4. 尽量用最短的证明

请给出可编译的 Lean 4 代码，用 ```lean 和 ``` 包围。"""

        response = await self.think(prompt)

        # 提取代码
        from turing.agents.prover import ProverAgent
        lean_code = ProverAgent._extract_lean_code(response)

        if not lean_code:
            return {"success": False, "type": "proof", "task": task, "error": "未能生成代码"}

        result = await self.lean.compile(lean_code)

        if result.success:
            naming = await self._name_and_store_theorem(task, lean_code)
            return {"success": True, "type": "proof", "lean_code": lean_code, "theorem_naming": naming}
        else:
            return {
                "success": False,
                "type": "proof",
                "lean_code": lean_code,
                "error": result.error_summary,
            }

    async def _name_and_store_theorem(
        self, task: str, lean_code: str, area: str = ""
    ) -> dict:
        """
        为已验证的定理命名、分类并存入长期记忆。

        流程:
        1. 在线搜索是否是已知定理（Loogle / ProofWiki）
        2. 用 LLM 生成正式名称和描述
        3. 判断是否为 Mathlib 中不存在的新定理
        4. 存储到长期记忆
        """
        # 第一步：尝试在线搜索已有名称
        web_result = None
        try:
            web_result = await self.scraper.search_theorem_name(task, lean_code)
        except Exception as e:
            logger.debug(f"[命名] 在线搜索失败: {e}")

        # 第二步：用 LLM 命名 + 分类
        web_hint = ""
        if web_result:
            web_hint = f"\n在线查到可能的名称: {web_result.get('name', '')} (来源: {web_result.get('source', '')})"

        naming_prompt = f"""请为以下已验证的数学定理命名、分类并给出简短描述。

定理陈述: {task}

Lean 4 代码:
```lean
{lean_code}
```
{web_hint}

请以 JSON 格式回复（不要 markdown 代码块外壳），字段：
{{
  "theorem_name": "定理的标准名称（英文），如 Nat.add_comm 或 commutativity_of_addition。如果是已知定理用标准名，否则根据内容取一个描述性英文名",
  "chinese_name": "中文名称",
  "area": "数学分支（如 number_theory, algebra, linear_algebra, group_theory, ring_theory, field_theory, analysis, measure_theory, probability, topology, algebraic_topology, geometry, algebraic_geometry, category_theory, combinatorics, logic, set_theory, order_theory, computability, dynamics, model_theory, information_theory, condensed, representation_theory 等）",
  "description": "一句话描述定理内容（中文）",
  "is_novel": true/false,  // 是否可能是 Mathlib 中不存在的新定理（基于你的判断）
  "tags": ["标签1", "标签2"]
}}"""

        response = await self.think(naming_prompt, temperature=0.2, max_tokens=500)

        # 解析 JSON
        naming = {
            "theorem_name": "",
            "chinese_name": "",
            "area": area,
            "description": task[:100],
            "is_novel": False,
            "tags": [],
            "web_source": web_result,
        }

        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                naming.update(parsed)
        except Exception as e:
            logger.debug(f"[命名] JSON 解析失败: {e}")

        # 第三步：存储到长期记忆
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

        # 日志
        novel_tag = " 🆕 新定理!" if is_novel else ""
        logger.info(
            f"[命名] {theorem_name} ({naming.get('chinese_name', '')}) "
            f"[{thm_area}]{novel_tag} — {description[:60]}"
        )

        return naming

    async def _review_proof(
        self, theorem: str, nl_proof: str, lean_code: str
    ) -> dict:
        """使用 Critic 审查证明。"""
        critic = self.factory.create("critic")
        if not critic:
            return {"passed": True, "note": "Critic 不可用，跳过审查"}

        result = await critic.run(
            theorem,
            lean_code=lean_code,
            natural_language_proof=nl_proof,
            theorem_statement=theorem,
        )
        await self.factory.destroy(critic.agent_id)
        return result.result or {}

    async def _execute_exploration(self, task: str, plan: str) -> dict:
        """执行探索任务。"""
        explorer = self.factory.create(
            "explorer", long_term_memory=self.ltm
        )
        if explorer:
            result = await explorer.run(task, depth=3, focus="patterns")
            await self.factory.destroy(explorer.agent_id)
            return {"success": True, "type": "exploration", **(result.result or {})}

        # 后备
        response = await self.think(f"请探索以下数学主题，寻找模式和猜想：\n\n{task}")
        return {"success": True, "type": "exploration", "response": response}

    async def _execute_conjecture(self, task: str, plan: str) -> dict:
        """执行猜想生成任务。"""
        response = await self.think(
            f"基于以下方向，请提出数学猜想并给出支持证据：\n\n{task}"
        )
        if self.ltm:
            self.ltm.add(KnowledgeEntry(
                type="conjecture",
                natural_language=response[:500],
                confidence=0.4,
                source="self_proved",
            ))
        return {"success": True, "type": "conjecture", "response": response}

    async def _execute_organization(self, task: str) -> dict:
        """执行知识整理任务。"""
        librarian = self.factory.create(
            "librarian", long_term_memory=self.ltm
        )
        if librarian:
            result = await librarian.run(task, task_type="organize")
            await self.factory.destroy(librarian.agent_id)
            return {"success": True, "type": "organization", **(result.result or {})}
        return {"success": False, "type": "organization", "error": "无法创建 Librarian"}

    async def _execute_disproof(self, task: str, plan: str) -> dict:
        """执行反驳/找反例任务。"""
        response = await self.think(
            f"请尝试反驳或找到以下命题的反例：\n\n{task}"
        )
        return {"success": True, "type": "disproof", "response": response}

    # ------------------------------------------------------------------
    #  经验记录
    # ------------------------------------------------------------------

    async def _record_outcome(
        self,
        task: str,
        task_type: str,
        result: dict,
        duration: float,
        plan: str,
    ):
        """记录任务结果到持久记忆。"""
        success = result.get("success", False)
        status = "success" if success else "failure"

        # 记录经验
        if success:
            self.experience.record_success(
                context=task,
                strategy=plan[:200],
                lesson=f"成功{task_type}: {task[:100]}",
                category=task_type,
            )
        else:
            error = result.get("error", result.get("last_error", "未知"))
            self.experience.record_failure(
                context=task,
                strategy=plan[:200],
                failure_reason=str(error)[:200],
                category=task_type,
            )

        # 记录任务日志
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
    #  主动学习模式
    # ==================================================================

    async def enter_training_mode(self, duration_minutes: int = 60):
        """
        进入自主训练模式。

        在无外部任务时主动从题库选题、解题、积累经验。
        """
        logger.info(f"[Turing] 进入训练模式 ({duration_minutes} 分钟)")
        end_time = time.time() + duration_minutes * 60

        # 获取问题池
        scout = self.factory.create("scout", problem_scraper=self.scraper)
        problem_pool = []
        if scout:
            result = await scout.run(
                "搜寻训练问题",
                skill_level=max(self._skill_levels.values(), default=1),
                weak_areas=self._weak_areas,
                count=10,
            )
            problem_pool = (result.result or {}).get("problems", [])
            await self.factory.destroy(scout.agent_id)

        problems_solved = 0

        while time.time() < end_time and self._running:
            # 选择下一个问题
            problem = await self._select_training_problem(problem_pool)
            if not problem:
                logger.info("[Turing] 没有更多训练问题")
                break

            logger.info(
                f"[训练] 问题 #{problems_solved + 1}: "
                f"[{problem.get('area', '?')}] {problem.get('title', '')[:60]}"
            )

            # 解题
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

            # 检查是否需要反思
            if self.reflection.should_reflect():
                ref_result = await self.reflection.reflect()
                self._weak_areas = ref_result.get("parsed", {}).get("weak_areas", [])

        logger.info(f"[Turing] 训练模式结束: 解决了 {problems_solved} 个问题")
        return {"problems_solved": problems_solved}

    async def _select_training_problem(
        self, pool: list[dict]
    ) -> Optional[dict]:
        """根据训练策略选择下一个问题。"""
        import random

        cfg = self._turing_config.training
        skill = max(self._skill_levels.values(), default=cfg.initial_skill_level)
        r = random.random()

        if pool:
            if r < cfg.growth_zone_ratio:
                # 成长区
                candidates = [
                    p for p in pool
                    if abs(p.get("difficulty", 5) - (skill + 1)) <= 1
                ]
            elif r < cfg.growth_zone_ratio + cfg.weakness_ratio and self._weak_areas:
                candidates = [
                    p for p in pool
                    if p.get("area", "") in self._weak_areas
                ]
            else:
                candidates = pool

            if candidates:
                choice = random.choice(candidates)
                pool.remove(choice)
                return choice

        # 如果池空了，使用 LLM 生成一个
        scout = self.factory.create("scout", problem_scraper=self.scraper)
        if scout:
            result = await scout.run(
                "生成一个训练问题",
                skill_level=skill,
                weak_areas=self._weak_areas,
                count=1,
            )
            await self.factory.destroy(scout.agent_id)
            problems = (result.result or {}).get("problems", [])
            return problems[0] if problems else None

        return None

    # ==================================================================
    #  评估与演化流程
    # ==================================================================

    async def _evaluate_result(
        self, task: str, result: dict
    ) -> Optional[dict]:
        """使用 Evaluator 评估单次任务结果。"""
        if not self.resource_manager.can_spawn_agent():
            return None

        evaluator = self.factory.create(
            "evaluator",
            persistent_memory=self.pm,
            long_term_memory=self.ltm,
        )
        if not evaluator:
            return None

        try:
            eval_result = await evaluator.run(
                task, mode="result", result=result
            )
            return eval_result.result
        except Exception as e:
            logger.warning(f"[Turing] 评估失败: {e}")
            return None
        finally:
            await self.factory.destroy(evaluator.agent_id)

    async def evaluate_system(self) -> Optional[dict]:
        """
        对系统进行全面评估（public API 供交互模式调用）。
        """
        evaluator = self.factory.create(
            "evaluator",
            persistent_memory=self.pm,
            long_term_memory=self.ltm,
        )
        if not evaluator:
            return None

        try:
            result = await evaluator.run(
                "系统全面评估",
                mode="system",
                system_stats=self.pm.get_comprehensive_stats(),
            )
            return result.result
        except Exception as e:
            logger.error(f"[Turing] 系统评估失败: {e}")
            return None
        finally:
            await self.factory.destroy(evaluator.agent_id)

    async def evaluate_and_evolve(self) -> Optional[dict]:
        """
        完整的评估-演化流程（public API）。

        1. Evaluator 进行系统级评估
        2. Evaluator 基于评估生成演化方案
        3. Turing 执行演化方案
        """
        logger.info("[Turing] ===== 启动评估-演化流程 =====")

        evaluator = self.factory.create(
            "evaluator",
            persistent_memory=self.pm,
            long_term_memory=self.ltm,
        )
        if not evaluator:
            logger.warning("[Turing] 无法创建 Evaluator")
            return None

        try:
            # Phase 1: 系统评估
            eval_result = await evaluator.run(
                "系统评估并生成演化方案",
                mode="evolve",
                system_stats=self.pm.get_comprehensive_stats(),
                skill_levels=self._skill_levels,
            )
            data = eval_result.result or {}
            evaluation = data.get("evaluation")
            evolution_plan = data.get("evolution_plan", {})

            # 记录评估
            if evaluation:
                logger.info(f"[Turing] 评估总分: {evaluation.overall_score:.1f}/10")
                logger.info(f"[Turing] 建议演化: {evaluation.should_evolve}")

            # Phase 2: 执行演化方案
            if evolution_plan:
                await self._execute_evolution_plan(evolution_plan)

            return {
                "evaluation_summary": evaluation.summary() if evaluation else "无",
                "evolution_plan": evolution_plan,
                "executed": bool(evolution_plan),
            }

        except Exception as e:
            logger.error(f"[Turing] 评估-演化流程异常: {e}")
            return None
        finally:
            await self.factory.destroy(evaluator.agent_id)

    async def _run_evolution(self, evaluation_data: dict):
        """根据评估数据运行演化（内部调用）。"""
        evaluator = self.factory.create(
            "evaluator",
            persistent_memory=self.pm,
            long_term_memory=self.ltm,
        )
        if not evaluator:
            return

        try:
            from turing.agents.evaluator import EvaluationReport
            report = evaluation_data.get("report")
            if isinstance(report, EvaluationReport):
                plan = await evaluator.generate_evolution_plan(
                    report, skill_levels=self._skill_levels
                )
            else:
                # 重新评估
                result = await evaluator.run(
                    "生成演化方案",
                    mode="evolve",
                    skill_levels=self._skill_levels,
                )
                plan = (result.result or {}).get("evolution_plan", {})

            if plan:
                await self._execute_evolution_plan(plan)
        except Exception as e:
            logger.error(f"[Turing] 演化执行异常: {e}")
        finally:
            await self.factory.destroy(evaluator.agent_id)

    async def _execute_evolution_plan(self, plan: dict):
        """
        执行演化方案。

        处理 Evaluator 生成的结构化演化方案中的各项变更。
        """
        phase = plan.get("evolution_phase", "未命名")
        logger.info(f"[Turing] 执行演化方案: {phase}")

        # 1. Prompt 变更
        prompt_changes = plan.get("prompt_changes", [])
        for change in prompt_changes:
            agent_id = change.get("agent_id", "")
            content = change.get("content", "")
            reason = change.get("reason", "")
            if agent_id and content:
                try:
                    self.factory.modify_agent_prompt(
                        agent_id, content, reason=reason
                    )
                    logger.info(
                        f"[演化] Prompt 变更: {agent_id} — {reason[:60]}"
                    )
                except Exception as e:
                    logger.warning(f"[演化] Prompt 变更失败 ({agent_id}): {e}")

        # 2. 策略调整
        strategy_adjustments = plan.get("strategy_adjustments", [])
        for adj in strategy_adjustments:
            pattern = adj.get("strategy_pattern", "")
            action = adj.get("action", "")
            reason = adj.get("reason", "")
            if pattern:
                logger.info(
                    f"[演化] 策略调整: {action} '{pattern}' — {reason[:60]}"
                )
                # 通过经验管理器调整优先级
                if action == "boost":
                    experiences = self.pm.get_relevant_experiences(pattern, limit=5)
                    for exp in experiences:
                        self.experience.reinforce(exp.get("id", 0), success=True)
                elif action == "demote":
                    experiences = self.pm.get_relevant_experiences(pattern, limit=5)
                    for exp in experiences:
                        self.experience.reinforce(exp.get("id", 0), success=False)

        # 3. 新子智能体
        new_agents = plan.get("new_agents", [])
        for agent_spec in new_agents:
            name = agent_spec.get("name", "")
            prompt = agent_spec.get("system_prompt", "")
            if name and prompt:
                spec = {
                    "agent_type": agent_spec.get("agent_type", "custom"),
                    "name": name,
                    "system_prompt": prompt,
                    "lifecycle": agent_spec.get("lifecycle", "task_scoped"),
                }
                logger.info(f"[演化] 创建新智能体: {name}")
                self.factory.create_from_spec(spec)

        # 4. 训练重点
        training = plan.get("training_focus", {})
        if training.get("areas"):
            self._weak_areas = training["areas"]
            logger.info(
                f"[演化] 更新训练重点: {self._weak_areas}"
            )

        # 5. 记录演化日志
        expected = plan.get("expected_improvements", {})
        self.pm.record_agent_modification(
            agent_id="turing_main",
            modification_type="evolution",
            reason=phase,
            diff=json.dumps(plan, ensure_ascii=False, default=str)[:2000],
            validation_result="neutral",
        )

        logger.info(f"[Turing] 演化方案执行完成: {phase}")
        logger.info(
            f"[Turing] 预期改进: {json.dumps(expected, ensure_ascii=False, default=str)[:200]}"
        )

    # ==================================================================
    #  智能体生成与修改
    # ==================================================================

    async def create_custom_agent(self, need: str) -> Optional[BaseAgent]:
        """根据需求描述动态创建自定义智能体。"""
        architect = self.factory.create(
            "architect", persistent_memory=self.pm
        )
        if architect:
            spec = await architect.propose_new_agent(need)
            await self.factory.destroy(architect.agent_id)
            return self.factory.create_from_spec(spec)
        return None

    async def optimize_system(self):
        """
        系统优化循环 — 由 Architect 评估并优化整个系统。
        """
        architect = self.factory.create(
            "architect", persistent_memory=self.pm
        )
        if not architect:
            return None

        result = await architect.run(
            "评估并优化 Turing 系统",
            system_stats=self.pm.get_comprehensive_stats(),
            agent_reports=self.factory.list_active(),
        )
        await self.factory.destroy(architect.agent_id)
        return result.result

    # ==================================================================
    #  主循环
    # ==================================================================

    async def main_loop(self):
        """
        Turing 的主运行循环。

        1. 检查外部任务 → 执行
        2. 无外部任务 → 进入训练模式
        3. 定期反思和优化
        """
        logger.info("[Turing] 进入主循环")

        while self._running:
            try:
                # 检查任务队列
                if self._task_queue:
                    task_info = self._task_queue.pop(0)
                    await self.process_task(
                        task_info.get("task", ""),
                        **task_info.get("kwargs", {}),
                    )
                else:
                    # 进入训练模式
                    await self.enter_training_mode(
                        duration_minutes=30
                    )

                # 检查反思
                if self.reflection.should_reflect():
                    ref = await self.reflection.reflect()
                    self._weak_areas = ref.get("parsed", {}).get("weak_areas", [])
                    self._skill_levels = self._compute_skill_levels()

                    # 反思后触发评估-演化循环
                    await self.evaluate_and_evolve()

            except KeyboardInterrupt:
                logger.info("[Turing] 收到中断信号，停止运行")
                break
            except Exception as e:
                logger.error(f"[Turing] 主循环异常: {e}")
                await asyncio.sleep(5)

        logger.info("[Turing] 主循环结束")

    def submit_task(self, task: str, **kwargs):
        """向 Turing 提交一个外部任务。"""
        self._task_queue.append({"task": task, "kwargs": kwargs})
        logger.info(f"[Turing] 任务已提交: {task[:80]}...")

    async def shutdown(self):
        """安全关闭系统。"""
        logger.info("[Turing] 正在关闭...")
        self._running = False

        # 关闭所有子智能体
        await self.factory.destroy_all()

        # 备份持久记忆
        try:
            self.pm.backup()
        except Exception as e:
            logger.warning(f"关闭时备份失败: {e}")

        # 关闭网络连接
        await self.scraper.close()
        await self.llm_instance.close()

        logger.info("[Turing] 已安全关闭")
