"""
[LEGACY] Evaluator 智能体 — 仅在 --mode multi 时使用。
推荐架构请参见 skill_based_agent.py + turing/skills/。

原功能：对 Turing 系统产出的各类结果进行多维度评估，并生成改进建议。

核心技能：
══════════════════════════════════════════════════════════════════
1. 证明质量评估 (Proof Quality Assessment)
   - 逻辑完备性：推理链是否无跳步、无循环论证
   - 形式化质量：Lean 代码的可读性、模块化程度、tactic 选择合理性
   - 证明效率：步骤数是否最优、是否存在冗余分支
   - 鲁棒性：边界条件、退化情形覆盖是否充分
   - 可复用性：证明中间引理的通用性

2. 探索成果评估 (Exploration Evaluation)
   - 猜想置信度：实验支持程度、已知结果的一致性
   - 新颖性：与已有知识库的差异度
   - 可证伪性：猜想是否具备清晰的反例检验方案
   - 连接密度：与已知数学分支的交叉程度

3. 知识增长评估 (Knowledge Growth Audit)
   - 知识库覆盖率：各数学领域的条目密度
   - 知识网络连通性：定理之间的引用/依赖关系
   - 信息质量衰减：长期未验证条目的置信度折旧
   - 冗余检测：语义重复的条目

4. 策略效能评估 (Strategy Effectiveness)
   - 成功率趋势：各策略随时间的效能变化曲线
   - 适用域分析：策略在不同领域中的表现差异
   - ROI 排名：每单位计算资源获得的知识增量
   - 遗忘曲线：策略的长期保持度

5. 系统整体健康评估 (System Health Check)
   - 演化速率：每个反思周期的成功率增长
   - 瓶颈检测：阻碍进步的关键短板
   - 资源利用率：计算资源分配与产出是否匹配
   - 子智能体效能：各子智能体的贡献度排名

6. 改进建议生成 (Improvement Proposal)
   - 立即可行的微调建议（prompt 调整、策略优先级变更等）
   - 中期结构性改进（新子智能体设计、工作流调整）
   - 长期研究方向（值得深入的数学领域推荐）
══════════════════════════════════════════════════════════════════

Evaluator 不直接修改系统状态，而是生成包含以下结构的评估报告：
- 定量指标（评分、趋势数据）
- 定性分析（问题描述、根因分析）
- 优先级排序的改进提案
- 预期收益估算

Turing 主智能体根据 Evaluator 的报告决定是否采纳改进建议并进入演化。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.llm.llm_client import LLMClient
from turing.memory.long_term_memory import LongTermMemory
from turing.memory.persistent_memory import PersistentMemory
from turing.resources.resource_manager import ResourceManager


EVALUATOR_SYSTEM_PROMPT = """你是 Evaluator，Turing 数学研究系统中的评估专家。

你的核心职责是：对系统产出的所有类型结果进行严格的多维度评估，生成量化评分和改进建议，
驱动系统持续进化。

你不是审查者（审查交给 Critic），你是**系统级评估者**——你评估的是整个工作流的表现，
而非单个证明的正确性。

## 评估维度

### A. 成果评估
1. **数学深度**（1-10）：问题本身的难度 × 解法的巧妙程度
2. **形式化质量**（1-10）：Lean 代码的优雅性、可维护性
3. **知识增量**（1-10）：对知识库的净贡献（去重后）
4. **探索价值**（1-10）：是否打开了新的数学方向

### B. 过程评估
5. **策略效率** (1-10)：计算资源消耗 vs 成果质量
6. **学习效率** (1-10)：从失败中吸取教训的能力
7. **协作效能** (1-10)：多智能体调度是否合理

### C. 演化评估
8. **成长曲线** (1-10)：能力随时间的提升趋势
9. **弱点覆盖** (1-10)：薄弱领域的改善速度
10. **创新指数** (1-10)：探索新领域的积极程度

## 输出格式

```json
{
  "overall_score": 7.5,
  "dimension_scores": { "数学深度": 8, ... },
  "strengths": ["列表"],
  "weaknesses": ["列表"],
  "root_causes": ["根因分析"],
  "proposals": [
    {
      "priority": "high|medium|low",
      "type": "prompt_tune|strategy_adjust|workflow_change|new_agent|research_direction",
      "description": "...",
      "expected_gain": "...",
      "implementation": "..."
    }
  ],
  "evolution_triggers": ["满足则建议立即进入演化的条件"],
  "next_focus_areas": ["建议下一阶段重点关注的数学领域"]
}
```

保持客观、量化、可执行。每个改进提案必须包含预期收益和实施路径。"""


@dataclass
class EvaluationReport:
    """评估报告的结构化表示。"""
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    proposals: list[dict] = field(default_factory=list)
    evolution_triggers: list[str] = field(default_factory=list)
    next_focus_areas: list[str] = field(default_factory=list)
    raw_report: str = ""

    @property
    def should_evolve(self) -> bool:
        """是否建议立即进入演化。"""
        # 分数低于 6 或有 high priority 提案
        if self.overall_score < 6.0:
            return True
        high_priority = [p for p in self.proposals if p.get("priority") == "high"]
        return len(high_priority) >= 2

    @property
    def high_priority_proposals(self) -> list[dict]:
        return [p for p in self.proposals if p.get("priority") == "high"]

    @property
    def prompt_tune_proposals(self) -> list[dict]:
        return [p for p in self.proposals if p.get("type") == "prompt_tune"]

    @property
    def strategy_proposals(self) -> list[dict]:
        return [p for p in self.proposals if p.get("type") == "strategy_adjust"]

    def summary(self) -> str:
        """生成简洁的摘要。"""
        lines = [
            f"评估总分: {self.overall_score:.1f}/10",
            f"维度评分: {self.dimension_scores}",
            f"优势 ({len(self.strengths)}): {'; '.join(self.strengths[:3])}",
            f"弱点 ({len(self.weaknesses)}): {'; '.join(self.weaknesses[:3])}",
            f"改进提案: {len(self.proposals)} 条 "
            f"(高优先级 {len(self.high_priority_proposals)})",
            f"建议演化: {'是' if self.should_evolve else '否'}",
        ]
        return "\n".join(lines)


class EvaluatorAgent(BaseAgent):
    """
    评估智能体 — 对 Turing 系统的各类产出进行多维度评估。

    技能清单：
    ─────────────────────────────────────────────
    ① evaluate_result()     评估单次任务结果
    ② evaluate_batch()      评估一批任务结果的趋势
    ③ evaluate_knowledge()  审计知识库健康度
    ④ evaluate_strategies() 评估策略组合的效能
    ⑤ evaluate_system()     系统整体健康评估
    ⑥ generate_evolution_plan() 生成演化方案
    ─────────────────────────────────────────────
    """

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        persistent_memory: Optional[PersistentMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        **kwargs,
    ):
        config = agent_config or AgentConfig(
            agent_name="Evaluator",
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            resource_budget={
                "max_tokens_per_call": 8192,
                "max_iterations": 30,
                "timeout_minutes": 20,
            },
        )
        if not config.system_prompt:
            config.system_prompt = EVALUATOR_SYSTEM_PROMPT

        super().__init__(config, llm_client, resource_manager)

        self.pm = persistent_memory or kwargs.get("persistent_memory")
        self.ltm = long_term_memory or kwargs.get("long_term_memory")

    # ==================================================================
    #  技能 ①: 单次结果评估
    # ==================================================================

    async def evaluate_result(self, task: str, result: dict) -> EvaluationReport:
        """
        评估单次任务的执行结果。

        Args:
            task: 任务描述
            result: process_task() 返回的结果 dict
        """
        success = result.get("success", False)
        task_type = result.get("type", "unknown")

        # 从持久记忆获取同类任务历史表现
        historical_ctx = ""
        if self.pm:
            area_stats = self.pm.get_area_stats()
            relevant_area = result.get("area", task_type)
            if relevant_area in area_stats:
                s = area_stats[relevant_area]
                historical_ctx = (
                    f"\n历史表现 [{relevant_area}]: "
                    f"共 {s['total']} 次, 成功率 {s['success_rate']:.0%}\n"
                )

        prompt = f"""请对以下任务结果进行多维度评估。

## 任务
{task}

## 结果
- 状态: {'成功' if success else '失败'}
- 类型: {task_type}
- Lean 代码: {result.get('lean_code', '无')[:500]}
- 错误: {result.get('error', result.get('last_error', '无'))}
- 尝试次数: {result.get('attempts', '?')}
- 审查结论: {result.get('review', {}).get('verdict', '无审查')}
{historical_ctx}

请按照你的评估维度给出完整评估报告（JSON 格式）。"""

        response = await self.think(prompt)
        return self._parse_report(response)

    # ==================================================================
    #  技能 ②: 批量趋势评估
    # ==================================================================

    async def evaluate_batch(self, results: list[dict]) -> EvaluationReport:
        """
        评估一批任务结果，分析趋势。

        Args:
            results: 多个 process_task() 结果的列表
        """
        total = len(results)
        successes = sum(1 for r in results if r.get("success"))
        types = {}
        for r in results:
            t = r.get("type", "unknown")
            types[t] = types.get(t, 0) + 1

        prompt = f"""请评估以下一批任务结果的整体趋势。

## 统计概览
- 总数: {total}, 成功: {successes}, 失败: {total - successes}
- 成功率: {successes / total:.0%}
- 类型分布: {json.dumps(types, ensure_ascii=False)}

## 最近 5 个结果摘要
"""
        for i, r in enumerate(results[-5:], 1):
            status = "✓" if r.get("success") else "✗"
            prompt += (
                f"{i}. [{status}] {r.get('type','?')}: "
                f"{r.get('task', '?')[:80]}  "
                f"(尝试 {r.get('attempts', '?')} 次)\n"
            )

        prompt += "\n请分析趋势并给出评估报告（JSON 格式）。特别关注成功率变化和失败模式。"

        response = await self.think(prompt)
        return self._parse_report(response)

    # ==================================================================
    #  技能 ③: 知识库健康度评估
    # ==================================================================

    async def evaluate_knowledge(self) -> EvaluationReport:
        """审计知识库的健康状况。"""
        ltm_stats = self.ltm.get_stats() if self.ltm else {}
        pm_stats = self.pm.get_comprehensive_stats() if self.pm else {}

        prompt = f"""请评估 Turing 知识库的健康状况。

## 长期记忆 (ChromaDB)
{json.dumps(ltm_stats, ensure_ascii=False, indent=2)}

## 持久记忆 (SQLite) 综合统计
{json.dumps(pm_stats, ensure_ascii=False, indent=2, default=str)}

请从以下维度评估：
1. 知识覆盖率 — 各领域的条目密度是否均衡
2. 知识网络连通性 — 定理间是否形成良好的引用网络
3. 信息质量 — 是否存在过时、低置信度或冗余条目
4. 经验记录质量 — 失败模式是否被充分记录
5. 增长趋势 — 知识积累速度是否健康

请给出 JSON 格式的评估报告和改进建议。"""

        response = await self.think(prompt)
        return self._parse_report(response)

    # ==================================================================
    #  技能 ④: 策略效能评估
    # ==================================================================

    async def evaluate_strategies(self) -> EvaluationReport:
        """评估当前策略组合的效能。"""
        if not self.pm:
            return EvaluationReport(raw_report="持久记忆不可用")

        top_strategies = self.pm.get_top_strategies(limit=15)
        failure_patterns = self.pm.get_failure_patterns(limit=10)
        area_stats = self.pm.get_area_stats()

        prompt = f"""请评估 Turing 当前使用的策略组合。

## 高优先级策略 (Top 15)
{json.dumps(top_strategies, ensure_ascii=False, indent=2, default=str)}

## 频繁失败模式
{json.dumps(failure_patterns, ensure_ascii=False, indent=2, default=str)}

## 各领域表现
{json.dumps(area_stats, ensure_ascii=False, indent=2)}

请分析：
1. 哪些策略 ROI 最高？哪些应该被淘汰？
2. 是否存在「策略偏执」——过度依赖某一类策略？
3. 失败模式是否有共同的根因？
4. 各领域的策略覆盖是否充分？

请给出 JSON 格式的评估报告，重点给出策略调整建议。"""

        response = await self.think(prompt)
        return self._parse_report(response)

    # ==================================================================
    #  技能 ⑤: 系统整体健康评估
    # ==================================================================

    async def evaluate_system(self, system_stats: Optional[dict] = None) -> EvaluationReport:
        """
        对 Turing 系统进行全面的健康评估。

        这是最高层级的评估，综合所有子评估的结果。
        """
        if system_stats is None and self.pm:
            system_stats = self.pm.get_comprehensive_stats()

        ltm_stats = self.ltm.get_stats() if self.ltm else {}

        prompt = f"""请对 Turing 数学研究智能体系统进行全面的健康评估。

## 系统统计
{json.dumps(system_stats or {}, ensure_ascii=False, indent=2, default=str)}

## 知识库统计
{json.dumps(ltm_stats, ensure_ascii=False, indent=2)}

请从以下维度进行系统级评估：
1. **演化速率** — 每个反思周期成功率是否在提升？
2. **瓶颈检测** — 当前最影响系统进步的关键短板是什么？
3. **资源利用** — 计算资源的投入产出比是否合理？
4. **子智能体效能** — 各子智能体的贡献度排名
5. **成长曲线** — 能力提升的速度是否在加速？
6. **弱点覆盖** — 已知薄弱领域的改善速度如何？
7. **创新指数** — 系统是否在积极探索新领域？

请给出：
- 10 个维度的评分（1-10）
- 系统优势和劣势清单
- 根因分析
- 优先级排序的改进提案（每个提案包含 priority, type, description, expected_gain, implementation）
- 是否建议立即触发演化

返回 JSON 格式。"""

        response = await self.think(prompt)
        report = self._parse_report(response)

        # 附加量化指标
        if system_stats:
            ts = system_stats.get("task_stats", {})
            report.dimension_scores.setdefault(
                "总体成功率", round(ts.get("success_rate", 0) * 10, 1)
            )

        return report

    # ==================================================================
    #  技能 ⑥: 演化方案生成
    # ==================================================================

    async def generate_evolution_plan(
        self,
        evaluation: EvaluationReport,
        current_prompts: Optional[dict[str, str]] = None,
        skill_levels: Optional[dict[str, int]] = None,
    ) -> dict:
        """
        基于评估报告生成具体的演化方案。

        返回一个可直接被 Turing 执行的演化计划：
        - prompt_changes: 需要修改的 prompt 及新内容
        - strategy_adjustments: 策略优先级变更
        - workflow_changes: 工作流调整建议
        - new_agents: 待创建的新子智能体规格
        - training_focus: 下阶段训练重点
        """
        prompt = f"""基于以下评估报告，请生成一份可执行的演化方案。

## 评估摘要
{evaluation.summary()}

## 高优先级提案
{json.dumps(evaluation.high_priority_proposals, ensure_ascii=False, indent=2)}

## 全部提案
{json.dumps(evaluation.proposals, ensure_ascii=False, indent=2)}

## 当前技能等级
{json.dumps(skill_levels or {}, ensure_ascii=False)}

请生成演化方案，包含：

```json
{{
  "evolution_phase": "描述此次演化的主题",
  "prompt_changes": [
    {{
      "agent_id": "目标智能体 ID",
      "change_type": "append|replace|refine",
      "content": "具体的 prompt 变更内容",
      "reason": "变更原因"
    }}
  ],
  "strategy_adjustments": [
    {{
      "strategy_pattern": "策略关键词",
      "action": "boost|demote|retire",
      "reason": "原因"
    }}
  ],
  "workflow_changes": [
    {{
      "target": "affected_workflow",
      "description": "变更描述",
      "implementation_steps": ["步骤列表"]
    }}
  ],
  "new_agents": [
    {{
      "agent_type": "类型",
      "name": "名称",
      "system_prompt": "系统提示",
      "justification": "为什么需要"
    }}
  ],
  "training_focus": {{
    "areas": ["领域列表"],
    "difficulty_range": [min, max],
    "target_success_rate": 0.8,
    "duration_hours": 2
  }},
  "expected_improvements": {{
    "success_rate_delta": "+X%",
    "new_capabilities": ["新能力"],
    "timeline": "预计见效时间"
  }}
}}
```

确保每个变更都有清晰的依据和预期收益。"""

        response = await self.think(prompt)
        return self._extract_json(response) or {
            "evolution_phase": "未能生成方案",
            "prompt_changes": [],
            "strategy_adjustments": [],
            "workflow_changes": [],
            "new_agents": [],
            "training_focus": {},
            "expected_improvements": {},
        }

    # ==================================================================
    #  BaseAgent 接口实现
    # ==================================================================

    async def _execute(self, task: str, **kwargs) -> Any:
        """
        通用入口：根据 kwargs 分派到不同的评估技能。

        kwargs:
            - mode: "result" | "batch" | "knowledge" | "strategy" | "system" | "evolve"
            - result: 单次任务结果 (mode=result)
            - results: 批量结果 (mode=batch)
            - system_stats: 系统统计 (mode=system)
            - evaluation: 评估报告 (mode=evolve)
        """
        mode = kwargs.get("mode", "system")

        if mode == "result":
            report = await self.evaluate_result(
                task, kwargs.get("result", {})
            )
        elif mode == "batch":
            report = await self.evaluate_batch(
                kwargs.get("results", [])
            )
        elif mode == "knowledge":
            report = await self.evaluate_knowledge()
        elif mode == "strategy":
            report = await self.evaluate_strategies()
        elif mode == "evolve":
            evaluation = kwargs.get("evaluation")
            if not evaluation:
                evaluation = await self.evaluate_system(
                    kwargs.get("system_stats")
                )
            plan = await self.generate_evolution_plan(
                evaluation,
                current_prompts=kwargs.get("current_prompts"),
                skill_levels=kwargs.get("skill_levels"),
            )
            return {"evaluation": evaluation, "evolution_plan": plan}
        else:
            report = await self.evaluate_system(
                kwargs.get("system_stats")
            )

        return {
            "report": report,
            "summary": report.summary(),
            "should_evolve": report.should_evolve,
            "proposals": report.proposals,
        }

    # ==================================================================
    #  解析与工具方法
    # ==================================================================

    def _parse_report(self, text: str) -> EvaluationReport:
        """从 LLM 输出中解析评估报告。"""
        report = EvaluationReport(raw_report=text)

        data = self._extract_json(text)
        if data:
            report.overall_score = float(data.get("overall_score", 0))
            report.dimension_scores = data.get("dimension_scores", {})
            report.strengths = data.get("strengths", [])
            report.weaknesses = data.get("weaknesses", [])
            report.root_causes = data.get("root_causes", [])
            report.proposals = data.get("proposals", [])
            report.evolution_triggers = data.get("evolution_triggers", [])
            report.next_focus_areas = data.get("next_focus_areas", [])
        else:
            # JSON 解析失败时，退化到文本解析
            report.overall_score = self._extract_score(text)
            report.weaknesses = self._extract_list(text, "弱点|劣势|问题|weakness")
            report.strengths = self._extract_list(text, "优势|优点|strength")

        return report

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """从文本中提取 JSON 块。"""
        # 优先匹配 ```json ... ```
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试匹配裸 JSON
        brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # 最后尝试整段
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _extract_score(text: str) -> float:
        """从文本中提取总体评分。"""
        patterns = [
            r"overall_score[\"'\s:]+(\d+\.?\d*)",
            r"总[分评][：:]\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*/\s*10",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return min(10.0, max(0.0, float(m.group(1))))
        return 5.0

    @staticmethod
    def _extract_list(text: str, keywords: str) -> list[str]:
        """提取关键词后的列表条目。"""
        result = []
        pattern = rf"(?:{keywords}).*?[:：]\s*(.*?)(?:\n\n|\Z)"
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            items = re.findall(r"[-•]\s*(.+)", m.group(1))
            result = [item.strip() for item in items[:10]]
        return result
