"""
技能系统 — 基于同一 LLM 通过技能定义智能体行为，替代多智能体架构。

核心理念：
  原多智能体架构中，Prover / Critic / Explorer / Evaluator 等分别创建独立的
  Agent 实例，每次创建都带有独立的 system prompt 和对话历史，导致大量冗余
  的 LLM 调用（单次证明任务需 9-12 次）。

  技能架构将所有"智能体行为"抽象为 **Skill**：
  - 每个 Skill = prompt 模板 + 响应解析器 + 可选的后处理逻辑
  - 单一 Agent 持有所有 Skill，通过 SkillRegistry 动态调用
  - 多个步骤可合并为单次 LLM 调用（如分类+计划+证明大纲合为一次）
  - 对话上下文在同一 Agent 内延续，无需重复说明任务背景

  单次证明任务 LLM 调用次数：9-12 → 3-7
"""

from turing.skills.skill_registry import Skill, SkillRegistry
from turing.skills.math_skills import register_all_skills

__all__ = ["Skill", "SkillRegistry", "register_all_skills"]
