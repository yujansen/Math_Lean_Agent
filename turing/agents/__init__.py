"""
智能体层 — 提供基础 Agent 抽象和具体实现。

推荐入口::

    from turing.agents import SkillBasedTuringAgent   # v2 技能驱动（推荐）
    from turing.agents import TuringAgent              # v1 多智能体（legacy）

- :class:`BaseAgent`: 所有智能体的抽象父类（生命周期 / LLM 通信 / 资源控制）
- :class:`SkillBasedTuringAgent`: v2 技能驱动单智能体（推荐，LLM 调用 ↓75%）
- :class:`TuringAgent`: v1 多智能体调度器（legacy，``--mode multi``）

Legacy 子智能体（Prover / Critic / Explorer 等）已移至 :mod:`turing.agents.legacy`。
"""

from .base_agent import BaseAgent
from .skill_based_agent import SkillBasedTuringAgent

__all__ = ["BaseAgent", "SkillBasedTuringAgent"]
