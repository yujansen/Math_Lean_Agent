"""
自我演化层 — 经验积累与周期性反思。

- :class:`ExperienceManager`: 从任务结果提取可复用经验并存入长期记忆
- :class:`ReflectionEngine`: 触发阶段性反思，分析弱点并生成改进计划
"""

from .experience import ExperienceManager
from .reflection import ReflectionEngine

__all__ = ["ExperienceManager", "ReflectionEngine"]
