"""
经验管理器 — 管理经验强化循环。

- 成功策略 priority 提升，优先采用
- 失败策略 priority 降低，主动规避
- 追踪策略应用效果
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from turing.config import EvolutionConfig, get_config
from turing.memory.long_term_memory import LongTermMemory
from turing.memory.persistent_memory import PersistentMemory


class ExperienceManager:
    """
    经验强化管理器。

    将每次任务的成功/失败转化为可检索的经验，
    通过优先级机制实现"优先采用成功策略，避免重蹈覆辙"。
    """

    def __init__(
        self,
        persistent_memory: Optional[PersistentMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        config: Optional[EvolutionConfig] = None,
    ):
        self.pm = persistent_memory or PersistentMemory()
        self.ltm = long_term_memory or LongTermMemory()
        self.config = config or get_config().evolution

    def record_success(
        self,
        context: str,
        strategy: str,
        lesson: str = "",
        category: str = "",
        tags: Optional[list[str]] = None,
    ) -> int:
        """记录成功经验。"""
        exp_id = self.pm.record_experience(
            context=context,
            strategy_used=strategy,
            outcome="success",
            lesson=lesson or f"策略'{strategy}'在此类问题上有效",
            category=category,
            tags=tags,
        )

        # 同时更新长期记忆中对应策略的成功率
        if self.ltm:
            tactics = self.ltm.search_tactics(strategy, top_k=1)
            if tactics:
                self.ltm.update_success_rate(tactics[0]["id"], success=True)

        logger.info(f"记录成功经验 #{exp_id}: {strategy[:60]}")
        return exp_id

    def record_failure(
        self,
        context: str,
        strategy: str,
        failure_reason: str,
        lesson: str = "",
        category: str = "",
        tags: Optional[list[str]] = None,
    ) -> int:
        """记录失败经验。"""
        if not lesson:
            lesson = f"策略'{strategy}'失败，原因: {failure_reason}"

        exp_id = self.pm.record_experience(
            context=context,
            strategy_used=strategy,
            outcome="failure",
            lesson=lesson,
            category=category,
            tags=tags,
        )

        # 更新长期记忆成功率
        if self.ltm:
            tactics = self.ltm.search_tactics(strategy, top_k=1)
            if tactics:
                self.ltm.update_success_rate(tactics[0]["id"], success=False)

        logger.info(f"记录失败经验 #{exp_id}: {failure_reason[:60]}")
        return exp_id

    def record_partial(
        self,
        context: str,
        strategy: str,
        partial_result: str,
        lesson: str = "",
        category: str = "",
        tags: Optional[list[str]] = None,
    ) -> int:
        """记录部分成功的经验。"""
        exp_id = self.pm.record_experience(
            context=context,
            strategy_used=strategy,
            outcome="partial",
            lesson=lesson or f"策略'{strategy}'部分有效: {partial_result}",
            category=category,
            tags=tags,
        )
        logger.info(f"记录部分成功经验 #{exp_id}")
        return exp_id

    def reinforce(self, experience_id: int, success: bool):
        """
        根据经验的再次应用结果更新优先级。

        成功: priority += increment (上限 1.0)
        失败: priority -= decrement (下限 0.0)
        """
        self.pm.update_priority(experience_id, success)
        action = "强化" if success else "削弱"
        logger.debug(f"经验 #{experience_id} 已{action}")

    def get_best_strategies(
        self,
        context: str,
        category: str = "",
        limit: int = 5,
    ) -> list[dict]:
        """
        获取最佳策略推荐。

        综合考虑：
        1. 上下文相关的成功经验（同类问题的直接经验）
        2. 高优先级的通用成功策略
        3. 需要避免的失败模式
        """
        recommendations = []

        # 1. 直接相关经验
        relevant = self.pm.get_relevant_experiences(
            context, limit=3, outcome_filter="success"
        )
        for exp in relevant:
            recommendations.append({
                "strategy": exp["strategy_used"],
                "priority": exp["priority"],
                "source": "direct_experience",
                "lesson": exp.get("lesson", ""),
            })

        # 2. 同类高优先级策略
        top = self.pm.get_top_strategies(category=category, limit=3)
        for exp in top:
            if exp["strategy_used"] not in [r["strategy"] for r in recommendations]:
                recommendations.append({
                    "strategy": exp["strategy_used"],
                    "priority": exp["priority"],
                    "source": "top_strategy",
                    "lesson": exp.get("lesson", ""),
                })

        # 按优先级排序
        recommendations.sort(key=lambda x: x["priority"], reverse=True)

        return recommendations[:limit]

    def get_avoid_list(self, context: str) -> list[dict]:
        """获取应该避免的策略列表。"""
        failures = self.pm.get_relevant_experiences(
            context, limit=5, outcome_filter="failure"
        )
        return [
            {
                "strategy": f["strategy_used"],
                "reason": f.get("lesson", ""),
                "priority": f["priority"],
            }
            for f in failures
            if f["priority"] < 0.3  # 只返回多次失败的策略
        ]

    def get_failure_patterns(self, limit: int = 10) -> list[dict]:
        """获取常见失败模式（用于反思）。"""
        return self.pm.get_failure_patterns(limit=limit)

    def log_task_completion(
        self,
        task_type: str,
        description: str,
        status: str,
        duration: float,
        strategies: Optional[list[str]] = None,
        final_strategy: str = "",
        lean_attempts: int = 0,
        lean_success: bool = False,
        area: str = "",
        difficulty: int = 0,
    ) -> int:
        """记录任务完成情况。"""
        return self.pm.log_task(
            task_type=task_type,
            task_description=description,
            status=status,
            duration_seconds=duration,
            strategies_tried=strategies,
            final_strategy=final_strategy,
            lean_attempts=lean_attempts,
            lean_success=lean_success,
            area=area,
            difficulty=difficulty,
        )
