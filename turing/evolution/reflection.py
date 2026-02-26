"""
反思引擎 — 实现阶段性反思，评估能力进步与不足。

每 20 个任务或 24 小时触发一次系统性反思。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from turing.config import EvolutionConfig, get_config
from turing.llm.llm_client import ChatMessage, LLMClient
from turing.memory.persistent_memory import PersistentMemory


class ReflectionEngine:
    """
    阶段性反思引擎。

    依据持久记忆中的任务日志和经验，生成反思报告，
    识别薄弱领域并制定改进计划。
    """

    def __init__(
        self,
        persistent_memory: Optional[PersistentMemory] = None,
        llm_client: Optional[LLMClient] = None,
        config: Optional[EvolutionConfig] = None,
    ):
        self.pm = persistent_memory or PersistentMemory()
        self.llm = llm_client or LLMClient()
        self.config = config or get_config().evolution

        self._last_reflection_time: float = 0
        self._tasks_since_reflection: int = 0

    def should_reflect(self) -> bool:
        """判断是否应该执行反思。"""
        # 条件 1: 任务数达到阈值
        if self._tasks_since_reflection >= self.config.reflection_task_interval:
            return True

        # 条件 2: 时间间隔达到阈值
        elapsed_hours = (time.time() - self._last_reflection_time) / 3600
        if self._last_reflection_time > 0 and elapsed_hours >= self.config.reflection_time_interval:
            return True

        # 初次运行
        if self._last_reflection_time == 0 and self._tasks_since_reflection > 3:
            return True

        return False

    def tick_task(self):
        """标记一个任务已完成（用于计数）。"""
        self._tasks_since_reflection += 1

    async def reflect(self) -> dict:
        """
        执行完整的阶段性反思。

        Returns:
            反思报告（结构化 dict）
        """
        logger.info("=" * 50)
        logger.info("  开始阶段性反思")
        logger.info("=" * 50)

        # 1. 收集数据
        task_stats = self.pm.get_task_stats()
        area_stats = self.pm.get_area_stats()
        failure_patterns = self.pm.get_failure_patterns(10)
        reflection_count = self.pm.get_reflection_count()
        last_reflection = self.pm.get_latest_reflection()

        # 2. 构建反思上下文
        context = self._build_reflection_context(
            task_stats, area_stats, failure_patterns, last_reflection
        )

        # 3. LLM 生成反思报告
        report_text = await self._generate_reflection(context, reflection_count + 1)

        # 4. 解析报告
        parsed = self._parse_reflection(report_text, task_stats, area_stats)

        # 5. 存入持久记忆
        self.pm.record_reflection(
            phase=reflection_count + 1,
            success_rate=task_stats.get("success_rate", 0.0),
            skills_gained=parsed.get("skills_gained", []),
            weak_areas=parsed.get("weak_areas", []),
            theorems_proved=task_stats.get("success", 0),
            reflection_report=report_text,
            improvements_planned=parsed.get("improvements", []),
        )

        # 6. 备份
        try:
            self.pm.backup()
        except Exception as e:
            logger.warning(f"反思后备份失败: {e}")

        # 7. 重置计数器
        self._last_reflection_time = time.time()
        self._tasks_since_reflection = 0

        logger.info("反思完成")
        return {
            "phase": reflection_count + 1,
            "report": report_text,
            "parsed": parsed,
            "stats": task_stats,
        }

    def _build_reflection_context(
        self,
        task_stats: dict,
        area_stats: dict,
        failure_patterns: list,
        last_reflection: Optional[dict],
    ) -> str:
        """构建反思所需的上下文。"""
        parts = []

        parts.append(f"## 任务统计\n{json.dumps(task_stats, ensure_ascii=False, indent=2)}")

        if area_stats:
            parts.append(f"## 按领域统计\n{json.dumps(area_stats, ensure_ascii=False, indent=2)}")

        if failure_patterns:
            parts.append("## 常见失败模式")
            for fp in failure_patterns:
                parts.append(f"  - ({fp.get('count', 0)}次) {fp.get('lesson', '')[:100]}")

        if last_reflection:
            parts.append(f"## 上次反思")
            parts.append(f"  时间: {last_reflection.get('timestamp', '?')}")
            parts.append(f"  成功率: {last_reflection.get('success_rate', 0):.1%}")
            weak = last_reflection.get("weak_areas", "[]")
            if isinstance(weak, str):
                weak = json.loads(weak)
            if weak:
                parts.append(f"  上次薄弱领域: {', '.join(weak)}")
            improvements = last_reflection.get("improvements_planned", "[]")
            if isinstance(improvements, str):
                improvements = json.loads(improvements)
            if improvements:
                parts.append(f"  计划的改进: {', '.join(improvements)}")

        return "\n\n".join(parts)

    async def _generate_reflection(self, context: str, phase: int) -> str:
        """使用 LLM 生成反思报告。"""
        prompt = f"""请作为 Turing 数学智能体执行第 {phase} 次阶段性反思。

根据以下系统数据，生成一份完整的反思报告：

{context}

请严格按照以下格式输出：

## 阶段反思报告 #{phase}
### 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

### 1. 能力评估
- 总体成功率及趋势
- 按领域分析
- Lean 编译效率

### 2. 进步
- 新掌握的技能/策略
- 量化进步指标

### 3. 不足与薄弱领域
- 失败率最高的领域
- 反复出现的错误模式
- 缺失的关键知识

### 4. 改进计划
- 针对每个薄弱领域的具体行动
- 系统优化建议
- 优先级排序

### 5. 下一阶段目标
- 3个具体可衡量的目标"""

        messages = [ChatMessage(role="user", content=prompt)]
        response = await self.llm.chat(
            messages,
            system_prompt="你是一个严谨的数学 AI 系统的反思模块。请基于数据给出客观、具体的评估。",
        )
        return response.content

    def _parse_reflection(
        self,
        report: str,
        task_stats: dict,
        area_stats: dict,
    ) -> dict:
        """从反思报告中提取结构化信息。"""
        parsed = {
            "skills_gained": [],
            "weak_areas": [],
            "improvements": [],
            "goals": [],
        }

        # 识别薄弱领域
        if area_stats:
            for area, stats in area_stats.items():
                if stats.get("success_rate", 1.0) < 0.5:
                    parsed["weak_areas"].append(area)

        # 从文本中提取信息（简化实现）
        lines = report.split("\n")
        current_section = ""
        for line in lines:
            line = line.strip()
            if "不足" in line or "薄弱" in line:
                current_section = "weak"
            elif "进步" in line or "掌握" in line:
                current_section = "skills"
            elif "改进" in line or "计划" in line:
                current_section = "improve"
            elif "目标" in line:
                current_section = "goals"
            elif line.startswith("- ") or line.startswith("* "):
                content = line[2:].strip()
                if current_section == "weak" and content:
                    parsed["weak_areas"].append(content[:100])
                elif current_section == "skills" and content:
                    parsed["skills_gained"].append(content[:100])
                elif current_section == "improve" and content:
                    parsed["improvements"].append(content[:100])
                elif current_section == "goals" and content:
                    parsed["goals"].append(content[:100])

        # 去重
        parsed["weak_areas"] = list(set(parsed["weak_areas"]))[:5]
        parsed["skills_gained"] = list(set(parsed["skills_gained"]))[:10]

        return parsed
