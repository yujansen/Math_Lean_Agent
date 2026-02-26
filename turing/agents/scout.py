"""
Scout 智能体 — 从网络题库抓取数学问题。

当 Turing 进入"训练模式"时被触发，主动寻找练习题。
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.llm.llm_client import LLMClient
from turing.resources.resource_manager import ResourceManager
from turing.web.problem_scraper import ProblemScraper


SCOUT_SYSTEM_PROMPT = """你是 Scout，Turing 数学研究团队中的问题搜寻者。

你的职责是：
1. 从各种来源搜集适合训练的数学问题
2. 评估问题的难度等级 (1-10)
3. 分类问题所属的数学领域
4. 判断问题是否适合形式化验证
5. 生成适合 Turing 能力水平的训练问题

问题来源（按优先级）：
1. Mathlib Issues / PRs
2. IMO / 各国数学竞赛
3. Project Euler
4. Mathematics Stack Exchange
5. 自主生成

输出格式：
对于每个问题：
- 题目: <描述>
- 领域: <数学领域>
- 难度: <1-10>
- 来源: <出处>
- 可形式化: <是/否>
- 建议策略: <初步解题思路>"""


class ScoutAgent(BaseAgent):
    """问题搜寻智能体。"""

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        problem_scraper: Optional[ProblemScraper] = None,
        **kwargs,
    ):
        config = agent_config or AgentConfig(
            agent_name="Scout",
            system_prompt=SCOUT_SYSTEM_PROMPT,
        )
        if not config.system_prompt:
            config.system_prompt = SCOUT_SYSTEM_PROMPT

        super().__init__(config, llm_client, resource_manager)
        self.scraper = problem_scraper or ProblemScraper()

    async def _execute(self, task: str, **kwargs) -> Any:
        """
        搜寻和生成训练问题。

        Args:
            task: 搜寻指令
            kwargs:
                - skill_level: 当前技能等级 (1-10)
                - weak_areas: 薄弱领域列表
                - count: 需要多少个问题
                - area: 指定数学领域
        """
        skill_level = kwargs.get("skill_level", 3)
        weak_areas = kwargs.get("weak_areas", [])
        count = kwargs.get("count", 5)
        area = kwargs.get("area", "")

        logger.info(
            f"[Scout] 搜寻问题: level={skill_level}, count={count}, "
            f"weak={weak_areas}, area={area}"
        )

        problems = []

        # 1. 尝试从网络抓取
        try:
            web_problems = await self.scraper.fetch_problems(
                area=area, difficulty=skill_level, count=count
            )
            problems.extend(web_problems)
        except Exception as e:
            logger.warning(f"[Scout] 网络抓取失败: {e}")

        # 2. 如果网络抓取不够，使用 LLM 生成
        remaining = count - len(problems)
        if remaining > 0:
            generated = await self._generate_problems(
                skill_level, weak_areas, remaining, area
            )
            problems.extend(generated)

        logger.info(f"[Scout] 搜寻完成: {len(problems)} 个问题")
        return {"problems": problems, "count": len(problems)}

    async def _generate_problems(
        self,
        skill_level: int,
        weak_areas: list[str],
        count: int,
        area: str = "",
    ) -> list[dict]:
        """使用 LLM 生成训练问题。"""

        area_hint = ""
        if area:
            area_hint = f"领域限制: {area}\n"
        elif weak_areas:
            area_hint = f"请重点关注薄弱领域: {', '.join(weak_areas)}\n"

        prompt = f"""请生成 {count} 个适合训练的数学问题。

要求:
- 难度等级: 约 {skill_level}/10（适度挑战）
- 问题应该可以被形式化为 Lean 4 证明
{area_hint}
对于每个问题，给出 JSON 格式:
```json
[
  {{
    "title": "问题简短标题",
    "statement": "完整的数学命题陈述",
    "area": "数学领域",
    "difficulty": <1-10>,
    "hints": "可能的解题提示",
    "source": "self_generated",
    "formalizable": true
  }}
]
```

请确保问题：
1. 陈述清晰、无歧义
2. 难度适中、循序渐进
3. 涵盖不同的证明技巧"""

        response = await self.think(prompt)

        # 解析 JSON
        problems = self._parse_problems(response)
        return problems

    @staticmethod
    def _parse_problems(text: str) -> list[dict]:
        """从 LLM 响应中解析问题列表。"""
        import json

        # 尝试提取 JSON 块
        json_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接解析整段
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # 尝试找到 [ ... ] 部分
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("[Scout] 无法解析 LLM 返回的问题，返回空列表")
        return []

    async def select_next_problem(
        self,
        skill_level: int,
        weak_areas: list[str],
        problem_pool: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """
        根据训练策略选择下一个问题。

        策略:
        - 70% 选择略高于当前水平的问题（成长区）
        - 20% 针对薄弱领域训练
        - 10% 随机探索
        """
        r = random.random()

        if problem_pool:
            if r < 0.7:
                # 成长区：difficulty ≈ skill_level + 1
                candidates = [
                    p for p in problem_pool
                    if abs(p.get("difficulty", 5) - (skill_level + 1)) <= 1
                ]
            elif r < 0.9 and weak_areas:
                # 薄弱领域
                candidates = [
                    p for p in problem_pool
                    if p.get("area", "") in weak_areas
                ]
            else:
                # 随机
                candidates = problem_pool

            if candidates:
                return random.choice(candidates)

        # 如果没有候选，生成一个新问题
        if r < 0.9 and weak_areas:
            area = random.choice(weak_areas)
        else:
            area = ""

        problems = await self._generate_problems(
            skill_level, weak_areas, 1, area
        )
        return problems[0] if problems else None
