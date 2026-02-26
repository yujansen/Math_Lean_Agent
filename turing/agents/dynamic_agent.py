"""
动态智能体 — 由 Turing 在运行时根据需求生成的通用智能体。

当 Turing 需要创建一个不在预定义角色中的子智能体时使用此类。
它通过 system prompt 定义行为。
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.llm.llm_client import LLMClient
from turing.resources.resource_manager import ResourceManager


class DynamicAgent(BaseAgent):
    """
    动态生成的通用智能体。

    行为完全由 system_prompt 驱动，可以处理各种自定义任务。
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        **kwargs,
    ):
        super().__init__(agent_config, llm_client, resource_manager)

    async def _execute(self, task: str, **kwargs) -> Any:
        """执行任务。通过 LLM 交互完成指定任务。"""

        logger.info(f"[{self.name}] 开始执行: {task[:100]}...")

        # 第一轮：理解任务
        plan = await self.think(
            f"你需要完成以下任务，请先制定一个详细的执行计划：\n\n{task}"
        )

        results = [f"## 执行计划\n{plan}"]

        # 迭代执行
        step = 0
        while not self.should_stop():
            step += 1
            prompt = (
                f"请执行计划的第{step}步。\n"
                f"之前的进展:\n{results[-1][:1000]}\n\n"
                f"如果所有步骤已完成，请回复 '[DONE]' 并给出最终结果。"
            )
            response = await self.think(prompt)

            results.append(f"## 步骤 {step}\n{response}")

            if "[DONE]" in response:
                break

        final = "\n\n".join(results)
        logger.info(f"[{self.name}] 完成，共 {step} 步")
        return final
