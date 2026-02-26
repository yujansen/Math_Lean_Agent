"""
Architect 智能体 — 评估并优化整个智能体系统的架构和 Prompt。

常规在每个反思周期或演化阶段触发。可以提议新智能体、合并/删除子智能体、优化资源分配。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.llm.llm_client import LLMClient
from turing.memory.persistent_memory import PersistentMemory
from turing.resources.resource_manager import ResourceManager


ARCHITECT_SYSTEM_PROMPT = """你是 Architect，Turing 数学研究团队中的系统架构师。

你的职责是：
1. 评估整个智能体系统的运行效率和效果
2. 分析各子智能体的表现，提出 prompt 优化建议
3. 决定是否需要新增、合并或删除子智能体
4. 优化资源分配策略
5. 设计系统架构的改进方案

评估维度：
- 成功率趋势：各领域的证明成功率是否在改善
- 效率：平均完成时间是否在减少
- 资源利用：是否有资源浪费或瓶颈
- 知识增长：知识库的增长率和质量
- 智能体协同：各智能体之间的协作是否顺畅

输出格式：
## 系统评估报告
- 整体健康度: [1-10]
- 各模块评分: {...}
- 发现的问题: [列表]
- 优化建议: [列表，按优先级排序]
- 建议的 prompt 修改: [目标智能体 → 修改建议]"""


class ArchitectAgent(BaseAgent):
    """系统架构师智能体。"""

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        persistent_memory: Optional[PersistentMemory] = None,
        **kwargs,
    ):
        config = agent_config or AgentConfig(
            agent_name="Architect",
            system_prompt=ARCHITECT_SYSTEM_PROMPT,
        )
        if not config.system_prompt:
            config.system_prompt = ARCHITECT_SYSTEM_PROMPT

        super().__init__(config, llm_client, resource_manager)
        self.persistent_memory = persistent_memory

    async def _execute(self, task: str, **kwargs) -> Any:
        """
        执行系统评估和优化。

        Args:
            task: 评估任务描述
            kwargs:
                - reflection_data: 反思报告数据
                - agent_reports: 各智能体的状态报告
                - system_stats: 系统统计信息
        """
        reflection_data = kwargs.get("reflection_data", {})
        agent_reports = kwargs.get("agent_reports", [])
        system_stats = kwargs.get("system_stats", {})

        logger.info("[Architect] 开始系统评估...")

        # 收集综合数据
        data = self._collect_system_data(
            reflection_data, agent_reports, system_stats
        )

        # 第一步：整体评估
        assessment = await self.think(
            f"""请对 Turing 系统的当前状态进行全面评估。

系统数据:
{json.dumps(data, ensure_ascii=False, indent=2)}

请按照评估规范给出详细报告。"""
        )

        # 第二步：Prompt 优化建议
        prompt_suggestions = await self.think(
            f"""基于你的评估，请给出具体的 prompt 优化建议。

评估结果:
{assessment[:2000]}

对于每个需要优化的智能体：
1. 指出当前 prompt 的不足
2. 给出修改后的 prompt 片段
3. 预期的改进效果

注意：优化应该基于具体的数据和失败案例。"""
        )

        # 第三步：架构调整建议
        arch_suggestions = await self.think(
            f"""请提出架构层面的改进建议：

1. 是否需要新增子智能体？如果是，描述其职责和 prompt
2. 是否有子智能体应该合并或删除？
3. 资源分配策略是否需要调整？
4. 工作流程是否有可以简化的环节？
5. 记忆系统是否需要优化？"""
        )

        result = {
            "assessment": assessment,
            "prompt_suggestions": prompt_suggestions,
            "architecture_suggestions": arch_suggestions,
            "data_analyzed": data,
        }

        # 保存到持久记忆
        if self.persistent_memory:
            self.persistent_memory.record_agent_modification(
                agent_id="system",
                modification_type="architecture_review",
                reason="定期系统评估",
                diff=json.dumps({
                    "assessment_summary": assessment[:500],
                    "suggestions_count": len(prompt_suggestions.split("\n")),
                }, ensure_ascii=False),
                validation_result="neutral",
            )

        return result

    def _collect_system_data(
        self,
        reflection_data: dict,
        agent_reports: list,
        system_stats: dict,
    ) -> dict:
        """汇总系统数据用于评估。"""
        data = {
            "reflection": reflection_data,
            "agents": agent_reports,
            "stats": system_stats,
        }

        if self.persistent_memory:
            data["persistent_stats"] = self.persistent_memory.get_comprehensive_stats()
            data["failure_patterns"] = self.persistent_memory.get_failure_patterns(10)
            latest_reflection = self.persistent_memory.get_latest_reflection()
            if latest_reflection:
                data["last_reflection"] = latest_reflection

        if self.resource_manager:
            snapshot = self.resource_manager.assess()
            data["resources"] = {
                "level": snapshot.level.value,
                "gpu": snapshot.gpu_name,
                "ram_free_gb": snapshot.ram_free_gb,
                "active_agents": snapshot.active_agents,
            }

        return data

    async def propose_new_agent(self, need_description: str) -> dict:
        """
        根据需求描述，提议一个新的子智能体。

        Returns:
            AgentConfig 兼容的 dict
        """
        prompt = f"""需要设计一个新的子智能体来满足以下需求：

{need_description}

请给出完整的智能体规格（JSON格式），包含：
- agent_name: 名称
- system_prompt: 完整的 system prompt
- tools: 需要的工具列表
- lifecycle: task_scoped 或 persistent
- resource_budget: 资源预算

```json
{{
  "agent_name": "...",
  "agent_type": "...",
  "system_prompt": "...",
  "tools": [...],
  "lifecycle": "...",
  "resource_budget": {{
    "max_tokens_per_call": 8192,
    "max_iterations": 50,
    "timeout_minutes": 30
  }}
}}
```"""

        response = await self.think(prompt)

        json_match = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        return {"agent_name": "CustomAgent", "system_prompt": response}
