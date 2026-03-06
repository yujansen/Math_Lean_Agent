"""
智能体工厂 — 动态创建和管理子智能体。

Turing 通过此工厂按需生成 Prover/Explorer/Critic 等子智能体。
支持版本控制、A/B 测试和回滚。
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional, Type

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.config import get_config
from turing.llm.llm_client import LLMClient
from turing.memory.persistent_memory import PersistentMemory
from turing.resources.resource_manager import ResourceManager


class AgentFactory:
    """
    智能体工厂，负责：
    - 动态创建子智能体
    - 管理活跃智能体池
    - 版本控制与回滚
    - A/B 测试
    """

    # 已注册的智能体类型
    _registry: dict[str, Type[BaseAgent]] = {}

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        persistent_memory: Optional[PersistentMemory] = None,
    ):
        self.llm = llm_client or LLMClient()
        self.resource_manager = resource_manager or ResourceManager()
        self.persistent_memory = persistent_memory
        self._active_agents: dict[str, BaseAgent] = {}
        self._agent_history: list[dict] = []

    # ------------------------------------------------------------------
    #  注册表
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]):
        """注册一种智能体类型。"""
        cls._registry[name] = agent_class
        logger.debug(f"注册智能体类型: {name} → {agent_class.__name__}")

    @classmethod
    def get_registered_types(cls) -> list[str]:
        return list(cls._registry.keys())

    # ------------------------------------------------------------------
    #  创建智能体
    # ------------------------------------------------------------------

    def create(
        self,
        agent_type: str,
        agent_config: Optional[AgentConfig] = None,
        **kwargs,
    ) -> Optional[BaseAgent]:
        """
        创建一个子智能体。

        Args:
            agent_type: 智能体类型（必须已注册）
            agent_config: 配置（可选，使用默认配置）
            **kwargs: 传递给智能体构造函数的额外参数

        Returns:
            BaseAgent 实例，或 None（资源不足时）
        """
        if not self.resource_manager.can_spawn_agent():
            logger.warning(
                f"资源不足，无法创建新智能体 {agent_type}。"
                f"当前活跃: {self.resource_manager.active_agent_count}"
            )
            return None

        if agent_type not in self._registry:
            logger.error(f"未注册的智能体类型: {agent_type}")
            return None

        agent_class = self._registry[agent_type]

        if agent_config is None:
            agent_config = AgentConfig(
                agent_id=f"agent_{int(time.time())}_{agent_type}",
                agent_name=agent_type.capitalize(),
                base_model=get_config().llm.model,
            )

        agent = agent_class(
            agent_config=agent_config,
            llm_client=self.llm,
            resource_manager=self.resource_manager,
            **kwargs,
        )

        self._active_agents[agent.agent_id] = agent
        self.resource_manager.register_agent()

        self._agent_history.append({
            "agent_id": agent.agent_id,
            "type": agent_type,
            "created_at": time.time(),
            "config": {
                "system_prompt": agent_config.system_prompt[:200],
                "lifecycle": agent_config.lifecycle,
            },
        })

        logger.info(f"创建智能体: [{agent_type}] {agent.agent_id}")
        return agent

    def create_from_spec(self, spec: dict) -> Optional[BaseAgent]:
        """
        从 JSON 规格创建智能体（用于 Turing 动态生成）。

        spec 格式参见 turing_system_prompt.md 第4.1节。
        """
        agent_name = spec.get("agent_name", "CustomAgent")
        agent_type = spec.get("agent_type", "custom")

        config = AgentConfig(
            agent_id=spec.get("agent_id", f"agent_{int(time.time())}_{agent_type}"),
            agent_name=agent_name,
            base_model=spec.get("base_model", get_config().llm.model),
            system_prompt=spec.get("system_prompt", ""),
            tools=spec.get("tools", []),
            memory_access=spec.get("memory_access", {"read": [], "write": []}),
            lifecycle=spec.get("lifecycle", "task_scoped"),
            resource_budget=spec.get("resource_budget", {
                "max_tokens_per_call": 8192,
                "max_iterations": 50,
                "timeout_minutes": 30,
            }),
            report_to=spec.get("report_to", "turing_main"),
        )

        # 如果类型未注册，记录警告并返回 None
        if agent_type not in self._registry:
            logger.warning(f"未注册的智能体类型: {agent_type}，无法从 spec 创建")
            return None

        return self.create(agent_type, config)

    # ------------------------------------------------------------------
    #  管理
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self._active_agents.get(agent_id)

    def list_active(self) -> list[dict]:
        return [agent.get_status_report() for agent in self._active_agents.values()]

    async def destroy(self, agent_id: str):
        """销毁一个智能体。"""
        agent = self._active_agents.pop(agent_id, None)
        if agent:
            await agent.stop()
            self.resource_manager.unregister_agent()
            logger.info(f"销毁智能体: {agent_id}")

    async def destroy_all(self):
        """销毁所有智能体。"""
        for agent_id in list(self._active_agents.keys()):
            await self.destroy(agent_id)

    # ------------------------------------------------------------------
    #  Prompt 修改与版本控制
    # ------------------------------------------------------------------

    def modify_agent_prompt(
        self,
        agent_id: str,
        new_prompt: str,
        reason: str = "",
    ) -> bool:
        """
        修改智能体的 system prompt，保留版本历史。
        """
        agent = self._active_agents.get(agent_id)
        if not agent:
            logger.error(f"智能体不存在: {agent_id}")
            return False

        old_prompt = agent.config.system_prompt

        # 保存旧版本到持久记忆
        if self.persistent_memory:
            self.persistent_memory.save_prompt_version(
                agent_id=agent_id,
                prompt_content=old_prompt,
                change_reason=reason,
            )

        # 更新 prompt
        agent.config.system_prompt = new_prompt
        agent.reset_conversation()

        # 记录修改
        if self.persistent_memory:
            self.persistent_memory.record_agent_modification(
                agent_id=agent_id,
                modification_type="prompt_update",
                reason=reason,
                diff=f"--- old\n{old_prompt[:500]}\n+++ new\n{new_prompt[:500]}",
            )

        logger.info(f"修改智能体 prompt: {agent_id}, 原因: {reason}")
        return True

    def rollback_agent_prompt(self, agent_id: str) -> bool:
        """回滚智能体的 prompt 到上一个版本。"""
        if not self.persistent_memory:
            return False

        old_prompt = self.persistent_memory.rollback_prompt(agent_id)
        if old_prompt is None:
            logger.warning(f"无法回滚: {agent_id} 没有历史版本")
            return False

        agent = self._active_agents.get(agent_id)
        if agent:
            agent.config.system_prompt = old_prompt
            agent.reset_conversation()
            logger.info(f"回滚智能体 prompt: {agent_id}")
        return True
