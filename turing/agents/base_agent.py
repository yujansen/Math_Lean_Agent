"""
基础智能体类 — 所有 Turing 子智能体的抽象父类。

定义统一的生命周期（initialize → run → stop）、LLM 通信协议和
资源控制框架。所有具体智能体（Prover、Critic、Explorer …）均需
继承 :class:`BaseAgent` 并实现 :meth:`_execute`。
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from turing.llm.llm_client import ChatMessage, LLMClient, LLMResponse
from turing.memory.working_memory import WorkingMemory
from turing.resources.resource_manager import ResourceManager


class AgentLifecycle(str, Enum):
    TASK_SCOPED = "task_scoped"     # 任务结束即销毁
    PERSISTENT = "persistent"       # 持久运行


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentConfig:
    """智能体配置规格。"""
    agent_id: str = ""
    agent_name: str = ""
    base_model: str = "qwen3-30b:coder"
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    memory_access: dict = field(default_factory=lambda: {"read": [], "write": []})
    lifecycle: str = "task_scoped"
    resource_budget: dict = field(default_factory=lambda: {
        "max_tokens_per_call": 8192,
        "max_iterations": 50,
        "timeout_minutes": 30,
    })
    report_to: str = "turing_main"


@dataclass
class AgentResult:
    """智能体执行结果。"""
    agent_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    iterations: int = 0
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    基础智能体抽象类。

    所有 Turing 系统中的智能体都继承此类。
    提供统一的：
    - 生命周期管理（初始化、运行、暂停、停止）
    - LLM 通信接口
    - 工作记忆
    - 结果报告
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
    ):
        self.config = agent_config
        if not self.config.agent_id:
            self.config.agent_id = f"agent_{int(time.time())}_{self.__class__.__name__}"

        self.llm = llm_client or LLMClient()
        self.resource_manager = resource_manager or ResourceManager()
        self.working_memory = WorkingMemory()

        self.status = AgentStatus.IDLE
        self._iteration_count = 0
        self._start_time: Optional[float] = None
        self._stop_event = asyncio.Event()
        self._conversation: list[ChatMessage] = []

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def name(self) -> str:
        return self.config.agent_name or self.__class__.__name__

    # ------------------------------------------------------------------
    #  生命周期
    # ------------------------------------------------------------------

    async def initialize(self):
        """初始化智能体。子类可重写以添加自定义初始化逻辑。"""
        self.status = AgentStatus.IDLE
        logger.info(f"[{self.name}] 初始化完成")

    async def run(self, task: str, **kwargs) -> AgentResult:
        """
        执行任务的主入口。

        Args:
            task: 任务描述
            **kwargs: 额外参数

        Returns:
            AgentResult
        """
        self.status = AgentStatus.RUNNING
        self._start_time = time.time()
        self._iteration_count = 0
        self._stop_event.clear()

        budget = self.config.resource_budget
        max_iter = budget.get("max_iterations", 50)
        timeout = budget.get("timeout_minutes", 30) * 60

        try:
            result = await asyncio.wait_for(
                self._execute(task, **kwargs),
                timeout=timeout,
            )
            self.status = AgentStatus.COMPLETED
            duration = time.time() - self._start_time

            return AgentResult(
                agent_id=self.agent_id,
                success=True,
                result=result,
                iterations=self._iteration_count,
                duration_seconds=duration,
            )

        except asyncio.TimeoutError:
            self.status = AgentStatus.FAILED
            duration = time.time() - self._start_time
            logger.warning(f"[{self.name}] 任务超时 ({timeout}s)")
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                error="任务超时",
                iterations=self._iteration_count,
                duration_seconds=duration,
            )

        except Exception as e:
            self.status = AgentStatus.FAILED
            duration = time.time() - (self._start_time or time.time())
            logger.error(f"[{self.name}] 任务失败: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                iterations=self._iteration_count,
                duration_seconds=duration,
            )

    @abstractmethod
    async def _execute(self, task: str, **kwargs) -> Any:
        """
        子类实现具体的任务执行逻辑。

        Args:
            task: 任务描述
        Returns:
            任意结果数据
        """
        ...

    async def stop(self):
        """请求停止智能体。"""
        self._stop_event.set()
        self.status = AgentStatus.PAUSED
        logger.info(f"[{self.name}] 收到停止请求")

    def should_stop(self) -> bool:
        """检查是否应该停止。"""
        if self._stop_event.is_set():
            return True

        budget = self.config.resource_budget
        max_iter = budget.get("max_iterations", 50)
        if self._iteration_count >= max_iter:
            logger.warning(f"[{self.name}] 达到最大迭代次数 {max_iter}")
            return True

        timeout = budget.get("timeout_minutes", 30) * 60
        if self._start_time and (time.time() - self._start_time > timeout):
            logger.warning(f"[{self.name}] 达到超时限制")
            return True

        return False

    # ------------------------------------------------------------------
    #  LLM 通信
    # ------------------------------------------------------------------

    async def think(self, prompt: str, **kwargs) -> str:
        """
        使用 LLM 进行思考/推理。

        这是所有智能体与 LLM 交互的统一方式。
        """
        self._iteration_count += 1

        # 构建消息
        messages = list(self._conversation)
        messages.append(ChatMessage(role="user", content=prompt))

        response = await self.llm.chat(
            messages,
            system_prompt=self.config.system_prompt,
            model=kwargs.get("model", self.config.base_model),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
        )

        # 更新会话历史
        self._conversation.append(ChatMessage(role="user", content=prompt))
        self._conversation.append(
            ChatMessage(role="assistant", content=response.content)
        )

        # 控制上下文长度
        if len(self._conversation) > 20:
            self._conversation = self._conversation[-14:]

        return response.content

    async def think_with_context(self, prompt: str, context: str = "", **kwargs) -> str:
        """带上下文的思考（注入工作记忆）。"""
        full_prompt = prompt
        if context:
            full_prompt = f"<context>\n{context}\n</context>\n\n{prompt}"
        elif self.working_memory.current_problem:
            wm_context = self.working_memory.export_context()
            full_prompt = f"<working_memory>\n{wm_context}\n</working_memory>\n\n{prompt}"
        return await self.think(full_prompt, **kwargs)

    def reset_conversation(self):
        """重置对话历史。"""
        self._conversation.clear()

    # ------------------------------------------------------------------
    #  工具方法
    # ------------------------------------------------------------------

    def get_status_report(self) -> dict:
        """获取智能体状态报告。"""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "iterations": self._iteration_count,
            "elapsed_seconds": round(elapsed, 1),
            "lifecycle": self.config.lifecycle,
        }

    def to_agent_config(self) -> AgentConfig:
        """导出当前智能体的配置。"""
        return AgentConfig(
            agent_id=self.agent_id,
            agent_name=self.name,
            base_model=self.config.base_model,
            system_prompt=self.config.system_prompt,
            tools=self.config.tools,
            memory_access=self.config.memory_access,
            lifecycle=self.config.lifecycle,
            resource_budget=self.config.resource_budget,
            report_to=self.config.report_to,
        )
