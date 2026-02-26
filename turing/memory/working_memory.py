"""
工作记忆（Working Memory）— 当前任务会话内的短期记忆。

管理当前推理步骤、证明草稿、临时假设和子任务列表。
当条目数接近阈值时自动压缩旧条目。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from loguru import logger


class StepStatus(str, Enum):
    EXPLORING = "探索中"
    VERIFIED = "已验证"
    ABANDONED = "已放弃"
    PENDING = "待验证"


@dataclass
class ReasoningStep:
    """单个推理/证明步骤。"""
    id: str
    content: str
    status: StepStatus = StepStatus.EXPLORING
    abandon_reason: Optional[str] = None
    lean_code: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "abandon_reason": self.abandon_reason,
            "lean_code": self.lean_code,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def summarize(self) -> str:
        """生成压缩摘要（用于转入长期记忆）。"""
        status_str = f"[{self.status.value}]"
        if self.abandon_reason:
            status_str += f"(原因: {self.abandon_reason})"
        return f"{status_str} {self.content[:200]}"


@dataclass
class SubTask:
    """子任务。"""
    id: str
    description: str
    priority: int = 1          # 1=最高优先级
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"    # pending | in_progress | done | failed
    result: Optional[str] = None


class WorkingMemory:
    """
    会话级工作记忆管理器。

    - 追踪当前推理步骤和子任务
    - 自动压缩旧条目防止溢出
    - 提供结构化的上下文导出
    """

    def __init__(self, max_items: int = 50, compression_threshold: int = 40):
        self.max_items = max_items
        self.compression_threshold = compression_threshold

        # 当前问题描述
        self.current_problem: Optional[str] = None
        self.problem_type: Optional[str] = None  # prove | disprove | conjecture | explore | organize

        # 推理步骤
        self.steps: list[ReasoningStep] = []

        # 子任务
        self.subtasks: list[SubTask] = []

        # 临时假设
        self.hypotheses: list[dict] = []

        # 压缩摘要（来自已压缩的旧步骤）
        self.compressed_summaries: list[str] = []

        # 当前上下文片段（从 RAG 检索注入的）
        self.context_snippets: list[dict] = []

        self._step_counter = 0

    # ------------------------------------------------------------------
    #  问题管理
    # ------------------------------------------------------------------

    def set_problem(self, problem: str, problem_type: str = "prove"):
        """设定当前工作问题。"""
        self.current_problem = problem
        self.problem_type = problem_type
        logger.info(f"工作记忆：设定问题 [{problem_type}] {problem[:100]}...")

    def clear(self):
        """清空工作记忆（开始新任务时调用）。"""
        self.current_problem = None
        self.problem_type = None
        self.steps.clear()
        self.subtasks.clear()
        self.hypotheses.clear()
        self.compressed_summaries.clear()
        self.context_snippets.clear()
        self._step_counter = 0

    # ------------------------------------------------------------------
    #  推理步骤
    # ------------------------------------------------------------------

    def add_step(
        self,
        content: str,
        status: StepStatus = StepStatus.EXPLORING,
        lean_code: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ReasoningStep:
        """添加一个推理步骤。"""
        self._step_counter += 1
        step = ReasoningStep(
            id=f"step_{self._step_counter:04d}",
            content=content,
            status=status,
            lean_code=lean_code,
            metadata=metadata or {},
        )
        self.steps.append(step)

        # 检查是否需要压缩
        if len(self.steps) >= self.compression_threshold:
            self._compress()

        return step

    def update_step(
        self,
        step_id: str,
        status: Optional[StepStatus] = None,
        content: Optional[str] = None,
        lean_code: Optional[str] = None,
        abandon_reason: Optional[str] = None,
    ):
        """更新推理步骤的状态。"""
        for step in self.steps:
            if step.id == step_id:
                if status is not None:
                    step.status = status
                if content is not None:
                    step.content = content
                if lean_code is not None:
                    step.lean_code = lean_code
                if abandon_reason is not None:
                    step.abandon_reason = abandon_reason
                return
        logger.warning(f"步骤 {step_id} 不存在")

    def get_verified_steps(self) -> list[ReasoningStep]:
        """获取已验证的步骤。"""
        return [s for s in self.steps if s.status == StepStatus.VERIFIED]

    def get_active_steps(self) -> list[ReasoningStep]:
        """获取当前活跃（探索中/待验证）的步骤。"""
        return [
            s for s in self.steps
            if s.status in (StepStatus.EXPLORING, StepStatus.PENDING)
        ]

    # ------------------------------------------------------------------
    #  子任务管理
    # ------------------------------------------------------------------

    def add_subtask(
        self,
        description: str,
        priority: int = 1,
        dependencies: Optional[list[str]] = None,
    ) -> SubTask:
        """新增子任务。"""
        task = SubTask(
            id=f"task_{len(self.subtasks) + 1:03d}",
            description=description,
            priority=priority,
            dependencies=dependencies or [],
        )
        self.subtasks.append(task)
        return task

    def update_subtask(self, task_id: str, status: str, result: Optional[str] = None):
        for t in self.subtasks:
            if t.id == task_id:
                t.status = status
                t.result = result
                return

    def get_next_subtask(self) -> Optional[SubTask]:
        """获取下一个可执行的子任务（按优先级，已满足依赖）。"""
        done_ids = {t.id for t in self.subtasks if t.status == "done"}
        pending = [
            t for t in self.subtasks
            if t.status == "pending" and all(d in done_ids for d in t.dependencies)
        ]
        if not pending:
            return None
        return sorted(pending, key=lambda t: t.priority)[0]

    # ------------------------------------------------------------------
    #  假设管理
    # ------------------------------------------------------------------

    def add_hypothesis(self, statement: str, confidence: float = 0.5) -> dict:
        hyp = {
            "id": f"hyp_{len(self.hypotheses) + 1:03d}",
            "statement": statement,
            "confidence": confidence,
            "status": "active",
        }
        self.hypotheses.append(hyp)
        return hyp

    # ------------------------------------------------------------------
    #  上下文注入
    # ------------------------------------------------------------------

    def inject_context(self, snippets: list[dict]):
        """从 RAG 检索结果注入上下文。"""
        self.context_snippets.extend(snippets)

    # ------------------------------------------------------------------
    #  压缩与导出
    # ------------------------------------------------------------------

    def _compress(self):
        """压缩旧的已完成步骤为摘要。"""
        # 保留最近10个步骤和所有活跃步骤
        active = [s for s in self.steps if s.status in (StepStatus.EXPLORING, StepStatus.PENDING)]
        completed = [s for s in self.steps if s.status not in (StepStatus.EXPLORING, StepStatus.PENDING)]

        if len(completed) <= 10:
            return

        # 将较旧的完成步骤压缩为摘要
        to_compress = completed[:-5]
        to_keep = completed[-5:]

        summaries = [s.summarize() for s in to_compress]
        batch_summary = f"[批量压缩 {len(to_compress)} 个步骤]\n" + "\n".join(summaries)
        self.compressed_summaries.append(batch_summary)

        self.steps = to_keep + active
        logger.info(f"工作记忆：压缩了 {len(to_compress)} 个步骤")

    def export_context(self) -> str:
        """导出当前完整工作记忆为格式化文本（注入 LLM 上下文）。"""
        parts = []

        if self.current_problem:
            parts.append(f"## 当前问题\n类型: {self.problem_type}\n{self.current_problem}")

        if self.compressed_summaries:
            parts.append("## 历史摘要\n" + "\n---\n".join(self.compressed_summaries[-3:]))

        if self.context_snippets:
            parts.append("## 检索到的相关知识")
            for snip in self.context_snippets[-5:]:
                parts.append(f"- [{snip.get('type', '?')}] {snip.get('content', '')[:200]}")

        if self.steps:
            parts.append("## 推理步骤")
            for s in self.steps:
                status = f"[{s.status.value}]"
                parts.append(f"  {s.id} {status} {s.content[:300]}")
                if s.lean_code:
                    parts.append(f"    ```lean\n    {s.lean_code[:500]}\n    ```")

        if self.subtasks:
            parts.append("## 子任务")
            for t in self.subtasks:
                parts.append(f"  [{t.status}] {t.id} (p={t.priority}) {t.description}")

        if self.hypotheses:
            parts.append("## 假设")
            for h in self.hypotheses:
                parts.append(
                    f"  [{h['status']}] {h['id']} (置信度={h['confidence']}) {h['statement']}"
                )

        return "\n\n".join(parts)

    def get_stats(self) -> dict:
        """返回工作记忆统计信息。"""
        return {
            "total_steps": len(self.steps),
            "verified_steps": len(self.get_verified_steps()),
            "active_steps": len(self.get_active_steps()),
            "subtasks_total": len(self.subtasks),
            "subtasks_done": sum(1 for t in self.subtasks if t.status == "done"),
            "hypotheses": len(self.hypotheses),
            "compressed_batches": len(self.compressed_summaries),
            "context_snippets": len(self.context_snippets),
        }
