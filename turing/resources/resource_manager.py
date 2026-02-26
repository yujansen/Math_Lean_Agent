"""
资源管理器 — 动态监测 CPU / RAM / GPU / 磁盘 / 网络。

根据可用资源将系统划分为 HIGH / MEDIUM / LOW 三级，
并据此决定智能体并发数、Lean 编译并行度和 RAG 检索量等策略。
支持 Apple Silicon (M1/M2/M3/M4) 共享内存 GPU 检测。
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import psutil
from loguru import logger

from turing.config import ResourcesConfig, get_config

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPUtil = None
    GPU_AVAILABLE = False


class ResourceLevel(str, Enum):
    HIGH = "high"       # GPU>16GB, RAM>32GB
    MEDIUM = "medium"   # GPU 8-16GB, RAM 16-32GB
    LOW = "low"         # GPU<8GB, RAM<16GB


@dataclass
class ResourceSnapshot:
    """资源快照。"""
    timestamp: float
    cpu_count: int
    cpu_percent: float
    ram_total_gb: float
    ram_free_gb: float
    ram_percent: float
    disk_free_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_memory_total_gb: float
    gpu_memory_free_gb: float
    gpu_utilization: float
    network_available: bool
    level: ResourceLevel
    active_agents: int


class ResourceManager:
    """
    计算资源动态管理器。

    - 检测 GPU/CPU/RAM/磁盘/网络
    - 根据资源水平决定策略
    - 管理活跃智能体数量
    """

    def __init__(self, config: Optional[ResourcesConfig] = None):
        self.config = config or get_config().resources
        self._active_agents: int = 0
        self._last_snapshot: Optional[ResourceSnapshot] = None

    def assess(self) -> ResourceSnapshot:
        """检测当前可用资源并返回快照。"""
        cpu_count = os.cpu_count() or 1
        cpu_percent = psutil.cpu_percent(interval=0.5)

        mem = psutil.virtual_memory()
        ram_total = mem.total / (1024 ** 3)
        ram_free = mem.available / (1024 ** 3)

        disk = shutil.disk_usage(".")
        disk_free = disk.free / (1024 ** 3)

        # GPU 信息
        gpu_available = False
        gpu_name = "N/A"
        gpu_mem_total = 0.0
        gpu_mem_free = 0.0
        gpu_util = 0.0

        if GPU_AVAILABLE and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_available = True
                    gpu_name = gpu.name
                    gpu_mem_total = gpu.memoryTotal / 1024  # MB → GB
                    gpu_mem_free = gpu.memoryFree / 1024
                    gpu_util = gpu.load * 100
            except Exception:
                pass

        # macOS Metal GPU 检测 (Apple Silicon)
        if not gpu_available and platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True, text=True, timeout=5,
                )
                if "Apple" in result.stdout or "M1" in result.stdout or "M2" in result.stdout or "M3" in result.stdout or "M4" in result.stdout:
                    gpu_available = True
                    # 提取芯片名称
                    for line in result.stdout.splitlines():
                        if "Chipset Model" in line or "Chip" in line:
                            gpu_name = line.split(":")[-1].strip()
                            break
                    # Apple Silicon 共享内存
                    gpu_mem_total = ram_total
                    gpu_mem_free = ram_free
            except Exception:
                pass

        # 网络检测
        network = self._check_network()

        # 资源等级评估
        level = self._assess_level(gpu_mem_free, ram_free)

        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            ram_total_gb=round(ram_total, 2),
            ram_free_gb=round(ram_free, 2),
            ram_percent=mem.percent,
            disk_free_gb=round(disk_free, 2),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_total_gb=round(gpu_mem_total, 2),
            gpu_memory_free_gb=round(gpu_mem_free, 2),
            gpu_utilization=round(gpu_util, 1),
            network_available=network,
            level=level,
            active_agents=self._active_agents,
        )

        self._last_snapshot = snapshot
        return snapshot

    def _assess_level(self, gpu_free_gb: float, ram_free_gb: float) -> ResourceLevel:
        """评估资源等级。"""
        if gpu_free_gb >= self.config.gpu_high and ram_free_gb >= self.config.ram_high:
            return ResourceLevel.HIGH
        if gpu_free_gb >= self.config.gpu_medium and ram_free_gb >= self.config.ram_medium:
            return ResourceLevel.MEDIUM
        return ResourceLevel.LOW

    @staticmethod
    def _check_network() -> bool:
        """检查网络可用性（尝试连接 DNS 8.8.8.8）。"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    # ------------------------------------------------------------------
    #  资源策略
    # ------------------------------------------------------------------

    def get_strategy(self, snapshot: Optional[ResourceSnapshot] = None) -> dict:
        """根据资源水平返回运行策略。"""
        if snapshot is None:
            snapshot = self._last_snapshot or self.assess()

        if snapshot.level == ResourceLevel.HIGH:
            return {
                "max_concurrent_agents": self.config.gpu_high // 4,  # ~4 agents
                "lean_parallel_compiles": 3,
                "rag_top_k": 10,
                "batch_size": "large",
                "quantization": False,
                "working_memory_max": 100,
                "description": "资源充足：可并行运行多个子智能体和 Lean 编译任务",
            }
        elif snapshot.level == ResourceLevel.MEDIUM:
            return {
                "max_concurrent_agents": 2,
                "lean_parallel_compiles": 1,
                "rag_top_k": 5,
                "batch_size": "medium",
                "quantization": False,
                "working_memory_max": 50,
                "description": "资源中等：串行运行子智能体，Lean 编译队列化",
            }
        else:
            return {
                "max_concurrent_agents": 1,
                "lean_parallel_compiles": 1,
                "rag_top_k": 3,
                "batch_size": "small",
                "quantization": True,
                "working_memory_max": 30,
                "description": "资源紧张：仅保留主智能体，序列化处理",
            }

    def can_spawn_agent(self) -> bool:
        """判断是否可以新增一个智能体。"""
        snapshot = self._last_snapshot or self.assess()
        strategy = self.get_strategy(snapshot)
        return self._active_agents < strategy["max_concurrent_agents"]

    def register_agent(self):
        """注册一个新的活跃智能体。"""
        self._active_agents += 1

    def unregister_agent(self):
        """注销一个智能体。"""
        self._active_agents = max(0, self._active_agents - 1)

    @property
    def active_agent_count(self) -> int:
        return self._active_agents

    # ------------------------------------------------------------------
    #  格式化输出
    # ------------------------------------------------------------------

    def format_report(self, snapshot: Optional[ResourceSnapshot] = None) -> str:
        """生成资源报告。"""
        s = snapshot or self._last_snapshot or self.assess()
        strategy = self.get_strategy(s)

        lines = [
            f"CPU: {s.cpu_count} 核心 ({s.cpu_percent}% 使用率)",
            f"RAM: {s.ram_free_gb:.1f}GB / {s.ram_total_gb:.1f}GB ({s.ram_percent}% 使用)",
            f"磁盘: {s.disk_free_gb:.1f}GB 可用",
        ]

        if s.gpu_available:
            lines.append(
                f"GPU: {s.gpu_name} "
                f"({s.gpu_memory_free_gb:.1f}GB / {s.gpu_memory_total_gb:.1f}GB free, "
                f"{s.gpu_utilization}% 负载)"
            )
        else:
            lines.append("GPU: 不可用")

        lines.append(f"网络: {'可用' if s.network_available else '不可用'}")
        lines.append(f"资源等级: {s.level.value}")
        lines.append(f"活跃智能体: {s.active_agents}")
        lines.append(f"策略: {strategy['description']}")

        return "\n".join(lines)
