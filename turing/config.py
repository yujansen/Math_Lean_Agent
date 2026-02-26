"""
配置管理模块 — 加载和管理 Turing 系统的全局配置。

使用 ``dataclass`` 构建层次化配置结构（LLM / Lean / Memory / …），
支持从 YAML 文件覆盖默认值，并以全局单例方式提供给各子模块。

使用方法::

    from turing.config import get_config
    cfg = get_config()           # 自动加载 config.yaml
    print(cfg.llm.model)         # -> "qwen3-coder:30b"
    print(cfg.lean.compile_timeout)  # -> 300
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class LLMConfig:
    """LLM 服务配置（支持 Ollama 和 OpenAI 兼容 API）。"""
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3-30b:coder"
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout: int = 120
    light_model: str = "qwen3-30b:coder"
    external_llm: dict = field(default_factory=lambda: {
        "enabled": False,
        "provider": "openai",
        "api_key": "",
        "model": "gpt-4",
        "base_url": "https://api.openai.com/v1",
    })


@dataclass
class LeanConfig:
    """Lean 4 编译器与 Mathlib 配置。

    ``compile_timeout`` 在 YAML 中建议设为 300，因为 Mathlib 导入开销较大。
    """
    executable: str = "lean"
    lake_executable: str = "lake"
    project_dir: str = "./lean_workspace"
    mathlib_available: bool = True
    compile_timeout: int = 60
    max_retries: int = 5


@dataclass
class WorkingMemoryConfig:
    max_items: int = 50
    compression_threshold: int = 40


@dataclass
class LongTermMemoryConfig:
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "turing_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.92
    default_top_k: int = 5


@dataclass
class PersistentMemoryConfig:
    db_path: str = "./data/turing_persistent.db"
    log_dir: str = "./data/logs"


@dataclass
class MemoryConfig:
    """三层记忆系统配置：工作记忆 / 长期记忆 (ChromaDB) / 持久记忆 (SQLite)。"""
    working: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    long_term: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)
    persistent: PersistentMemoryConfig = field(default_factory=PersistentMemoryConfig)


@dataclass
class AgentsConfig:
    max_concurrent: int = 3
    default_max_iterations: int = 50
    default_timeout_minutes: int = 30
    default_max_tokens_per_call: int = 8192


@dataclass
class ResourcesConfig:
    gpu_high: int = 16
    gpu_medium: int = 8
    ram_high: int = 32
    ram_medium: int = 16
    check_interval: int = 30


@dataclass
class EvolutionConfig:
    reflection_task_interval: int = 20
    reflection_time_interval: int = 24
    priority_increment: float = 0.1
    priority_decrement: float = 0.1
    priority_max: float = 1.0
    priority_min: float = 0.0


@dataclass
class TrainingConfig:
    growth_zone_ratio: float = 0.7
    weakness_ratio: float = 0.2
    exploration_ratio: float = 0.1
    initial_skill_level: int = 1


@dataclass
class WebConfig:
    enabled: bool = True
    timeout: int = 30
    user_agent: str = "Turing-Math-Agent/1.0"
    sources: list = field(default_factory=list)


@dataclass
class SystemConfig:
    name: str = "Turing"
    version: str = "1.0.0"
    log_level: str = "INFO"
    data_dir: str = "./data"
    snapshot_dir: str = "./data/snapshots"
    max_snapshots: int = 5
    backup_interval_hours: int = 6


@dataclass
class TuringConfig:
    """Turing 系统的顶层配置。"""
    system: SystemConfig = field(default_factory=SystemConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    lean: LeanConfig = field(default_factory=LeanConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TuringConfig":
        """从 YAML 文件加载配置。"""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config = cls()
        _update_dataclass(config.system, raw.get("system", {}))
        _update_dataclass(config.llm, raw.get("llm", {}))
        _update_dataclass(config.lean, raw.get("lean", {}))
        _update_dataclass(config.agents, raw.get("agents", {}))
        _update_dataclass(config.resources, raw.get("resources", {}))
        _update_dataclass(config.evolution, raw.get("evolution", {}))
        _update_dataclass(config.training, raw.get("training", {}))
        _update_dataclass(config.web, raw.get("web", {}))

        mem = raw.get("memory", {})
        _update_dataclass(config.memory.working, mem.get("working", {}))
        _update_dataclass(config.memory.long_term, mem.get("long_term", {}))
        _update_dataclass(config.memory.persistent, mem.get("persistent", {}))

        # 确保数据目录存在
        for d in [
            config.system.data_dir,
            config.system.snapshot_dir,
            config.memory.long_term.chroma_persist_dir,
            config.memory.persistent.log_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        return config

    def ensure_dirs(self):
        """确保所有必要目录存在。"""
        for d in [
            self.system.data_dir,
            self.system.snapshot_dir,
            self.memory.long_term.chroma_persist_dir,
            self.memory.persistent.log_dir,
        ]:
            os.makedirs(d, exist_ok=True)


def _update_dataclass(obj: Any, data: dict) -> None:
    """用字典数据就地更新 dataclass 实例的匹配字段。

    只设置 ``obj`` 中已存在的属性，忽略 YAML 中多余的键。
    """
    if not data:
        return
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


# 全局配置单例
_config: Optional[TuringConfig] = None


def get_config(config_path: str | Path | None = None) -> TuringConfig:
    """获取全局配置（懒加载）。"""
    global _config
    if _config is None:
        path = config_path or os.environ.get("TURING_CONFIG", "config.yaml")
        _config = TuringConfig.from_yaml(path)
    return _config


def reset_config():
    """重置全局配置（用于测试）。"""
    global _config
    _config = None
