"""
通用工具模块 — 日志配置、辅助函数等跨模块共享工具。

提供所有脚本和模块复用的基础设施，避免在多个入口点重复定义。
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_dir: str | Path = "./data/logs",
    log_prefix: str = "turing",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    配置全局日志系统（loguru）。

    同时输出到 stderr（INFO 级别）和按日期滚动的文件（DEBUG 级别）。

    Args:
        log_dir:    日志文件存放目录，不存在则自动创建。
        log_prefix: 日志文件前缀，生成格式为 ``{prefix}_{date}.log``。
        level:      stderr 输出等级。
        rotation:   日志文件大小上限，超过后自动轮转。
        retention:  历史日志保留时长。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        str(log_dir / f"{log_prefix}_{{time:YYYY-MM-DD}}.log"),
        rotation=rotation,
        retention=retention,
        level="DEBUG",
        encoding="utf-8",
    )
