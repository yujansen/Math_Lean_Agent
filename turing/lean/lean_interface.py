"""
Lean 4 编译器接口 — 编译代码、解析错误、管理 Lean 项目环境。

所有 Lean 编译操作都通过 ``lake env lean`` 执行，以确保 Mathlib
等依赖被正确加载。编译超时默认 300 秒（由 ``config.yaml`` 配置）。

典型用法::

    lean = LeanInterface()
    await lean.initialize()
    result = await lean.compile(\"import Mathlib\\ntheorem t : 1 + 1 = 2 := by norm_num\")
    print(result.success)  # True
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from turing.config import LeanConfig, get_config


@dataclass
class LeanResult:
    """Lean 编译结果。"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    duration_ms: float = 0.0
    code: str = ""

    @property
    def error_summary(self) -> str:
        if not self.errors:
            return ""
        return "\n".join(
            f"行{e.get('line', '?')}: {e.get('message', '')}" for e in self.errors
        )


@dataclass
class LeanError:
    """解析后的 Lean 错误信息。"""
    file: str = ""
    line: int = 0
    column: int = 0
    severity: str = "error"    # error | warning | info
    message: str = ""
    category: str = ""         # syntax | type_mismatch | unknown_tactic | logic | other


class LeanInterface:
    """
    Lean 4 编译器接口。

    - 编译 Lean 代码并返回结构化结果
    - 解析编译错误，分类错误类型
    - 管理 Lean 项目环境
    - 搜索 Mathlib 中的定理
    """

    def __init__(self, config: Optional[LeanConfig] = None):
        self.config = config or get_config().lean
        self._project_dir = Path(self.config.project_dir)
        self._initialized = False

    async def initialize(self):
        """初始化 Lean 项目环境。"""
        self._project_dir.mkdir(parents=True, exist_ok=True)

        # 检查 lean 是否可用
        lean_ok = await self._check_executable(self.config.executable)
        if not lean_ok:
            logger.warning(
                f"Lean 4 不可用 ({self.config.executable})。"
                "请安装 Lean 4: https://leanprover.github.io/lean4/doc/setup.html"
            )

        # 检查 lake 是否可用
        lake_ok = await self._check_executable(self.config.lake_executable)

        # 如果项目目录不存在 lakefile.lean, 初始化项目
        lakefile = self._project_dir / "lakefile.lean"
        if not lakefile.exists() and lake_ok:
            await self._init_lean_project()

        self._initialized = True
        logger.info(
            f"Lean 接口初始化: lean={'OK' if lean_ok else 'N/A'}, "
            f"lake={'OK' if lake_ok else 'N/A'}, "
            f"project={self._project_dir}"
        )

    async def _check_executable(self, cmd: str) -> bool:
        """检查可执行文件是否可用。"""
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            return proc.returncode == 0
        except (FileNotFoundError, asyncio.TimeoutError, Exception):
            return False

    async def _init_lean_project(self):
        """使用 lake 初始化 Lean 项目。"""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.config.lake_executable, "init", "TuringWorkspace",
                cwd=str(self._project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)
            logger.info("Lean 项目已初始化")
        except Exception as e:
            logger.warning(f"Lean 项目初始化失败: {e}")

    # ------------------------------------------------------------------
    #  核心编译接口
    # ------------------------------------------------------------------

    async def compile(self, lean_code: str, filename: Optional[str] = None) -> LeanResult:
        """
        编译一段 Lean 4 代码。

        Args:
            lean_code: Lean 4 源代码
            filename: 可选的文件名（用于临时文件）

        Returns:
            LeanResult 包含编译结果和结构化错误信息
        """
        t0 = time.monotonic()

        # 创建临时文件
        if filename is None:
            filename = f"turing_proof_{int(time.time())}.lean"

        file_path = self._project_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(lean_code, encoding="utf-8")

        try:
            # 使用 lake env lean 编译，以加载项目依赖（如 Mathlib）
            proc = await asyncio.create_subprocess_exec(
                self.config.lake_executable, "env",
                self.config.executable, str(file_path.resolve()),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._project_dir.resolve()),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self.config.compile_timeout
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            duration = (time.monotonic() - t0) * 1000

            # 解析错误
            errors, warnings = self._parse_output(stderr + stdout)

            success = proc.returncode == 0 and len(errors) == 0

            return LeanResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                errors=[e.__dict__ for e in errors],
                warnings=[w.__dict__ for w in warnings],
                duration_ms=duration,
                code=lean_code,
            )

        except asyncio.TimeoutError:
            duration = (time.monotonic() - t0) * 1000
            return LeanResult(
                success=False,
                stderr=f"编译超时 ({self.config.compile_timeout}s)",
                errors=[{"message": "编译超时", "category": "timeout"}],
                duration_ms=duration,
                code=lean_code,
            )
        except FileNotFoundError:
            return LeanResult(
                success=False,
                stderr=f"Lean 可执行文件未找到: {self.config.executable}",
                errors=[{"message": "Lean 未安装", "category": "environment"}],
                code=lean_code,
            )
        except Exception as e:
            return LeanResult(
                success=False,
                stderr=str(e),
                errors=[{"message": str(e), "category": "unknown"}],
                code=lean_code,
            )
        finally:
            # 清理临时文件
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    pass

    async def compile_and_retry(
        self,
        lean_code: str,
        fix_callback=None,
        max_retries: Optional[int] = None,
    ) -> tuple[LeanResult, int]:
        """
        编译代码，失败时自动重试。

        Args:
            lean_code: 初始 Lean 代码
            fix_callback: 异步回调函数 (code, errors) -> new_code，用于修复代码
            max_retries: 最大重试次数

        Returns:
            (最终结果, 尝试次数)
        """
        retries = max_retries or self.config.max_retries
        current_code = lean_code

        for attempt in range(1, retries + 1):
            result = await self.compile(current_code)

            if result.success:
                logger.info(f"Lean 编译成功（第{attempt}次尝试）")
                return result, attempt

            logger.info(
                f"Lean 编译失败（第{attempt}/{retries}次）: "
                f"{result.error_summary[:100]}"
            )

            if attempt < retries and fix_callback:
                try:
                    new_code = await fix_callback(current_code, result.errors)
                    if new_code and new_code != current_code:
                        current_code = new_code
                    else:
                        break  # 无法修复
                except Exception as e:
                    logger.error(f"代码修复回调失败: {e}")
                    break

        return result, attempt

    # ------------------------------------------------------------------
    #  错误解析
    # ------------------------------------------------------------------

    def _parse_output(self, output: str) -> tuple[list[LeanError], list[LeanError]]:
        """解析 Lean 编译输出，提取错误和警告。"""
        errors = []
        warnings = []

        # Lean 4 错误格式: file:line:col: error: message
        error_pattern = re.compile(
            r"(.+?):(\d+):(\d+):\s*(error|warning|info):\s*(.*)",
            re.MULTILINE,
        )

        for match in error_pattern.finditer(output):
            le = LeanError(
                file=match.group(1),
                line=int(match.group(2)),
                column=int(match.group(3)),
                severity=match.group(4),
                message=match.group(5).strip(),
            )
            le.category = self._categorize_error(le.message)

            if le.severity == "error":
                errors.append(le)
            elif le.severity == "warning":
                warnings.append(le)

        # 也检查通用错误信息
        if not errors and ("error" in output.lower() or "failed" in output.lower()):
            for line in output.splitlines():
                line = line.strip()
                if "error" in line.lower() or "failed" in line.lower():
                    errors.append(LeanError(
                        message=line,
                        category=self._categorize_error(line),
                    ))

        return errors, warnings

    @staticmethod
    def _categorize_error(message: str) -> str:
        """将错误信息分类。"""
        msg_lower = message.lower()

        if any(kw in msg_lower for kw in ["syntax", "expected", "unexpected", "parse"]):
            return "syntax"
        if any(kw in msg_lower for kw in ["type mismatch", "has type", "expected type"]):
            return "type_mismatch"
        if any(kw in msg_lower for kw in ["unknown identifier", "undeclared", "not found"]):
            return "unknown_identifier"
        if any(kw in msg_lower for kw in ["tactic", "unsolved goals", "no goals"]):
            return "tactic_error"
        if any(kw in msg_lower for kw in ["import", "module"]):
            return "import_error"
        return "other"

    # ------------------------------------------------------------------
    #  Lake 构建
    # ------------------------------------------------------------------

    async def lake_build(self) -> LeanResult:
        """使用 lake 构建整个项目。"""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.config.lake_executable, "build",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._project_dir),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=300
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            return LeanResult(
                success=proc.returncode == 0,
                stdout=stdout,
                stderr=stderr,
            )
        except Exception as e:
            return LeanResult(success=False, stderr=str(e))

    # ------------------------------------------------------------------
    #  辅助方法
    # ------------------------------------------------------------------

    async def get_lean_version(self) -> str:
        """获取 Lean 版本。"""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.config.executable, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            return stdout.decode().strip()
        except Exception:
            return "未知"

    async def search_mathlib(self, query: str) -> list[dict]:
        """
        搜索 Mathlib 中的相关定理（通过 lake env 或 grep）。

        这是一个简化实现；完整版本应使用 Mathlib 的搜索 API。
        """
        results = []
        mathlib_path = self._project_dir / ".lake" / "packages" / "mathlib"

        if not mathlib_path.exists():
            return results

        try:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-r", "--include=*.lean", "-l", query,
                str(mathlib_path / "Mathlib"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)

            for line in stdout.decode().splitlines()[:10]:
                results.append({
                    "file": line.strip(),
                    "query": query,
                })
        except Exception as e:
            logger.debug(f"Mathlib 搜索失败: {e}")

        return results

    async def check_status(self) -> dict:
        """检查 Lean 环境状态。"""
        version = await self.get_lean_version()
        lean_ok = version != "未知"

        lake_ok = await self._check_executable(self.config.lake_executable)

        mathlib_path = self._project_dir / ".lake" / "packages" / "mathlib"
        mathlib_ok = mathlib_path.exists()

        return {
            "lean_available": lean_ok,
            "lean_version": version,
            "lake_available": lake_ok,
            "mathlib_available": mathlib_ok,
            "project_dir": str(self._project_dir),
        }
