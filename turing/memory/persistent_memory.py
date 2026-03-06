"""
持久记忆（Persistent / Episodic Memory）— 基于 SQLite 的元认知记录系统。

跨所有生命周期记录：
- 决策经验（策略成功/失败及教训）
- 进化日志（阶段性能力评估）
- 系统配置历史（prompt 版本、架构变更）
- 反思报告
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from turing.config import PersistentMemoryConfig, get_config


@dataclass
class Experience:
    """一条经验记录。"""
    id: Optional[int] = None
    context_hash: str = ""
    strategy_used: str = ""
    outcome: str = ""            # success | partial | failure
    lesson: str = ""
    priority: float = 0.5
    created_at: str = ""
    last_applied: str = ""
    apply_count: int = 0
    category: str = ""           # prove | tactic | conjecture | ...
    tags: str = "[]"


@dataclass
class EvolutionLog:
    """进化日志条目。"""
    id: Optional[int] = None
    phase: int = 0
    timestamp: str = ""
    success_rate: float = 0.0
    skills_gained: str = "[]"
    weak_areas: str = "[]"
    theorems_proved: int = 0
    reflection_report: str = ""
    improvements_planned: str = "[]"


@dataclass
class PromptVersion:
    """Prompt 版本记录。"""
    id: Optional[int] = None
    agent_id: str = ""
    version: int = 0
    prompt_content: str = ""
    change_reason: str = ""
    performance_before: float = 0.0
    performance_after: float = 0.0
    timestamp: str = ""
    is_active: bool = True


class PersistentMemory:
    """
    SQLite 持久记忆系统。

    管理经验、进化日志、prompt 版本历史等元认知数据。
    """

    def __init__(self, config: Optional[PersistentMemoryConfig] = None):
        self.config = config or get_config().memory.persistent
        self._db_path = self.config.db_path
        self._initialized = False

    def initialize(self):
        """创建数据库和表结构。"""
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)

        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experience (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_hash TEXT,
                    strategy_used TEXT,
                    outcome TEXT CHECK(outcome IN ('success', 'partial', 'failure')),
                    lesson TEXT,
                    priority REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_applied TIMESTAMP,
                    apply_count INTEGER DEFAULT 0,
                    category TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS evolution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phase INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success_rate REAL DEFAULT 0.0,
                    skills_gained TEXT DEFAULT '[]',
                    weak_areas TEXT DEFAULT '[]',
                    theorems_proved INTEGER DEFAULT 0,
                    reflection_report TEXT DEFAULT '',
                    improvements_planned TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    version INTEGER,
                    prompt_content TEXT,
                    change_reason TEXT DEFAULT '',
                    performance_before REAL DEFAULT 0.0,
                    performance_after REAL DEFAULT 0.0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS task_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT,
                    task_description TEXT,
                    status TEXT CHECK(status IN ('success', 'partial', 'failure', 'abandoned')),
                    duration_seconds REAL,
                    strategies_tried TEXT DEFAULT '[]',
                    final_strategy TEXT DEFAULT '',
                    lean_attempts INTEGER DEFAULT 0,
                    lean_success BOOLEAN DEFAULT 0,
                    area TEXT DEFAULT '',
                    difficulty INTEGER DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS agent_modifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    modification_type TEXT,
                    reason TEXT,
                    diff TEXT,
                    validation_result TEXT CHECK(validation_result IN ('improved', 'degraded', 'neutral')),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_experience_context ON experience(context_hash);
                CREATE INDEX IF NOT EXISTS idx_experience_outcome ON experience(outcome);
                CREATE INDEX IF NOT EXISTS idx_experience_priority ON experience(priority DESC);
                CREATE INDEX IF NOT EXISTS idx_task_log_area ON task_log(area);
                CREATE INDEX IF NOT EXISTS idx_task_log_status ON task_log(status);
            """)

        self._initialized = True
        logger.info(f"持久记忆初始化完成: {self._db_path}")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_init(self):
        if not self._initialized:
            self.initialize()

    # ------------------------------------------------------------------
    #  经验管理
    # ------------------------------------------------------------------

    @staticmethod
    def compute_context_hash(context: str) -> str:
        """计算问题情境的语义哈希。"""
        return hashlib.sha256(context.encode()).hexdigest()[:16]

    def record_experience(
        self,
        context: str,
        strategy_used: str,
        outcome: str,
        lesson: str,
        category: str = "",
        tags: Optional[list[str]] = None,
    ) -> int:
        """记录一条经验。"""
        self._ensure_init()
        ctx_hash = self.compute_context_hash(context)
        now = datetime.now().isoformat()
        tags_json = json.dumps(tags or [], ensure_ascii=False)

        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO experience
                   (context_hash, strategy_used, outcome, lesson, created_at, category, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ctx_hash, strategy_used, outcome, lesson, now, category, tags_json),
            )
            return cursor.lastrowid

    def get_relevant_experiences(
        self,
        context: str,
        limit: int = 5,
        outcome_filter: Optional[str] = None,
    ) -> list[dict]:
        """检索相关经验，按优先级排序。"""
        self._ensure_init()
        ctx_hash = self.compute_context_hash(context)

        query = "SELECT * FROM experience WHERE context_hash = ?"
        params: list = [ctx_hash]

        if outcome_filter:
            query += " AND outcome = ?"
            params.append(outcome_filter)

        query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_top_strategies(self, category: str = "", limit: int = 10) -> list[dict]:
        """获取高优先级的成功策略。"""
        self._ensure_init()
        query = """SELECT * FROM experience
                   WHERE outcome = 'success'"""
        params: list = []
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY priority DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def update_priority(self, experience_id: int, success: bool):
        """更新经验的优先级权重。"""
        self._ensure_init()
        # 经验优先级调整参数（硬编码，避免配置膨胀）
        _PRIORITY_STEP = 0.1
        _PRIORITY_MAX = 1.0
        _PRIORITY_MIN = 0.0

        with self._connect() as conn:
            row = conn.execute(
                "SELECT priority, apply_count FROM experience WHERE id = ?",
                (experience_id,),
            ).fetchone()

            if not row:
                return

            current_priority = row["priority"]
            apply_count = row["apply_count"]

            if success:
                new_priority = min(current_priority + _PRIORITY_STEP, _PRIORITY_MAX)
            else:
                new_priority = max(current_priority - _PRIORITY_STEP, _PRIORITY_MIN)

            conn.execute(
                """UPDATE experience
                   SET priority = ?, last_applied = ?, apply_count = ?
                   WHERE id = ?""",
                (new_priority, datetime.now().isoformat(), apply_count + 1, experience_id),
            )

    def get_failure_patterns(self, limit: int = 10) -> list[dict]:
        """获取常见的失败模式。"""
        self._ensure_init()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT lesson, COUNT(*) as count, category
                   FROM experience
                   WHERE outcome = 'failure'
                   GROUP BY lesson
                   ORDER BY count DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    #  任务日志
    # ------------------------------------------------------------------

    def log_task(
        self,
        task_type: str,
        task_description: str,
        status: str,
        duration_seconds: float,
        strategies_tried: Optional[list[str]] = None,
        final_strategy: str = "",
        lean_attempts: int = 0,
        lean_success: bool = False,
        area: str = "",
        difficulty: int = 0,
    ) -> int:
        """记录任务执行情况。"""
        self._ensure_init()
        now = datetime.now().isoformat()
        strategies = json.dumps(strategies_tried or [], ensure_ascii=False)

        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO task_log
                   (task_type, task_description, status, duration_seconds,
                    strategies_tried, final_strategy, lean_attempts, lean_success,
                    area, difficulty, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (task_type, task_description, status, duration_seconds,
                 strategies, final_strategy, lean_attempts, lean_success,
                 area, difficulty, now),
            )
            return cursor.lastrowid

    def get_task_stats(self, area: Optional[str] = None) -> dict:
        """获取任务统计信息。"""
        self._ensure_init()
        with self._connect() as conn:
            base_query = "SELECT status, COUNT(*) as count FROM task_log"
            params = []
            if area:
                base_query += " WHERE area = ?"
                params.append(area)
            base_query += " GROUP BY status"

            rows = conn.execute(base_query, params).fetchall()
            stats = {row["status"]: row["count"] for row in rows}

            # 总数和成功率
            total = sum(stats.values())
            success = stats.get("success", 0)
            stats["total"] = total
            stats["success_rate"] = success / total if total > 0 else 0.0

            # 平均耗时
            avg_query = "SELECT AVG(duration_seconds) as avg_dur FROM task_log"
            if area:
                avg_query += " WHERE area = ?"
            row = conn.execute(avg_query, params).fetchone()
            stats["avg_duration_seconds"] = row["avg_dur"] or 0.0

            # Lean 编译通过率
            lean_query = """SELECT
                SUM(CASE WHEN lean_success THEN 1 ELSE 0 END) as lean_ok,
                SUM(lean_attempts) as lean_total
                FROM task_log"""
            if area:
                lean_query += " WHERE area = ?"
            row = conn.execute(lean_query, params).fetchone()
            lean_ok = row["lean_ok"] or 0
            lean_total = row["lean_total"] or 0
            stats["lean_first_pass_rate"] = lean_ok / max(lean_total, 1)
            stats["total_tasks"] = total

            return stats

    def get_task_count(self) -> int:
        """获取总任务数。"""
        self._ensure_init()
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM task_log").fetchone()
            return row["c"]

    def get_area_stats(self) -> dict[str, dict]:
        """按领域获取统计信息。"""
        self._ensure_init()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT area,
                    COUNT(*) as total,
                    SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as success,
                    SUM(CASE WHEN status='failure' THEN 1 ELSE 0 END) as failure
                   FROM task_log
                   WHERE area != ''
                   GROUP BY area"""
            ).fetchall()

            result = {}
            for r in rows:
                area = r["area"]
                total = r["total"]
                success_count = r["success"]
                result[area] = {
                    "total": total,
                    "success": success_count,
                    "failure": r["failure"],
                    "success_rate": success_count / total if total > 0 else 0.0,
                }
            return result

    # ------------------------------------------------------------------
    #  进化日志
    # ------------------------------------------------------------------

    def record_reflection(
        self,
        phase: int,
        success_rate: float,
        skills_gained: list[str],
        weak_areas: list[str],
        theorems_proved: int,
        reflection_report: str,
        improvements_planned: list[str],
    ) -> int:
        """记录阶段性反思。"""
        self._ensure_init()
        now = datetime.now().isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO evolution_log
                   (phase, timestamp, success_rate, skills_gained, weak_areas,
                    theorems_proved, reflection_report, improvements_planned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (phase, now, success_rate,
                 json.dumps(skills_gained, ensure_ascii=False),
                 json.dumps(weak_areas, ensure_ascii=False),
                 theorems_proved, reflection_report,
                 json.dumps(improvements_planned, ensure_ascii=False)),
            )
            return cursor.lastrowid

    def get_latest_reflection(self) -> Optional[dict]:
        """获取最近一次反思记录。"""
        self._ensure_init()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evolution_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_reflection_count(self) -> int:
        """获取反思次数。"""
        self._ensure_init()
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM evolution_log").fetchone()
            return row["c"]

    # ------------------------------------------------------------------
    #  Prompt 版本管理
    # ------------------------------------------------------------------

    def save_prompt_version(
        self,
        agent_id: str,
        prompt_content: str,
        change_reason: str = "",
        performance_before: float = 0.0,
    ) -> int:
        """保存 prompt 新版本。"""
        self._ensure_init()

        with self._connect() as conn:
            # 获取当前最大版本号
            row = conn.execute(
                "SELECT MAX(version) as v FROM prompt_versions WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            new_version = (row["v"] or 0) + 1

            # 旧版本标记为非活跃
            conn.execute(
                "UPDATE prompt_versions SET is_active = 0 WHERE agent_id = ?",
                (agent_id,),
            )

            cursor = conn.execute(
                """INSERT INTO prompt_versions
                   (agent_id, version, prompt_content, change_reason,
                    performance_before, timestamp, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, 1)""",
                (agent_id, new_version, prompt_content, change_reason,
                 performance_before, datetime.now().isoformat()),
            )
            return cursor.lastrowid

    def get_active_prompt(self, agent_id: str) -> Optional[str]:
        """获取指定智能体的当前活跃 prompt。"""
        self._ensure_init()
        with self._connect() as conn:
            row = conn.execute(
                """SELECT prompt_content FROM prompt_versions
                   WHERE agent_id = ? AND is_active = 1
                   ORDER BY version DESC LIMIT 1""",
                (agent_id,),
            ).fetchone()
            return row["prompt_content"] if row else None

    def rollback_prompt(self, agent_id: str) -> Optional[str]:
        """回滚到上一个 prompt 版本。"""
        self._ensure_init()
        with self._connect() as conn:
            # 获取最新两个版本
            rows = conn.execute(
                """SELECT id, version, prompt_content FROM prompt_versions
                   WHERE agent_id = ?
                   ORDER BY version DESC LIMIT 2""",
                (agent_id,),
            ).fetchall()

            if len(rows) < 2:
                return None

            # 当前版本标记为非活跃
            conn.execute(
                "UPDATE prompt_versions SET is_active = 0 WHERE id = ?",
                (rows[0]["id"],),
            )
            # 上一版本标记为活跃
            conn.execute(
                "UPDATE prompt_versions SET is_active = 1 WHERE id = ?",
                (rows[1]["id"],),
            )
            return rows[1]["prompt_content"]

    # ------------------------------------------------------------------
    #  智能体修改记录
    # ------------------------------------------------------------------

    def record_agent_modification(
        self,
        agent_id: str,
        modification_type: str,
        reason: str,
        diff: str,
        validation_result: str = "neutral",
    ) -> int:
        """记录智能体修改。"""
        self._ensure_init()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO agent_modifications
                   (agent_id, modification_type, reason, diff, validation_result, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (agent_id, modification_type, reason, diff,
                 validation_result, datetime.now().isoformat()),
            )
            return cursor.lastrowid

    # ------------------------------------------------------------------
    #  综合统计
    # ------------------------------------------------------------------

    def get_comprehensive_stats(self) -> dict:
        """获取综合统计信息。"""
        self._ensure_init()
        return {
            "task_stats": self.get_task_stats(),
            "area_stats": self.get_area_stats(),
            "failure_patterns": self.get_failure_patterns(5),
            "total_experiences": self._count_table("experience"),
            "total_reflections": self.get_reflection_count(),
            "total_tasks": self.get_task_count(),
        }

    def _count_table(self, table: str) -> int:
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()
            return row["c"]

    # ------------------------------------------------------------------
    #  备份
    # ------------------------------------------------------------------

    def backup(self, backup_path: Optional[str] = None) -> str:
        """备份数据库。"""
        self._ensure_init()
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(
                self.config.log_dir, f"persistent_backup_{timestamp}.db"
            )

        import shutil
        shutil.copy2(self._db_path, backup_path)
        logger.info(f"持久记忆备份: {backup_path}")
        return backup_path
