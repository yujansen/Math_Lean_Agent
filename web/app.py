"""
Turing Web API — FastAPI 后端，为前端提供 REST + SSE 接口。

API 概览::

    GET  /api/status          系统状态
    GET  /api/knowledge       知识库统计 + 定理列表
    GET  /api/experience      经验统计 + 反思记录
    GET  /api/evolution       全分支演化数据（从 theorem_library.json）
    POST /api/prove           提交证明任务（异步 SSE 流）
    GET  /api/theorems        已证明定理列表
    GET  /api/task-history     历史任务列表

启动::

    python -m web.app
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from turing.config import get_config, reset_config
from turing.agents.skill_based_agent import SkillBasedTuringAgent
from turing.utils import setup_logging


# ── 全局状态 ──────────────────────────────────────────────────────────

_agent: Optional[SkillBasedTuringAgent] = None
_proving_lock = asyncio.Lock()      # 同时只允许一个证明任务
_active_task: dict | None = None    # 当前正在执行的任务信息


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时初始化/清理 Turing 智能体。"""
    global _agent
    config = get_config()
    setup_logging(
        log_dir=Path(config.system.data_dir) / "logs",
        log_prefix="turing_web",
        level=config.system.log_level,
    )
    _agent = SkillBasedTuringAgent(config)
    await _agent.initialize()
    logger.info("[Web] Turing 智能体初始化完成")
    yield
    if _agent:
        await _agent.shutdown()
        logger.info("[Web] Turing 智能体已关闭")


# ── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Turing — 数学研究智能体",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Pydantic Models ─────────────────────────────────────────────────

class ProveRequest(BaseModel):
    task: str


# ── API: 系统状态 ────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """返回系统整体状态。"""
    assert _agent is not None
    lean_status = await _agent.lean.check_status()
    ltm_stats = _agent.ltm.get_stats()
    task_count = _agent.pm.get_task_count()
    reflection_count = _agent.pm.get_reflection_count()
    resource_snapshot = _agent.resource_manager.assess()

    return {
        "system": {
            "name": _agent._turing_config.system.name,
            "version": _agent._turing_config.system.version,
            "mode": _agent._turing_config.agents.mode,
            "model": _agent._turing_config.llm.model,
            "light_model": _agent._turing_config.llm.light_model,
        },
        "lean": lean_status,
        "knowledge": ltm_stats,
        "tasks": {
            "total": task_count,
            "reflections": reflection_count,
        },
        "resources": {
            "level": resource_snapshot.level.value,
            "cpu_percent": resource_snapshot.cpu_percent,
            "ram_used_gb": round(resource_snapshot.ram_total_gb - resource_snapshot.ram_free_gb, 1),
            "ram_total_gb": round(resource_snapshot.ram_total_gb, 1),
        },
        "busy": _active_task is not None,
    }


# ── API: 知识库 ──────────────────────────────────────────────────────

@app.get("/api/knowledge")
async def get_knowledge():
    """知识库统计 + 最近的定理列表。"""
    assert _agent is not None
    stats = _agent.ltm.get_stats()
    theorems = _agent.ltm.get_proven_theorems(limit=100)

    return {
        "stats": stats,
        "theorems": theorems,
    }


@app.get("/api/theorems")
async def get_theorems(area: str = "", limit: int = 50):
    """获取已证明的定理列表。"""
    assert _agent is not None
    theorems = _agent.ltm.get_proven_theorems(area=area, limit=limit)
    return {"theorems": theorems, "total": len(theorems)}


# ── API: 经验与演化 ──────────────────────────────────────────────────

@app.get("/api/experience")
async def get_experience():
    """经验统计、反思记录、失败模式。"""
    assert _agent is not None
    comprehensive = _agent.pm.get_comprehensive_stats()
    latest_reflection = _agent.pm.get_latest_reflection()
    area_stats = _agent.pm.get_area_stats()

    return {
        "comprehensive": comprehensive,
        "latest_reflection": latest_reflection,
        "area_stats": area_stats,
        "skill_levels": getattr(_agent, '_skill_levels', {}),
        "weak_areas": getattr(_agent, '_weak_areas', []),
    }


@app.get("/api/task-history")
async def get_task_history(limit: int = 50):
    """最近的任务执行记录。"""
    assert _agent is not None
    recent = list(reversed(getattr(_agent, '_recent_results', [])))[:limit]
    return {"tasks": recent, "total": len(recent)}


@app.get("/api/evolution")
async def get_evolution():
    """从 theorem_library.json 读取完整的演化数据。"""
    lib_path = Path(_PROJECT_ROOT) / "data" / "theorem_library.json"
    if not lib_path.exists():
        return {"available": False, "data": None}
    try:
        data = json.loads(lib_path.read_text(encoding="utf-8"))
        return {"available": True, "data": data}
    except Exception as e:
        logger.warning(f"[Web] 读取 theorem_library.json 失败: {e}")
        return {"available": False, "error": str(e)}


# ── API: 证明任务（SSE 流式输出）────────────────────────────────────

@app.post("/api/prove")
async def prove_theorem(req: ProveRequest):
    """
    提交证明任务。返回 SSE (Server-Sent Events) 流。

    事件类型:
      - status:      阶段更新
      - llm_output:  LLM 推理输出（左窗口）
      - lean_result: Lean 4 编译结果（右窗口）
      - result:      最终结果（元数据）
      - error:       错误信息
    """
    assert _agent is not None

    if _active_task is not None:
        raise HTTPException(status_code=429, detail="已有任务正在执行，请等待完成")

    task_text = req.task.strip()
    if not task_text:
        raise HTTPException(status_code=400, detail="任务不能为空")

    async def event_stream():
        global _active_task
        task_id = str(uuid.uuid4())[:8]
        _active_task = {"id": task_id, "task": task_text, "start": time.time()}

        try:
            # 阶段 1: 开始
            yield _sse("status", {"phase": "started", "task": task_text, "task_id": task_id})
            await asyncio.sleep(0.05)

            # 阶段 2: 执行证明
            yield _sse("status", {"phase": "proving", "message": "正在调用 Turing 智能体..."})

            t0 = time.time()
            result = await _agent.process_task(task_text)
            elapsed = time.time() - t0

            result["elapsed_seconds"] = round(elapsed, 1)
            result["task_id"] = task_id

            # 分步推送：LLM 推理 → Lean 编译 → 最终结果
            llm_text = result.get("llm_response") or result.get("natural_language_proof", "")
            if llm_text:
                yield _sse("llm_output", {"text": llm_text})
                await asyncio.sleep(0.05)

            lean_code = result.get("lean_code", "")
            if lean_code:
                yield _sse("lean_result", {
                    "code": lean_code,
                    "success": result.get("success", False),
                    "attempts": result.get("attempts", 0),
                })
                await asyncio.sleep(0.05)

            # 阶段 3: 完整结果（元数据）
            yield _sse("result", _serialize_result(result))

        except Exception as e:
            logger.exception(f"[Web] 证明任务异常: {e}")
            yield _sse("error", {"message": str(e)})
        finally:
            _active_task = None

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 首页 ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端单页应用。"""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Turing Web — 前端文件未找到</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ── 工具函数 ─────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    """格式化一条 SSE 消息。"""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _serialize_result(result: dict) -> dict:
    """清理 process_task 返回值，确保 JSON 可序列化。"""
    safe = {}
    for key, val in result.items():
        if isinstance(val, (str, int, float, bool, type(None))):
            safe[key] = val
        elif isinstance(val, (list, dict)):
            try:
                json.dumps(val, default=str)
                safe[key] = val
            except (TypeError, ValueError):
                safe[key] = str(val)
        else:
            safe[key] = str(val)
    return safe


# ── 入口 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Turing Web 前端服务器")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    uvicorn.run(
        "web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
