"""
Turing Web 前端 — 基于 FastAPI 的简约数学定理证明演示界面。

包含：
  - ``app.py``: FastAPI 应用（REST API + SSE 流式输出）
  - ``static/index.html``: 单页前端（原生 HTML/CSS/JS，无构建步骤）

启动方式::

    python -m web.app            # 默认 http://localhost:8000
    python -m web.app --port 9000
"""
