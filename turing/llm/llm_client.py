"""
LLM 客户端 — 统一接口与本地 Ollama 及外部 OpenAI 兼容 API 通信。

支持同步和流式（streaming）输出，自动处理连接错误和超时。
默认通过 Ollama REST API 调用本地 ``qwen3-coder:30b`` 模型。

典型用法::

    client = LLMClient(config)
    await client.initialize()
    response = await client.chat([ChatMessage(role=\"user\", content=\"证明 1+1=2\")])
    print(response.content)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx
from loguru import logger

from turing.config import LLMConfig, get_config


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class LLMClient:
    """
    与 Ollama (或 OpenAI 兼容 API）通信的异步客户端。

    - 主模型：本地 Qwen3-30B:coder (通过 Ollama)
    - 轻量模型：用于简单任务
    - 外部模型：可选的备用 API
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm
        self._client: Optional[httpx.AsyncClient] = None
        self._external_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout, connect=10),
            )
        return self._client

    async def _get_external_client(self) -> httpx.AsyncClient:
        ext = self.config.external_llm
        if self._external_client is None or self._external_client.is_closed:
            self._external_client = httpx.AsyncClient(
                base_url=ext.get("base_url", "https://api.openai.com/v1"),
                timeout=httpx.Timeout(self.config.timeout, connect=10),
                headers={"Authorization": f"Bearer {ext.get('api_key', '')}"},
            )
        return self._external_client

    # ------------------------------------------------------------------
    #  核心聊天接口
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_external: bool = False,
    ) -> LLMResponse:
        """
        发送聊天请求到 LLM 并返回完整响应。

        Args:
            messages: 对话消息列表
            model: 覆盖默认模型名称
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大 token 数
            system_prompt: 可选的 system prompt（会插入到消息头部）
            use_external: 是否使用外部 LLM
        """
        if use_external and self.config.external_llm.get("enabled"):
            return await self._chat_external(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            )

        return await self._chat_ollama(
            messages,
            model=model or self.config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    async def chat_simple(
        self, prompt: str, *, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """简化接口：直接传入用户消息，返回文本。"""
        messages = [ChatMessage(role="user", content=prompt)]
        resp = await self.chat(messages, system_prompt=system_prompt, **kwargs)
        return resp.content

    async def chat_light(
        self, prompt: str, *, system_prompt: Optional[str] = None
    ) -> str:
        """使用轻量模型完成简单任务。"""
        messages = [ChatMessage(role="user", content=prompt)]
        resp = await self.chat(
            messages, model=self.config.light_model, system_prompt=system_prompt
        )
        return resp.content

    # ------------------------------------------------------------------
    #  Ollama 实现
    # ------------------------------------------------------------------

    async def _chat_ollama(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        system_prompt: Optional[str],
    ) -> LLMResponse:
        client = await self._get_client()

        msg_dicts = []
        if system_prompt:
            msg_dicts.append({"role": "system", "content": system_prompt})
        for m in messages:
            msg_dicts.append({"role": m.role, "content": m.content})

        payload = {
            "model": model,
            "messages": msg_dicts,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }

        t0 = time.monotonic()
        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} {e.response.text}")
            raise
        except httpx.ConnectError:
            logger.error(
                f"无法连接到 Ollama ({self.config.base_url})。"
                "请确保 Ollama 已启动：ollama serve"
            )
            raise
        except Exception as e:
            logger.error(f"LLM 请求失败: {e}")
            raise

        latency = (time.monotonic() - t0) * 1000

        content = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return LLMResponse(
            content=content,
            model=model,
            tokens_used=tokens,
            latency_ms=latency,
            raw=data,
        )

    # ------------------------------------------------------------------
    #  Ollama 流式接口
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """流式返回 LLM 响应。"""
        client = await self._get_client()

        msg_dicts = []
        if system_prompt:
            msg_dicts.append({"role": "system", "content": system_prompt})
        for m in messages:
            msg_dicts.append({"role": m.role, "content": m.content})

        payload = {
            "model": model or self.config.model,
            "messages": msg_dicts,
            "stream": True,
        }

        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    # ------------------------------------------------------------------
    #  外部 LLM 实现
    # ------------------------------------------------------------------

    async def _chat_external(
        self,
        messages: list[ChatMessage],
        model: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> LLMResponse:
        ext = self.config.external_llm
        client = await self._get_external_client()

        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        payload = {
            "model": model or ext.get("model", "gpt-4"),
            "messages": msg_dicts,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        t0 = time.monotonic()
        resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.monotonic() - t0) * 1000

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        tokens = usage.get("total_tokens", 0)

        return LLMResponse(
            content=content,
            model=ext.get("model", "gpt-4"),
            tokens_used=tokens,
            latency_ms=latency,
            raw=data,
        )

    # ------------------------------------------------------------------
    #  模型管理
    # ------------------------------------------------------------------

    async def check_model_available(self, model: Optional[str] = None) -> bool:
        """检查指定模型是否在 Ollama 中可用。"""
        try:
            client = await self._get_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            target = model or self.config.model
            return any(target in m for m in models)
        except Exception as e:
            logger.warning(f"模型检查失败: {e}")
            return False

    async def list_models(self) -> list[str]:
        """列出 Ollama 上可用的模型。"""
        try:
            client = await self._get_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    #  生命周期
    # ------------------------------------------------------------------

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._external_client and not self._external_client.is_closed:
            await self._external_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
