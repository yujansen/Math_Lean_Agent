"""
网络题目抓取与定理名称搜索。

功能：
- 从 Project Euler / Mathlib GitHub Issues 获取训练题目。
- 通过 Loogle（Mathlib 搜索引擎）和 ProofWiki 查询定理标准名称。
- 从 Lean 代码中提取 ``exact Xxx.yyy`` 引用的 Mathlib 定理名。
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

from loguru import logger

from turing.config import WebConfig, get_config

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False


class ProblemScraper:
    """
    数学问题网络抓取器。

    从多种在线来源获取数学问题，支持：
    - Project Euler (算法 & 数论)
    - GitHub Mathlib Issues (形式化数学)
    - 自定义来源
    """

    def __init__(self, config: Optional[WebConfig] = None):
        self.config = config or get_config().web
        self._session: Optional[Any] = None

    async def _get_session(self):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp 未安装。运行: pip install aiohttp beautifulsoup4")
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self.config.user_agent},
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    #  统一接口
    # ------------------------------------------------------------------

    async def fetch_problems(
        self,
        area: str = "",
        difficulty: int = 3,
        count: int = 5,
        source: str = "",
    ) -> list[dict]:
        """
        从网络获取数学问题。

        Args:
            area: 数学领域过滤
            difficulty: 目标难度 (1-10)
            count: 问题数量
            source: 指定来源 (project_euler | mathlib_issues)
        """
        if not self.config.enabled:
            return []

        problems = []

        try:
            if source == "project_euler" or not source:
                pe_problems = await self._fetch_project_euler(count)
                problems.extend(pe_problems)

            if source == "mathlib_issues" or (not source and len(problems) < count):
                ml_problems = await self._fetch_mathlib_issues(
                    count - len(problems)
                )
                problems.extend(ml_problems)

        except Exception as e:
            logger.warning(f"问题抓取失败: {e}")

        return problems[:count]

    # ------------------------------------------------------------------
    #  Project Euler
    # ------------------------------------------------------------------

    async def _fetch_project_euler(self, count: int = 5) -> list[dict]:
        """从 Project Euler 获取问题。"""
        problems = []

        try:
            session = await self._get_session()

            # Project Euler recent problems page
            url = "https://projecteuler.net/archives"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return problems
                html = await resp.text()

            if not BS4_AVAILABLE:
                return []

            soup = BeautifulSoup(html, "html.parser")

            # 提取问题列表
            for row in soup.select("table#problems_table tr")[1:count+1]:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    num = cols[0].get_text(strip=True)
                    link = cols[1].find("a")
                    if link:
                        title = link.get_text(strip=True)
                        problem_url = f"https://projecteuler.net/{link.get('href', '')}"

                        problems.append({
                            "title": f"Project Euler #{num}: {title}",
                            "statement": title,
                            "area": "number_theory",
                            "difficulty": min(int(num) // 100 + 2, 10) if num.isdigit() else 3,
                            "source": "project_euler",
                            "url": problem_url,
                            "formalizable": True,
                        })

        except Exception as e:
            logger.debug(f"Project Euler 抓取失败: {e}")

        return problems

    # ------------------------------------------------------------------
    #  Mathlib Issues
    # ------------------------------------------------------------------

    async def _fetch_mathlib_issues(self, count: int = 5) -> list[dict]:
        """从 Mathlib GitHub Issues 获取问题。"""
        problems = []

        try:
            session = await self._get_session()

            url = "https://api.github.com/repos/leanprover-community/mathlib4/issues"
            params = {
                "state": "open",
                "labels": "good first issue",
                "per_page": count,
                "sort": "updated",
            }

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    # 尝试不带 label 过滤
                    params.pop("labels", None)
                    async with session.get(url, params=params) as resp2:
                        if resp2.status != 200:
                            return problems
                        issues = await resp2.json()
                else:
                    issues = await resp.json()

            for issue in issues[:count]:
                title = issue.get("title", "")
                body = issue.get("body", "")
                labels = [l.get("name", "") for l in issue.get("labels", [])]

                # 推断数学领域
                area = "formalization"
                for label in labels:
                    if any(a in label.lower() for a in [
                        "algebra", "analysis", "topology",
                        "number", "combinat", "geometry",
                    ]):
                        area = label
                        break

                problems.append({
                    "title": title,
                    "statement": (body or "")[:500],
                    "area": area,
                    "difficulty": 5,
                    "source": "mathlib_issues",
                    "url": issue.get("html_url", ""),
                    "formalizable": True,
                    "labels": labels,
                })

        except Exception as e:
            logger.debug(f"Mathlib Issues 抓取失败: {e}")

        return problems

    # ------------------------------------------------------------------
    #  定理名称搜索
    # ------------------------------------------------------------------

    async def search_theorem_name(self, theorem_statement: str, lean_code: str = "") -> Optional[dict]:
        """
        搜索定理的标准名称。

        尝试多种来源:
        1. Mathlib 文档 (leanprover-community)
        2. Wikipedia 数学定理列表
        3. ProofWiki

        Returns:
            {"name": str, "description": str, "source": str, "url": str}
            或 None（未找到）
        """
        if not self.config.enabled or not AIOHTTP_AVAILABLE:
            return None

        # 尝试从 Lean 代码中提取 Mathlib 定理名
        mathlib_name = self._extract_mathlib_theorem_name(lean_code)
        if mathlib_name:
            return {
                "name": mathlib_name,
                "description": f"Mathlib 定理: {mathlib_name}",
                "source": "mathlib",
                "url": f"https://leanprover-community.github.io/mathlib4_docs/find/#{mathlib_name}",
            }

        # 尝试搜索 loogle (Mathlib 搜索引擎)
        result = await self._search_loogle(theorem_statement)
        if result:
            return result

        # 尝试 ProofWiki
        result = await self._search_proofwiki(theorem_statement)
        if result:
            return result

        return None

    @staticmethod
    def _extract_mathlib_theorem_name(lean_code: str) -> Optional[str]:
        """从 Lean 代码中提取使用的 Mathlib 定理名。"""
        if not lean_code:
            return None
        # 匹配 exact Xxx.yyy.zzz 模式
        match = re.search(r'exact\s+([A-Z][\w.]+)', lean_code)
        if match:
            name = match.group(1)
            if '.' in name and not name.startswith('Eq.'):
                return name
        # 匹配 simp [Xxx.yyy] 中的引理
        match = re.search(r'simp\s*\[([A-Z][\w.]+)', lean_code)
        if match:
            return match.group(1)
        return None

    async def _search_loogle(self, query: str) -> Optional[dict]:
        """通过 Loogle (Mathlib 搜索引擎) 搜索定理。"""
        try:
            session = await self._get_session()
            url = "https://loogle.lean-lang.org/api/search"
            params = {"q": query[:200]}
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    hits = data.get("hits", [])
                    if hits:
                        hit = hits[0]
                        name = hit.get("name", "")
                        doc = hit.get("doc", "")
                        return {
                            "name": name,
                            "description": doc[:200] if doc else f"Mathlib: {name}",
                            "source": "loogle",
                            "url": f"https://leanprover-community.github.io/mathlib4_docs/find/#{name}",
                        }
        except Exception as e:
            logger.debug(f"Loogle 搜索失败: {e}")
        return None

    async def _search_proofwiki(self, query: str) -> Optional[dict]:
        """通过 ProofWiki 搜索定理名称。"""
        try:
            session = await self._get_session()
            url = "https://proofwiki.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query[:100],
                "srnamespace": "0",
                "srlimit": "3",
                "format": "json",
            }
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("query", {}).get("search", [])
                    if results:
                        hit = results[0]
                        title = hit.get("title", "")
                        snippet = hit.get("snippet", "")
                        # 清理 HTML
                        snippet = re.sub(r'<[^>]+>', '', snippet)
                        return {
                            "name": title,
                            "description": snippet[:200],
                            "source": "proofwiki",
                            "url": f"https://proofwiki.org/wiki/{title.replace(' ', '_')}",
                        }
        except Exception as e:
            logger.debug(f"ProofWiki 搜索失败: {e}")
        return None

    # ------------------------------------------------------------------
    #  生命周期
    # ------------------------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
