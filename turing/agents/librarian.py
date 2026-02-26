"""
Librarian 智能体 — 管理知识库，执行去重、标注、检索优化。

当知识库达到一定规模后定期运行。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.llm.llm_client import LLMClient
from turing.memory.long_term_memory import KnowledgeEntry, LongTermMemory
from turing.resources.resource_manager import ResourceManager


LIBRARIAN_SYSTEM_PROMPT = """你是 Librarian，Turing 数学研究团队中的知识管理专家。

你的职责是：
1. 维护和优化数学知识库的组织结构
2. 发现知识条目之间的关联
3. 识别和合并重复内容
4. 为知识条目添加准确的标签和分类
5. 建议缺失的关键知识点

你管理的知识类型：
- theorem: 已证明的数学定理
- lemma: 辅助引理
- tactic: 证明策略和技巧
- error_log: 失败案例和教训
- conjecture: 未证明的猜想
- concept: 数学概念和定义

你应该像一位严谨的图书馆员一样，确保知识库的条理化和完整性。"""


class LibrarianAgent(BaseAgent):
    """知识库管理智能体。"""

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        **kwargs,
    ):
        config = agent_config or AgentConfig(
            agent_name="Librarian",
            system_prompt=LIBRARIAN_SYSTEM_PROMPT,
        )
        if not config.system_prompt:
            config.system_prompt = LIBRARIAN_SYSTEM_PROMPT

        super().__init__(config, llm_client, resource_manager)
        self.ltm = long_term_memory or LongTermMemory()

    async def _execute(self, task: str, **kwargs) -> Any:
        """
        执行知识库维护任务。

        Args:
            task: 维护任务类型
                  "organize" - 整体组织优化
                  "find_connections" - 发现知识关联
                  "audit" - 完整性审计
                  "tag" - 标签优化
        """
        task_type = kwargs.get("task_type", task)
        logger.info(f"[Librarian] 执行知识维护: {task_type}")

        stats = self.ltm.get_stats()

        if task_type == "organize" or task_type == "整理知识库":
            return await self._organize(stats)
        elif task_type == "find_connections":
            return await self._find_connections(stats)
        elif task_type == "audit":
            return await self._audit(stats)
        elif task_type == "tag":
            return await self._optimize_tags(stats)
        else:
            # 通用任务
            return await self._general_maintenance(task, stats)

    async def _organize(self, stats: dict) -> dict:
        """整体组织优化。"""
        # 获取知识库概览
        all_ids = self.ltm.get_all_ids()
        sample_entries = []
        for entry_id in all_ids[:20]:  # 采样前20条
            results = self.ltm.search(entry_id, top_k=1)
            if results:
                sample_entries.append(results[0])

        overview = f"知识库统计: {json.dumps(stats, ensure_ascii=False)}\n\n"
        overview += "示例条目:\n"
        for e in sample_entries[:10]:
            overview += f"- [{e.get('type', '?')}] {e.get('natural_language', '')[:100]}\n"

        prompt = f"""请分析知识库的当前组织状况并提出优化建议。

{overview}

请评估以下方面：
1. 知识的覆盖面是否均衡（各数学领域的分布）
2. 标签体系是否一致
3. 是否有明显的缺口或冗余
4. 组织结构的改进建议"""

        analysis = await self.think(prompt)
        return {"type": "organization_report", "analysis": analysis, "stats": stats}

    async def _find_connections(self, stats: dict) -> dict:
        """发现知识条目之间的关联。"""
        all_ids = self.ltm.get_all_ids()
        connections = []

        # 对每对知识寻找关联（限制规模）
        sample_ids = all_ids[:15]
        for i, id1 in enumerate(sample_ids):
            results1 = self.ltm.search(id1, top_k=1)
            if not results1:
                continue

            # 用该条目的内容搜索相关条目
            related = self.ltm.search(
                results1[0].get("natural_language", ""), top_k=3
            )
            for r in related:
                if r["id"] != id1 and r["similarity"] > 0.5:
                    connections.append({
                        "from": id1,
                        "to": r["id"],
                        "similarity": r["similarity"],
                        "from_desc": results1[0].get("natural_language", "")[:80],
                        "to_desc": r.get("natural_language", "")[:80],
                    })

        # 让 LLM 分析这些关联
        if connections:
            conn_text = "\n".join(
                f"  {c['from_desc']} ↔ {c['to_desc']} (相似度:{c['similarity']:.2f})"
                for c in connections[:10]
            )
            analysis = await self.think(
                f"以下是知识库中发现的关联，请分析它们的数学关系：\n{conn_text}"
            )
        else:
            analysis = "未发现显著关联"

        return {"connections": connections, "analysis": analysis}

    async def _audit(self, stats: dict) -> dict:
        """知识库完整性审计。"""
        prompt = f"""请对知识库进行完整性审计。

当前统计: {json.dumps(stats, ensure_ascii=False)}

请检查：
1. 是否有低置信度的条目需要验证
2. 是否有过时或需要更新的信息
3. 关键数学领域是否有覆盖
4. 知识之间的依赖关系是否完整
5. 建议接下来应该补充哪些知识"""

        report = await self.think(prompt)
        return {"type": "audit_report", "report": report, "stats": stats}

    async def _optimize_tags(self, stats: dict) -> dict:
        """优化标签体系。"""
        # 收集现有标签
        all_ids = self.ltm.get_all_ids()
        all_tags = set()

        for entry_id in all_ids[:30]:
            results = self.ltm.search(entry_id, top_k=1)
            if results:
                tags_str = results[0].get("tags", "[]")
                try:
                    tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
                    if isinstance(tags, list):
                        all_tags.update(tags)
                except Exception:
                    pass

        prompt = f"""请优化以下知识库标签体系。

当前标签: {sorted(all_tags)}

请提出：
1. 标签分类的层次结构建议
2. 应该合并的同义标签
3. 应该新增的关键标签
4. 标签命名规范建议"""

        suggestion = await self.think(prompt)
        return {"current_tags": sorted(all_tags), "suggestion": suggestion}

    async def _general_maintenance(self, task: str, stats: dict) -> dict:
        """通用维护任务。"""
        prompt = f"""作为知识库管理员，请执行以下任务：

任务: {task}
知识库统计: {json.dumps(stats, ensure_ascii=False)}

请给出详细的执行报告。"""

        report = await self.think(prompt)
        return {"type": "general", "task": task, "report": report}
