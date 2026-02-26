"""
长期记忆（Long-term Memory）— 基于 ChromaDB 的向量数据库 RAG 系统。

存储定理、策略、错误日志和概念关系，支持语义检索和去重。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    logger.warning("chromadb 未安装，长期记忆功能受限。运行: pip install chromadb")

from turing.config import LongTermMemoryConfig, get_config


# ------------------------------------------------------------------
#  知识条目数据结构
# ------------------------------------------------------------------

@dataclass
class KnowledgeEntry:
    """长期记忆中的一条知识。"""
    id: str = ""
    type: str = "theorem"               # theorem | lemma | tactic | error_log | conjecture | concept
    natural_language: str = ""
    lean_code: str = ""
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "self_proved"          # self_proved | mathlib | external
    timestamp: str = ""
    retrieval_count: int = 0
    success_rate_in_application: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = json.dumps(self.tags, ensure_ascii=False)
        d["dependencies"] = json.dumps(self.dependencies, ensure_ascii=False)
        d["metadata"] = json.dumps(self.metadata, ensure_ascii=False)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeEntry":
        tags = d.get("tags", "[]")
        if isinstance(tags, str):
            tags = json.loads(tags)
        deps = d.get("dependencies", "[]")
        if isinstance(deps, str):
            deps = json.loads(deps)
        meta = d.get("metadata", "{}")
        if isinstance(meta, str):
            meta = json.loads(meta)

        return cls(
            id=d.get("id", ""),
            type=d.get("type", "theorem"),
            natural_language=d.get("natural_language", ""),
            lean_code=d.get("lean_code", ""),
            tags=tags,
            dependencies=deps,
            confidence=float(d.get("confidence", 1.0)),
            source=d.get("source", "self_proved"),
            timestamp=d.get("timestamp", ""),
            retrieval_count=int(d.get("retrieval_count", 0)),
            success_rate_in_application=d.get("success_rate_in_application"),
            metadata=meta,
        )

    def content_for_embedding(self) -> str:
        """生成用于 embedding 的文本。"""
        parts = [self.natural_language]
        if self.lean_code:
            parts.append(f"Lean: {self.lean_code[:500]}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(parts)


class LongTermMemory:
    """
    基于 ChromaDB 的长期记忆系统。

    - 存储定理、策略、错误日志、概念图谱
    - 支持语义检索（RAG）
    - 入库前自动去重（余弦相似度 > threshold）
    - 追踪检索次数和应用成功率
    """

    def __init__(self, config: Optional[LongTermMemoryConfig] = None):
        self.config = config or get_config().memory.long_term
        self._client = None
        self._collection = None
        self._initialized = False

    def initialize(self):
        """初始化 ChromaDB 连接。"""
        if chromadb is None:
            logger.error("chromadb 未安装！")
            self._initialized = False
            return

        persist_dir = Path(self.config.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._initialized = True
        count = self._collection.count()
        logger.info(f"长期记忆初始化完成: {count} 条知识已加载")

    def _ensure_init(self):
        if not self._initialized:
            self.initialize()

    # ------------------------------------------------------------------
    #  知识入库
    # ------------------------------------------------------------------

    def add(self, entry: KnowledgeEntry, skip_dedup: bool = False) -> tuple[bool, str]:
        """
        将知识条目添加到长期记忆。

        Returns:
            (added: bool, message: str)
            - added=True 表示新增成功
            - added=False 表示重复未添加（或合并了元数据）
        """
        self._ensure_init()
        if not self._initialized:
            return False, "长期记忆未初始化"

        if not entry.id:
            entry.id = f"{entry.type}_{uuid.uuid4().hex[:8]}"
        if not entry.timestamp:
            entry.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        embed_text = entry.content_for_embedding()

        # 去重检查
        if not skip_dedup:
            is_dup, existing_id = self._check_duplicate(embed_text)
            if is_dup:
                self._merge_metadata(existing_id, entry)
                return False, f"重复内容，已与 {existing_id} 合并元数据"

        meta = entry.to_dict()
        # ChromaDB 元数据值必须是字符串/数字/布尔
        clean_meta = {}
        for k, v in meta.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)

        try:
            self._collection.add(
                ids=[entry.id],
                documents=[embed_text],
                metadatas=[clean_meta],
            )
        except Exception as e:
            # ChromaDB 偶发 readonly 错误，尝试重新连接
            logger.warning(f"长期记忆写入失败，尝试重新连接: {e}")
            try:
                self._initialized = False
                self.initialize()
                self._collection.add(
                    ids=[entry.id],
                    documents=[embed_text],
                    metadatas=[clean_meta],
                )
            except Exception as e2:
                logger.error(f"长期记忆写入重试失败: {e2}")
                return False, f"写入失败: {e2}"

        logger.info(f"长期记忆：已添加 [{entry.type}] {entry.id}: {entry.natural_language[:60]}...")
        return True, f"已添加: {entry.id}"

    def add_theorem(
        self,
        natural_language: str,
        lean_code: str = "",
        tags: Optional[list[str]] = None,
        source: str = "self_proved",
        confidence: float = 1.0,
        theorem_name: str = "",
        area: str = "",
        description: str = "",
        is_novel: bool = False,
        external_url: str = "",
    ) -> tuple[bool, str]:
        """快捷方法：添加定理。

        Args:
            natural_language: 定理的自然语言描述
            lean_code: Lean 4 代码
            tags: 标签列表
            source: 来源
            confidence: 置信度
            theorem_name: 定理名称（如 Nat.add_comm 或自定义命名）
            area: 数学分支（如 number_theory, algebra, topology 等）
            description: 简短描述
            is_novel: 是否为 Mathlib 中不存在的新定理
            external_url: 外部参考链接
        """
        metadata = {}
        if theorem_name:
            metadata["theorem_name"] = theorem_name
        if area:
            metadata["area"] = area
        if description:
            metadata["description"] = description
        if is_novel:
            metadata["is_novel"] = True
        if external_url:
            metadata["external_url"] = external_url

        nl = natural_language
        if theorem_name:
            nl = f"[{theorem_name}] {natural_language}"

        entry = KnowledgeEntry(
            type="theorem",
            natural_language=nl,
            lean_code=lean_code,
            tags=tags or ([area] if area else []),
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
        return self.add(entry)

    def get_proven_theorems(self, area: str = "", limit: int = 50) -> list[dict]:
        """获取已证明的定理，可按领域过滤。

        返回带有 theorem_name 和 lean_code 的定理列表，供 Prover 直接引用。
        """
        self._ensure_init()
        if not self._initialized:
            return []
        try:
            where_filter = {"type": "theorem", "source": "self_proved"}
            if area:
                where_filter = {
                    "$and": [
                        {"type": "theorem"},
                        {"source": "self_proved"},
                    ]
                }
            results = self._collection.get(
                where={"type": "theorem"},
                limit=limit,
            )
            entries = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    if meta.get("source") != "self_proved":
                        continue
                    if area and meta.get("area", "") != area:
                        continue
                    lean_code = meta.get("lean_code", "")
                    if not lean_code:
                        continue
                    entries.append({
                        "id": doc_id,
                        "theorem_name": meta.get("theorem_name", ""),
                        "natural_language": meta.get("natural_language", ""),
                        "lean_code": lean_code,
                        "area": meta.get("area", ""),
                        "description": meta.get("description", ""),
                    })
            return entries
        except Exception as e:
            logger.warning(f"获取已证定理失败: {e}")
            return []

    # 跨分支关联表：定义哪些数学分支之间有知识复用价值
    RELATED_AREAS: dict[str, list[str]] = {
        "linear_algebra": ["algebra", "group_theory", "ring_theory", "field_theory"],
        "group_theory": ["algebra", "ring_theory", "representation_theory"],
        "ring_theory": ["algebra", "field_theory", "algebraic_geometry", "number_theory"],
        "field_theory": ["algebra", "ring_theory", "number_theory", "algebraic_geometry"],
        "measure_theory": ["analysis", "topology", "probability"],
        "probability": ["measure_theory", "analysis", "combinatorics"],
        "algebraic_geometry": ["algebra", "ring_theory", "field_theory", "topology", "category_theory"],
        "algebraic_topology": ["topology", "algebra", "group_theory", "category_theory"],
        "category_theory": ["algebra", "topology", "set_theory", "order_theory"],
        "geometry": ["analysis", "linear_algebra", "topology"],
        "representation_theory": ["algebra", "group_theory", "linear_algebra", "ring_theory"],
        "condensed": ["topology", "category_theory", "algebra"],
        "model_theory": ["logic", "set_theory", "algebra"],
        "information_theory": ["combinatorics", "probability", "number_theory"],
        "dynamics": ["analysis", "topology"],
        "computability": ["logic", "set_theory", "number_theory"],
        "topology": ["analysis", "set_theory", "order_theory"],
        "analysis": ["topology", "order_theory", "algebra"],
        "number_theory": ["algebra", "ring_theory", "combinatorics"],
        "algebra": ["number_theory", "group_theory", "ring_theory", "linear_algebra"],
        "order_theory": ["set_theory", "algebra", "topology"],
        "set_theory": ["logic", "order_theory"],
        "logic": ["set_theory", "computability", "model_theory"],
        "combinatorics": ["number_theory", "probability", "set_theory"],
        "functions": ["set_theory", "analysis", "algebra"],
    }

    def get_cross_branch_theorems(self, area: str, limit: int = 10) -> list[dict]:
        """获取当前分支 + 相关分支的已证定理，用于跨分支知识复用。"""
        related_areas = self.RELATED_AREAS.get(area, [])
        all_areas = [area] + related_areas

        theorems = []
        seen_ids = set()
        for a in all_areas:
            for thm in self.get_proven_theorems(area=a, limit=limit):
                if thm["id"] not in seen_ids:
                    seen_ids.add(thm["id"])
                    theorems.append(thm)
        return theorems[:limit]

    def get_novel_theorems(self) -> list[dict]:
        """获取所有 Mathlib 中不存在的原创定理。"""
        self._ensure_init()
        if not self._initialized:
            return []
        try:
            results = self._collection.get(where={"is_novel": True})
            entries = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    entries.append({
                        "id": doc_id,
                        "theorem_name": meta.get("theorem_name", ""),
                        "natural_language": meta.get("natural_language", ""),
                        "lean_code": meta.get("lean_code", ""),
                        "area": meta.get("area", ""),
                        "description": meta.get("description", ""),
                    })
            return entries
        except Exception as e:
            logger.warning(f"获取原创定理失败: {e}")
            return []

    def add_tactic(
        self,
        description: str,
        lean_code: str = "",
        tags: Optional[list[str]] = None,
        success_rate: Optional[float] = None,
    ) -> tuple[bool, str]:
        """快捷方法：添加策略。"""
        entry = KnowledgeEntry(
            type="tactic",
            natural_language=description,
            lean_code=lean_code,
            tags=tags or [],
            confidence=0.8,
            source="self_proved",
            success_rate_in_application=success_rate,
        )
        return self.add(entry)

    def add_error_log(
        self,
        problem: str,
        error_description: str,
        lean_code: str = "",
        tags: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """快捷方法：添加错误日志。"""
        entry = KnowledgeEntry(
            type="error_log",
            natural_language=f"问题: {problem}\n错误: {error_description}",
            lean_code=lean_code,
            tags=tags or [],
            confidence=1.0,
            source="self_proved",
        )
        return self.add(entry)

    # ------------------------------------------------------------------
    #  检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        type_filter: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        """
        语义检索相关知识。

        Args:
            query: 检索查询文本
            top_k: 返回条目数
            type_filter: 按类型过滤（theorem/tactic/error_log 等）
            min_confidence: 最小置信度过滤
        """
        self._ensure_init()
        if not self._initialized:
            return []

        k = top_k or self.config.default_top_k

        where_filter = None
        if type_filter:
            where_filter = {"type": type_filter}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count() or 1),
                where=where_filter if where_filter else None,
            )
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

        entries = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1 - distance  # cosine distance → similarity

                confidence = float(meta.get("confidence", 0))
                if confidence < min_confidence:
                    continue

                # 更新检索计数
                self._increment_retrieval_count(doc_id)

                entries.append({
                    "id": doc_id,
                    "similarity": similarity,
                    "type": meta.get("type", ""),
                    "natural_language": meta.get("natural_language", ""),
                    "lean_code": meta.get("lean_code", ""),
                    "tags": meta.get("tags", "[]"),
                    "confidence": confidence,
                    "source": meta.get("source", ""),
                    "content": results["documents"][0][i] if results["documents"] else "",
                })

        return entries

    def search_similar_errors(self, error_msg: str, top_k: int = 3) -> list[dict]:
        """搜索类似的错误记录。"""
        return self.search(error_msg, top_k=top_k, type_filter="error_log")

    def search_tactics(self, problem_desc: str, top_k: int = 5) -> list[dict]:
        """搜索可能适用的策略。"""
        return self.search(problem_desc, top_k=top_k, type_filter="tactic")

    # ------------------------------------------------------------------
    #  去重
    # ------------------------------------------------------------------

    def _check_duplicate(self, text: str) -> tuple[bool, str]:
        """检查是否存在语义重复的条目。"""
        if self._collection.count() == 0:
            return False, ""

        results = self._collection.query(
            query_texts=[text],
            n_results=1,
        )

        if results and results["distances"] and results["distances"][0]:
            distance = results["distances"][0][0]
            similarity = 1 - distance
            if similarity >= self.config.similarity_threshold:
                existing_id = results["ids"][0][0]
                return True, existing_id

        return False, ""

    def _merge_metadata(self, existing_id: str, new_entry: KnowledgeEntry):
        """合并重复条目的元数据。"""
        try:
            existing = self._collection.get(ids=[existing_id])
            if existing and existing["metadatas"]:
                meta = existing["metadatas"][0]
                # 合并 tags
                old_tags = json.loads(meta.get("tags", "[]"))
                new_tags = list(set(old_tags + new_entry.tags))
                meta["tags"] = json.dumps(new_tags, ensure_ascii=False)
                # 更新置信度（取最高）
                meta["confidence"] = max(
                    float(meta.get("confidence", 0)), new_entry.confidence
                )
                self._collection.update(ids=[existing_id], metadatas=[meta])
        except Exception as e:
            logger.warning(f"合并元数据失败: {e}")

    def _increment_retrieval_count(self, doc_id: str):
        """增加检索计数。"""
        try:
            existing = self._collection.get(ids=[doc_id])
            if existing and existing["metadatas"]:
                meta = existing["metadatas"][0]
                count = int(meta.get("retrieval_count", 0)) + 1
                meta["retrieval_count"] = count
                self._collection.update(ids=[doc_id], metadatas=[meta])
        except Exception:
            pass  # 非关键操作

    # ------------------------------------------------------------------
    #  管理
    # ------------------------------------------------------------------

    def update_success_rate(self, entry_id: str, success: bool):
        """更新某条策略/定理的应用成功率。"""
        self._ensure_init()
        if not self._initialized:
            return

        try:
            existing = self._collection.get(ids=[entry_id])
            if existing and existing["metadatas"]:
                meta = existing["metadatas"][0]
                rate = meta.get("success_rate_in_application", "")
                count = int(meta.get("retrieval_count", 1)) or 1
                if rate == "" or rate is None:
                    new_rate = 1.0 if success else 0.0
                else:
                    old_rate = float(rate)
                    new_rate = (old_rate * (count - 1) + (1.0 if success else 0.0)) / count
                meta["success_rate_in_application"] = new_rate
                self._collection.update(ids=[entry_id], metadatas=[meta])
        except Exception as e:
            logger.warning(f"更新成功率失败: {e}")

    def get_stats(self) -> dict:
        """获取长期记忆统计信息。"""
        self._ensure_init()
        if not self._initialized:
            return {"initialized": False}

        total = self._collection.count()

        # 按类型统计
        stats = {"total": total, "initialized": True}
        for t in ["theorem", "lemma", "tactic", "error_log", "conjecture", "concept"]:
            try:
                results = self._collection.get(where={"type": t})
                stats[t] = len(results["ids"]) if results["ids"] else 0
            except Exception:
                stats[t] = 0
        return stats

    def get_all_ids(self) -> list[str]:
        """获取所有知识条目的 ID。"""
        self._ensure_init()
        if not self._initialized:
            return []
        try:
            results = self._collection.get()
            return results["ids"] if results["ids"] else []
        except Exception:
            return []

    def delete(self, entry_id: str) -> bool:
        """删除指定条目。"""
        self._ensure_init()
        try:
            self._collection.delete(ids=[entry_id])
            return True
        except Exception as e:
            logger.warning(f"删除失败: {e}")
            return False
