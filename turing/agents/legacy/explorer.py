"""
[LEGACY] Explorer 智能体 — 仅在 --mode multi 时使用。
推荐架构请参见 skill_based_agent.py + turing/skills/。

原功能：在指定数学领域进行探索性搜索，寻找模式与猜想。

当 Turing 进入"探索模式"时被触发。
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.llm.llm_client import LLMClient
from turing.memory.long_term_memory import LongTermMemory, KnowledgeEntry
from turing.resources.resource_manager import ResourceManager


EXPLORER_SYSTEM_PROMPT = """你是 Explorer，Turing 数学研究团队中的数学探索者。

你的职责是：
1. 在指定数学领域中寻找模式和规律
2. 通过类比、归纳和合情推理提出数学猜想
3. 探索已有定理之间的联系
4. 发现可能的推广或特化方向

工作方法：
- 考察具体的例子和特殊情况
- 寻找数值模式（如生成序列、计算特例）
- 类比不同数学领域的类似结构
- 检查已有定理是否可以推广
- 寻找反例来测试猜想

输出格式：
对于每个发现，请给出：
1. [猜想] 猜想的精确陈述
2. [证据] 支持这个猜想的例子或推理
3. [置信度] 你认为这个猜想为真的概率 (0.0-1.0)
4. [方向] 可能的证明方向建议
5. [联系] 与已知定理或概念的关联"""


class ExplorerAgent(BaseAgent):
    """数学探索者智能体。"""

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        **kwargs,
    ):
        config = agent_config or AgentConfig(
            agent_name="Explorer",
            system_prompt=EXPLORER_SYSTEM_PROMPT,
        )
        if not config.system_prompt:
            config.system_prompt = EXPLORER_SYSTEM_PROMPT

        super().__init__(config, llm_client, resource_manager)
        self.ltm = long_term_memory

    async def _execute(self, task: str, **kwargs) -> Any:
        """
        在指定领域或主题上进行探索。

        Args:
            task: 探索领域或主题描述
            kwargs:
                - depth: 探索深度 (1-5)
                - focus: 探索焦点 (patterns | generalizations | connections)
        """
        depth = kwargs.get("depth", 3)
        focus = kwargs.get("focus", "patterns")

        logger.info(f"[Explorer] 开始探索: {task[:100]}... (深度={depth})")

        # 检索已有知识
        context = ""
        if self.ltm:
            related = self.ltm.search(task, top_k=5)
            if related:
                context = "已有相关知识:\n"
                for r in related:
                    context += f"- [{r['type']}] {r['natural_language'][:150]}\n"

        # 第一阶段：广泛探索
        prompt = f"""请在以下数学领域/主题中进行探索性研究:

领域: {task}
探索焦点: {focus}
{context}

请进行系统性探索：
1. 列举这个领域中的关键概念和已知结论
2. 考察具体的例子和特殊情况
3. 寻找可能的模式或规律
4. 尝试提出至少2个猜想

按照规定的输出格式给出每个发现。"""

        exploration = await self.think(prompt)
        results = {"explorations": [exploration], "conjectures": []}

        # 深度探索迭代
        for d in range(1, depth):
            if self.should_stop():
                break

            deep_prompt = f"""基于之前的探索结果，请进一步深入研究。

之前发现:
{exploration[:2000]}

请：
1. 对最有前景的猜想进行更细致的分析
2. 寻找更多支持或反对的证据
3. 尝试建立不同发现之间的联系
4. 提出新的、更精确的猜想"""

            exploration = await self.think(deep_prompt)
            results["explorations"].append(exploration)

        # 总结猜想
        summary_prompt = f"""请总结你所有的探索结果。

对于每个猜想，给出一个结构化的摘要：
- 猜想陈述（用精确的数学语言）
- 置信度评分 (0.0-1.0)
- 支持证据概要
- 建议的验证方向

所有探索发现:
{chr(10).join(results['explorations'][-2:])}"""

        summary = await self.think(summary_prompt)
        results["summary"] = summary

        # 将猜想存入长期记忆
        if self.ltm:
            entry = KnowledgeEntry(
                type="conjecture",
                natural_language=f"[探索-{task}]\n{summary[:500]}",
                tags=["exploration", task.split()[0] if task else "math"],
                confidence=0.4,
                source="self_proved",
            )
            self.ltm.add(entry)

        logger.info(f"[Explorer] 探索完成，深度={d+1}")
        return results
