"""
[LEGACY] Critic 智能体 — 仅在 --mode multi 时使用。
推荐架构请参见 skill_based_agent.py + turing/skills/。

原功能：严格审查证明草稿，寻找逻辑漏洞和技术错误。

审查维度：逻辑正确性、前提完备性、特殊情况覆盖、Lean 代码质量、推广潜力。
输出 1–10 分评分和结构化审查报告。
"""

from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger

from turing.agents.base_agent import AgentConfig, BaseAgent
from turing.llm.llm_client import LLMClient
from turing.resources.resource_manager import ResourceManager


CRITIC_SYSTEM_PROMPT = """你是 Critic，Turing 数学研究团队中的审查专家。

你的职责是：严格审查数学证明和 Lean 代码，寻找逻辑漏洞、技术错误和可改进之处。

审查维度：
1. **逻辑正确性**：推理链是否完整？是否有跳步？是否有循环论证？
2. **前提检查**：所有使用的前提是否已经证明或明确假设？
3. **特殊情况**：是否遗漏了边界情况或退化情况？
4. **Lean 代码质量**：代码是否简洁、可读？是否可以用更优雅的 tactic？
5. **推广可能性**：结论能否进一步加强？前提能否进一步放宽？

输出格式：
## 审查报告
- **整体评分**: [1-10]
- **逻辑正确性**: [通过/存疑/错误] + 详细说明
- **发现的问题**: [编号列表]
- **改进建议**: [编号列表]
- **结论**: [通过/需要修改/拒绝]

你必须保持严格、客观的态度。宁可误报问题，也不要放过潜在的错误。"""


class CriticAgent(BaseAgent):
    """证明审查智能体。"""

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
        resource_manager: Optional[ResourceManager] = None,
        **kwargs,
    ):
        config = agent_config or AgentConfig(
            agent_name="Critic",
            system_prompt=CRITIC_SYSTEM_PROMPT,
        )
        if not config.system_prompt:
            config.system_prompt = CRITIC_SYSTEM_PROMPT

        super().__init__(config, llm_client, resource_manager)

    async def _execute(self, task: str, **kwargs) -> Any:
        """
        审查证明。

        Args:
            task: 待审查的证明内容
            kwargs:
                - lean_code: Lean 代码
                - natural_language_proof: 自然语言证明
                - theorem_statement: 定理陈述
        """
        lean_code = kwargs.get("lean_code", "")
        nl_proof = kwargs.get("natural_language_proof", "")
        theorem = kwargs.get("theorem_statement", "")

        logger.info(f"[Critic] 开始审查: {task[:100]}...")

        prompt = f"""请严格审查以下数学证明。

## 定理
{theorem or task}

## 自然语言证明
{nl_proof or '(未提供)'}

## Lean 4 代码
```lean
{lean_code or '(未提供)'}
```

请按照审查规范给出详细的审查报告。特别注意：
1. 每一步推理是否有充分的依据
2. 是否有遗漏的情况需要分别讨论
3. 使用的引理/定理是否都已经被证明
4. Lean 代码是否完整且可编译"""

        review = await self.think(prompt)

        # 如果发现问题，进一步分析
        if any(kw in review.lower() for kw in ["存疑", "错误", "问题", "需要修改", "拒绝"]):
            followup = await self.think(
                f"你发现了一些问题。请给出具体的修复建议：\n"
                f"1. 对于每个问题，建议如何修正\n"
                f"2. 如果需要额外的引理，请指出\n"
                f"3. 如果 Lean 代码有问题，给出修正后的代码片段"
            )
            review += f"\n\n## 修复建议\n{followup}"

        # 解析评分
        score = self._extract_score(review)

        return {
            "review": review,
            "score": score,
            "passed": score >= 7,
            "needs_revision": 4 <= score < 7,
            "rejected": score < 4,
        }

    @staticmethod
    def _extract_score(review: str) -> int:
        """从审查报告中提取评分（1–10）。"""
        patterns = [
            r"整体评分[：:]\s*\[?(\d+)\]?",
            r"评分[：:]\s*(\d+)",
            r"\[(\d+)/10\]",
            r"(\d+)\s*/\s*10",
        ]
        for pattern in patterns:
            match = re.search(pattern, review)
            if match:
                score = int(match.group(1))
                return min(10, max(1, score))
        return 5  # 默认中等
