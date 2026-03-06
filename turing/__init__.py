"""
Turing — 基于 Lean 4 的数学研究智能体系统。

一个运行在本地 Qwen3-coder:30b 模型上的自主数学研究智能体，
利用 Lean 4 定理证明器持续扩展可形式化验证的数学知识边界。

模块结构::

    turing.agents      — 智能体（SkillBasedTuringAgent / legacy TuringAgent）
    turing.skills      — 技能系统（SkillRegistry / TaskRouter / 11 项数学技能）
    turing.llm         — LLM 客户端（Ollama / OpenAI API）
    turing.lean        — Lean 4 接口（编译 / 错误解析）
    turing.memory      — 三层记忆（Working / LongTerm / Persistent）
    turing.evolution   — 自我演化（经验提取 / 反思引擎）
    turing.resources   — 资源管理（GPU / RAM 监控）
    turing.web         — 网络层（题目抓取 / Loogle 搜索）
    turing.config      — 配置管理（YAML → dataclass 单例）
"""

__version__ = "2.1.0"
__agent_name__ = "Turing"
