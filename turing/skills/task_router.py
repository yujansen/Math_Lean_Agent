"""
任务路由与规则引擎 — 用低成本手段处理「大模型不该做的事」。

实现原则 A/B/D/E：
  A: 关键决策用大模型，细活用规则 / 小模型
  B: 简单任务走便宜路径，复杂任务才升级到大模型
  D: Token 预算与早停
  E: Lean 编译器本身就是终极验证器

包含：
  - 规则型任务分类器（零 LLM 调用）
  - 规则型 Lean 错误修复（常见模式直接 patch，省 LLM 调用）
  - 难度评估器（决定用大模型还是小模型）
  - 定理命名规则（常见类型直接映射，不调 LLM）
  - TaskState: 共享状态对象，禁止重复读取
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ======================================================================
#  共享任务状态 (原则 C: 禁止重复阅读)
# ======================================================================

@dataclass
class TaskState:
    """
    一个任务的全生命周期共享状态。

    所有中间产物存在此处，后续步骤直接复用，不重查。
    """
    # 输入
    task: str = ""
    area: str = ""
    difficulty: int = 0

    # Step 1: 分类 + 路由
    task_type: str = ""            # prove / disprove / conjecture / explore / organize
    classified_by: str = ""        # "rule" | "llm"
    difficulty_tier: str = ""      # "trivial" | "easy" | "medium" | "hard"

    # 检索结果（一次检索，全程复用）
    context_items: list[dict] = field(default_factory=list)
    context_text: str = ""
    theorem_toolkit: str = ""
    strategies_text: str = ""
    avoid_text: str = ""

    # Step 2: 计划 + 大纲
    plan: str = ""
    outline: str = ""

    # Step 3: 证明
    lean_code: str = ""
    compile_success: bool = False
    compile_error: str = ""
    attempts: int = 0

    # Step 4: 命名 + 审查
    naming: dict = field(default_factory=dict)
    review_score: int = 0
    review_issues: list[str] = field(default_factory=list)

    # Token 预算跟踪 (原则 D)
    llm_calls: int = 0
    max_llm_calls: int = 8         # 硬上限
    tokens_used: int = 0

    def budget_exhausted(self) -> bool:
        return self.llm_calls >= self.max_llm_calls


# ======================================================================
#  规则型任务分类器 (原则 A+B: 不用 LLM 也能判)
# ======================================================================

# 关键词 → 任务类型
_PROVE_KEYWORDS = [
    "证明", "prove", "show that", "verify", "求证", "证",
    "定理", "theorem", "lemma", "引理",
    "for all", "∀", "forall", "对任意", "对所有",
    "if and only if", "iff", "当且仅当",
    "implies", "蕴含", "推出",
]
_DISPROVE_KEYWORDS = ["反驳", "disprove", "反例", "counterexample", "否定", "refute"]
_CONJECTURE_KEYWORDS = ["猜想", "conjecture", "假设", "hypothesis", "猜测"]
_EXPLORE_KEYWORDS = ["探索", "explore", "研究", "investigation", "调查", "分析"]
_ORGANIZE_KEYWORDS = ["整理", "organize", "分类", "归纳总结", "review"]


def classify_by_rules(task: str) -> Optional[str]:
    """
    用关键词和正则规则进行任务分类。
    返回 task_type 字符串，或 None 表示需要 LLM 辅助。

    成本: 0 LLM 调用。
    """
    lower = task.lower().strip()

    # 优先检查明确意图
    for kw in _DISPROVE_KEYWORDS:
        if kw in lower:
            return "disprove"
    for kw in _CONJECTURE_KEYWORDS:
        if kw in lower:
            return "conjecture"
    for kw in _EXPLORE_KEYWORDS:
        if kw in lower:
            return "explore"
    for kw in _ORGANIZE_KEYWORDS:
        if kw in lower:
            return "organize"

    # "prove" 是默认大类：数学表达式 / 等式 / 不等式 → 大概率是证明
    if any(kw in lower for kw in _PROVE_KEYWORDS):
        return "prove"

    # 含有明显的数学结构（等号、不等号、∀、→）→ 默认 prove
    if re.search(r"[=≤≥<>∀∃→⟹⟺]", task):
        return "prove"

    # 无法判断
    return None


# ======================================================================
#  难度评估器 (原则 B: 分层路由)
# ======================================================================

# "一眼可解"的模式 → 直接给出 tactic 建议
_TRIVIAL_PATTERNS: list[tuple[str, str]] = [
    # (regex_on_task, suggested_tactic)
    (r"^\s*\d+\s*[+\-*/]\s*\d+\s*=\s*\d+", "norm_num"),       # 纯数值等式
    (r"n\s*\+\s*0\s*=\s*n|0\s*\+\s*n\s*=\s*n", "omega"),       # n+0=n
    (r"n\s*\*\s*1\s*=\s*n|1\s*\*\s*n\s*=\s*n", "ring"),        # n*1=n
    (r"n\s*\*\s*0\s*=\s*0|0\s*\*\s*n\s*=\s*0", "ring"),        # n*0=0
    (r"a\s*\+\s*b\s*=\s*b\s*\+\s*a", "omega"),                 # 加法交换律
    (r"a\s*\*\s*b\s*=\s*b\s*\*\s*a", "ring"),                   # 乘法交换律
    (r"True", "trivial"),
]

_HARD_INDICATORS = [
    "measure", "probability", "σ-algebra", "topology", "category",
    "functor", "homology", "cohomology", "spectrum", "sheaf",
    "Galois", "algebraic geometry", "representation",
    "condensed", "induction on", "归纳", "递归",
    "测度", "概率", "拓扑", "范畴", "函子", "层",
]


@dataclass
class DifficultyAssessment:
    """难度评估结果。"""
    tier: str                  # "trivial" | "easy" | "medium" | "hard"
    use_light_model: bool      # 用轻量模型/规则即可
    suggested_tactic: str = "" # trivial 时的建议 tactic
    reason: str = ""


def assess_difficulty(task: str) -> DifficultyAssessment:
    """
    评估任务难度（零 LLM 调用）。

    Returns:
        DifficultyAssessment: 含 tier + 是否可用轻量路径
    """
    lower = task.lower()

    # Trivial: 直接匹配到已知 tactic
    for pattern, tactic in _TRIVIAL_PATTERNS:
        if re.search(pattern, task, re.IGNORECASE):
            return DifficultyAssessment(
                tier="trivial", use_light_model=True,
                suggested_tactic=tactic,
                reason=f"模式匹配 → {tactic}",
            )

    # Hard: 包含高级数学术语
    hard_count = sum(1 for ind in _HARD_INDICATORS if ind in lower)
    if hard_count >= 2:
        return DifficultyAssessment(
            tier="hard", use_light_model=False,
            reason=f"检测到 {hard_count} 个高级术语",
        )

    # 任务文本长度也是间接难度指标
    if len(task) < 50:
        return DifficultyAssessment(
            tier="easy", use_light_model=True,
            reason="短任务",
        )
    elif len(task) < 200:
        return DifficultyAssessment(
            tier="medium", use_light_model=False,
            reason="中等长度",
        )
    else:
        return DifficultyAssessment(
            tier="hard", use_light_model=False,
            reason="长任务",
        )


# ======================================================================
#  规则型 Lean 错误修复 (原则 A+E: 能用规则就不调 LLM)
# ======================================================================

def try_rule_based_fix(lean_code: str, error: str) -> Optional[str]:
    """
    尝试用纯规则修复常见 Lean 编译错误。
    返回修复后的代码，或 None 表示需要 LLM。

    成本: 0 LLM 调用。
    """
    # 1. "No goals to be solved" → 删除多余的 tactic 行
    if "no goals to be solved" in error.lower():
        # 找到 "by" 块，尝试逐行缩减
        lines = lean_code.split("\n")
        if len(lines) > 3:
            # 从末尾删除空行和可能多余的 tactic 行
            while lines and lines[-1].strip() == "":
                lines.pop()
            if lines and _is_tactic_line(lines[-1]):
                lines.pop()
                return "\n".join(lines)

    # 2. 缺少 `import Mathlib`
    if "unknown identifier" in error.lower() or "unknown namespace" in error.lower():
        if "import Mathlib" not in lean_code:
            return f"import Mathlib\n\n{lean_code}"

    # 3. Lean 3 → Lean 4 语法修复: `let x := ... in` → `let x := ...`
    if "expected ';' or line break" in error or "expected token" in error:
        fixed = re.sub(r'\blet\s+(.+?)\s*:=\s*(.+?)\s+in\b', r'let \1 := \2',
                        lean_code)
        if fixed != lean_code:
            return fixed

    # 4. 其他复杂错误 → 交给 LLM
    return None


def _is_tactic_line(line: str) -> bool:
    """判断是否是 tactic 行。"""
    stripped = line.strip()
    return bool(stripped) and any(stripped.startswith(t) for t in [
        "simp", "omega", "ring", "norm_num", "decide", "exact",
        "apply", "rw", "intro", "induction", "cases", "constructor",
        "have", "let", "calc", "aesop", "trivial", "rfl",
    ])


# ======================================================================
#  规则型定理命名 (原则 A: 常见模式直接映射)
# ======================================================================

_NAME_PATTERNS: list[tuple[str, dict]] = [
    (r"(\w)\s*\+\s*0\s*=\s*\1|0\s*\+\s*(\w)\s*=\s*\2",
     {"theorem_name": "add_zero_identity", "area": "algebra", "description": "加法零元"}),

    (r"(\w)\s*\*\s*1\s*=\s*\1|1\s*\*\s*(\w)\s*=\s*\2",
     {"theorem_name": "mul_one_identity", "area": "algebra", "description": "乘法幺元"}),

    (r"(\w)\s*\+\s*(\w)\s*=\s*\2\s*\+\s*\1",
     {"theorem_name": "add_comm", "area": "algebra", "description": "加法交换律"}),

    (r"(\w)\s*\*\s*(\w)\s*=\s*\2\s*\*\s*\1",
     {"theorem_name": "mul_comm", "area": "algebra", "description": "乘法交换律"}),
]


def try_rule_based_naming(task: str) -> Optional[dict]:
    """
    尝试用规则为已知模式的定理命名。
    返回 naming dict 或 None。

    成本: 0 LLM 调用。
    """
    for pattern, naming in _NAME_PATTERNS:
        if re.search(pattern, task, re.IGNORECASE):
            return {
                **naming,
                "chinese_name": naming["description"],
                "is_novel": False,
                "tags": [naming["area"]],
            }
    return None


# ======================================================================
#  构造 Trivial 证明代码 (原则 A: 不调 LLM)
# ======================================================================

def try_trivial_lean_code(task: str, theorem_name: str, tactic: str) -> str:
    """
    为 trivial 难度任务直接构造 Lean 代码。

    成本: 0 LLM 调用。
    """
    # 用一个通用模板，tactic 由 assess_difficulty 提供
    return f"""import Mathlib

theorem {theorem_name} : {task} := by
  {tactic}
"""
