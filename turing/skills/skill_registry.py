"""
技能注册表 — 定义 Skill 数据结构和全局注册 / 查找机制。

每个 Skill 包含：
  - name: 唯一标识
  - description: 简短说明（用于日志和调试）
  - prompt_template: 包含 {占位符} 的 prompt 模板
  - response_parser: 可选的结果解析函数 (str → dict)
  - merge_group: 可选的合并组标识，同组技能可在一次调用中合并执行
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Skill:
    """一项可调用的技能。"""

    name: str
    description: str
    prompt_template: str
    response_parser: Optional[Callable[[str], dict]] = None
    merge_group: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    use_light: bool = False  # True → 优先用轻量模型（原则 F）

    def build_prompt(self, **kwargs) -> str:
        """用 kwargs 填充 prompt 模板，跳过缺失的占位符。"""
        prompt = self.prompt_template
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def parse_response(self, raw: str) -> dict:
        """解析 LLM 原始响应。"""
        if self.response_parser:
            return self.response_parser(raw)
        return {"response": raw}


class SkillRegistry:
    """全局技能注册表。"""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        """注册一个技能。"""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def get_by_group(self, group: str) -> list[Skill]:
        return [s for s in self._skills.values() if s.merge_group == group]


# ---------------------------------------------------------------------------
#  通用响应解析器
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict:
    """从 LLM 响应中提取 JSON 对象。"""
    # 先尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 代码块
    m = re.search(r"```json\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取 { ... } 最外层
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return {"raw": raw}


def parse_lean_code(raw: str) -> dict:
    """从 LLM 响应中提取 Lean 代码块。"""
    patterns = [
        r"```lean4?\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return {"lean_code": match.group(1).strip(), "raw": raw}

    # 尝试识别 import/theorem 等关键词开头的内容
    lines = raw.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if any(line.strip().startswith(kw) for kw in [
            "import", "theorem", "def", "lemma", "example",
            "open", "namespace", "section", "#check",
        ]):
            in_code = True
        if in_code:
            code_lines.append(line)

    code = "\n".join(code_lines).strip()
    return {"lean_code": code, "raw": raw} if code else {"lean_code": "", "raw": raw}


def parse_classify_plan_outline(raw: str) -> dict:
    """解析合并的分类+计划+证明大纲响应。"""
    result: dict[str, Any] = {"raw": raw}

    # 提取任务类型
    for t in ["prove", "disprove", "conjecture", "explore", "organize"]:
        pattern = rf"任务类型[：:]\s*{t}"
        if re.search(pattern, raw, re.IGNORECASE):
            result["task_type"] = t
            break
    if "task_type" not in result:
        # 回退：在前100字符中搜索
        head = raw[:200].lower()
        for t in ["prove", "disprove", "conjecture", "explore", "organize"]:
            if t in head:
                result["task_type"] = t
                break
        else:
            result["task_type"] = "prove"

    # 提取执行计划（标号列表）
    plan_lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if re.match(r"^(\d+[\.\)、]|[-•])\s+", stripped):
            plan_lines.append(stripped)
    result["plan"] = "\n".join(plan_lines) if plan_lines else raw[:500]

    # 提取证明大纲（## 证明大纲 或 ## Proof Outline 之后的内容）
    outline_match = re.search(
        r"(?:证明(?:大纲|思路)|proof\s*outline)[：:\s]*\n(.*?)(?=\n##|\Z)",
        raw, re.IGNORECASE | re.DOTALL,
    )
    result["outline"] = outline_match.group(1).strip() if outline_match else ""

    return result


def parse_name_and_review(raw: str) -> dict:
    """解析合并的命名+审查响应。"""
    result: dict[str, Any] = {"raw": raw}

    # 提取 JSON 部分（命名信息）
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        try:
            naming = json.loads(json_match.group())
            result["naming"] = naming
        except json.JSONDecodeError:
            result["naming"] = {}
    else:
        result["naming"] = {}

    # 提取评分
    score_match = re.search(r"(?:评分|score)[：:]\s*(\d+)", raw, re.IGNORECASE)
    result["score"] = int(score_match.group(1)) if score_match else 7

    # 提取问题列表
    issues = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if any(kw in stripped for kw in ["问题", "错误", "存疑", "issue", "problem"]):
            issues.append(stripped)
    result["issues"] = issues
    result["passed"] = result["score"] >= 7

    return result
