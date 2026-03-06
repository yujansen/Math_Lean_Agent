"""
数学技能集 — 将原多智能体的各项能力封装为可复用的 Skill。

合并策略（LLM 调用优化）：
  原流程                          合并后
  ─────────────                   ──────────
  classify (1) + plan (1)         → analyze_and_plan (1)
    + proof outline (1)
  Prover 初次生成 (1)             → lean_prove (1)
  Prover 错误修复 (1-4)           → lean_fix (1-4)
  命名 (1) + Critic 审查 (1-2)   → name_and_review (1)
  Evaluator 评估 (1)             → evaluate (0-1, 可选/批量)
  ─────────────                   ──────────
  合计: 9-12 次                   → 3-7 次
"""

from __future__ import annotations

from turing.skills.skill_registry import (
    Skill,
    SkillRegistry,
    parse_classify_plan_outline,
    parse_json_response,
    parse_lean_code,
    parse_name_and_review,
)


# ======================================================================
#  技能 1: 分析 + 计划 + 证明大纲（合并 3 → 1）
# ======================================================================

ANALYZE_AND_PLAN_TEMPLATE = """请完成以下三项工作，在同一个回复中依次给出。

## 任务
{task}

## 已有知识上下文
{context}

## 推荐策略
{strategies}

## 应避免的策略
{avoid_list}

---

### 第一部分：任务分类
判断任务类型，只写一个标签：
任务类型: prove / disprove / conjecture / explore / organize

### 第二部分：执行计划
给出简洁的分步计划（3-5步），每步标注预期结果。

### 第三部分：证明大纲
如果任务类型是 prove，请给出自然语言证明思路。
优先考虑 Lean 的 simp/omega/ring/norm_num 等自动化 tactic 是否可以一步解决。
只有这些都不行时才考虑手动归纳或分步证明。

{theorem_toolkit}"""


# ======================================================================
#  技能 2: Lean 证明生成
# ======================================================================

LEAN_PROVE_TEMPLATE = """请将以下数学命题形式化为 Lean 4 代码并完成证明。

命题: {task}
定理名称: {theorem_name}
证明思路: {hints}

{context}
{theorem_toolkit}

⚠️ 策略要求（必须遵守）：
1. 必须 `import Mathlib`
2. 需要时用 open 打开命名空间（如 open MeasureTheory / open CategoryTheory）
3. 先尝试一行自动化 tactic（按顺序：simp, omega, ring, norm_num, decide, exact?, aesop）
4. 如果一行 tactic 不够，尝试直接引用 Mathlib 定理（如 exact Nat.add_comm ..）
5. 仅在以上全部失败时才使用 induction，且归纳分支内也优先用 simp/omega
6. 绝对不要在证明目标已解决后添加多余的 tactic 行
7. 证明越短越好

请给出完整的 Lean 4 代码，用 ```lean 和 ``` 包围。"""


# ======================================================================
#  技能 3: Lean 错误修复
# ======================================================================

LEAN_FIX_TEMPLATE = """Lean 4 编译失败。请修正代码。

当前代码:
```lean
{lean_code}
```

编译错误:
{error_info}
{error_guidance}

修正要求：
1. 优先尝试用 simp/omega/ring/norm_num 一行替换出错的部分
2. 不要在证明已完成后添加多余的行
3. 确保 `import Mathlib` 在代码开头
4. 给出修正后的完整代码，用 ```lean 和 ``` 包围。"""


# ======================================================================
#  技能 4: 定理命名 + 快速审查（合并 2-3 → 1）
# ======================================================================

NAME_AND_REVIEW_TEMPLATE = """请为以下已验证的定理完成两项工作。

## 定理陈述
{task}

## Lean 4 代码
```lean
{lean_code}
```

## 证明思路
{proof_outline}

{web_hint}

---

### 第一部分：命名与分类
以 JSON 格式回复：
{{
  "theorem_name": "标准英文名称（如 Nat.add_comm）",
  "chinese_name": "中文名称",
  "area": "数学分支",
  "description": "一句话描述（中文）",
  "is_novel": true/false,
  "tags": ["标签1", "标签2"]
}}

### 第二部分：快速质量审查
对证明给出 1-10 分评分，指出任何明显问题或可改进之处（2-3 句话即可）。
评分: <分数>"""


# ======================================================================
#  技能 5: 探索
# ======================================================================

EXPLORE_TEMPLATE = """请在以下数学领域/主题中进行探索性研究。

领域: {task}
探索焦点: {focus}
探索深度: {depth}

{context}

请进行系统性探索：
1. 列举关键概念和已知结论
2. 考察具体的例子和特殊情况
3. 寻找模式或规律
4. 提出至少2个猜想

对于每个猜想，给出：
- [猜想] 精确陈述
- [证据] 支持的例子或推理
- [置信度] 为真的概率 (0.0-1.0)
- [方向] 可能的证明方向"""


# ======================================================================
#  技能 6: 深度探索迭代
# ======================================================================

EXPLORE_DEEPER_TEMPLATE = """基于之前的探索结果，请进一步深入研究。

之前发现:
{previous_findings}

请：
1. 对最有前景的猜想进行更细致的分析
2. 寻找更多支持或反对的证据
3. 建立不同发现之间的联系
4. 提出新的、更精确的猜想"""


# ======================================================================
#  技能 7: 探索总结
# ======================================================================

EXPLORE_SUMMARY_TEMPLATE = """请总结所有探索结果。

所有探索发现:
{all_findings}

对于每个猜想，给出结构化摘要：
- 猜想陈述（精确数学语言）
- 置信度评分 (0.0-1.0)
- 支持证据概要
- 建议的验证方向"""


# ======================================================================
#  技能 8: 系统评估（按需 / 批量）
# ======================================================================

EVALUATE_BATCH_TEMPLATE = """请对以下一批任务结果作简要评估。

## 统计概览
总数: {total}, 成功: {successes}, 失败: {failures}
成功率: {success_rate}
类型分布: {type_distribution}

## 最近的结果摘要
{recent_summaries}

请给出 JSON 格式的简要评估：
{{
  "overall_score": 1-10,
  "strengths": ["优势"],
  "weaknesses": ["弱点"],
  "proposals": [{{"priority": "high|medium|low", "description": "建议"}}],
  "should_evolve": true/false
}}"""


# ======================================================================
#  技能 9: 猜想生成
# ======================================================================

CONJECTURE_TEMPLATE = """基于以下方向，请提出数学猜想并给出支持证据。

方向: {task}

{context}

请给出：
1. 至少 2 个猜想的精确陈述
2. 每个猜想的支持证据或类比推理
3. 置信度评估"""


# ======================================================================
#  技能 10: 反驳 / 找反例
# ======================================================================

DISPROVE_TEMPLATE = """请尝试反驳或找到以下命题的反例。

命题: {task}

{context}

请系统地尝试：
1. 检查边界情况和退化情况
2. 构造具体的反例
3. 如果无法反驳，分析命题可能为真的原因"""


# ======================================================================
#  技能 11: 反思
# ======================================================================

REFLECTION_TEMPLATE = """请对以下阶段的工作进行反思总结。

## 阶段统计
{stats}

## 典型成功案例
{success_examples}

## 典型失败案例
{failure_examples}

请分析：
1. 最有效的策略模式
2. 反复出现的失败原因
3. 需要重点改进的领域
4. 下阶段的工作建议

请给出 JSON 格式：
{{
  "effective_strategies": ["策略"],
  "recurring_failures": ["失败模式"],
  "weak_areas": ["薄弱领域"],
  "next_focus": ["下阶段重点"],
  "skill_adjustments": ["技能调整建议"]
}}"""


# ======================================================================
#  注册所有技能
# ======================================================================

def register_all_skills(registry: SkillRegistry):
    """将所有内置数学技能注册到 registry。"""

    registry.register(Skill(
        name="analyze_and_plan",
        description="任务分类 + 执行计划 + 证明大纲（三合一）",
        prompt_template=ANALYZE_AND_PLAN_TEMPLATE,
        response_parser=parse_classify_plan_outline,
        merge_group="planning",
        temperature=0.3,
    ))

    registry.register(Skill(
        name="lean_prove",
        description="生成 Lean 4 形式化证明代码",
        prompt_template=LEAN_PROVE_TEMPLATE,
        response_parser=parse_lean_code,
        temperature=0.4,
    ))

    registry.register(Skill(
        name="lean_fix",
        description="修复 Lean 4 编译错误",
        prompt_template=LEAN_FIX_TEMPLATE,
        response_parser=parse_lean_code,
        temperature=0.3,
        use_light=True,  # 错误修复可用轻量模型
    ))

    registry.register(Skill(
        name="name_and_review",
        description="定理命名 + 分类 + 快速质量审查（合并）",
        prompt_template=NAME_AND_REVIEW_TEMPLATE,
        response_parser=parse_name_and_review,
        temperature=0.2,
        max_tokens=600,
        use_light=True,  # 命名+审查可用轻量模型
    ))

    registry.register(Skill(
        name="explore",
        description="数学领域探索",
        prompt_template=EXPLORE_TEMPLATE,
        temperature=0.8,
    ))

    registry.register(Skill(
        name="explore_deeper",
        description="深度探索迭代",
        prompt_template=EXPLORE_DEEPER_TEMPLATE,
        temperature=0.7,
    ))

    registry.register(Skill(
        name="explore_summary",
        description="探索结果总结",
        prompt_template=EXPLORE_SUMMARY_TEMPLATE,
        temperature=0.5,
    ))

    registry.register(Skill(
        name="evaluate_batch",
        description="批量任务评估",
        prompt_template=EVALUATE_BATCH_TEMPLATE,
        response_parser=parse_json_response,
        temperature=0.3,
        use_light=True,  # 评估可用轻量模型
    ))

    registry.register(Skill(
        name="conjecture",
        description="数学猜想生成",
        prompt_template=CONJECTURE_TEMPLATE,
        temperature=0.8,
    ))

    registry.register(Skill(
        name="disprove",
        description="反驳命题 / 找反例",
        prompt_template=DISPROVE_TEMPLATE,
        temperature=0.6,
    ))

    registry.register(Skill(
        name="reflect",
        description="阶段性反思",
        prompt_template=REFLECTION_TEMPLATE,
        response_parser=parse_json_response,
        temperature=0.4,
    ))
