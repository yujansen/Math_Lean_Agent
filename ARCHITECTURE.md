# Turing 系统架构 (v2.1.0)

本文档详细描述 Turing 数学研究智能体的内部架构、数据流和设计决策。

---

## 整体架构

```
用户输入 (CLI / Web UI)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   SkillBasedTuringAgent                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Step -1: KB 快速路径 (_try_kb_proof)                   │  │
│  │   ChromaDB 语义检索 → 直接复用已证明定理的 Lean 代码     │  │
│  └────────────────────────┬──────────────────────────────┘  │
│               未命中 ↓                                       │
│  ┌────────────────────────┴──────────────────────────────┐  │
│  │ Step 0: LLM 直接证明 (_direct_llm_proof)              │  │
│  │   1 次 LLM 调用 → Lean 编译 → 通过即完成               │  │
│  └────────────────────────┬──────────────────────────────┘  │
│               未通过 ↓ 降级                                   │
│  ┌────────────────────────┴──────────────────────────────┐  │
│  │ TaskRouter 规则引擎                                    │  │
│  │   classify / fix / name / trivial (零 LLM)            │  │
│  └────────────────────────┬──────────────────────────────┘  │
│               复杂任务 ↓                                     │
│  ┌────────────────────────┴──────────────────────────────┐  │
│  │ SkillRegistry (11 项数学技能)                          │  │
│  │   analyze_and_plan → lean_prove → lean_fix → ...      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │                              │
    ▼                              ▼
┌──────────┐               ┌──────────────┐
│ Lean 4   │               │ 三层记忆系统  │
│ 编译验证  │               │ Working/LT/PM │
└──────────┘               └──────────────┘
```

---

## 核心模块

### 1. 智能体层 (`turing/agents/`)

| 文件 | 职责 |
|------|------|
| `base_agent.py` | 抽象基类，定义生命周期（initialize/shutdown）、LLM 通信、资源控制 |
| `skill_based_agent.py` | **v2 主智能体**，集成技能系统、三层记忆、KB 快速路径 |
| `turing_agent.py` | v1 多智能体调度器（`--mode multi`），通过 AgentFactory 创建子智能体 |
| `agent_factory.py` | 智能体工厂，按配置创建 Prover/Critic/Explorer 等实例 |
| `legacy/` | v1 子智能体（architect/critic/evaluator/explorer/librarian/prover/scout） |

**设计决策**: v2 将 v1 的 7 个独立 Agent 实例合并为 1 个 SkillBasedTuringAgent +
11 个 Skill 定义。每个 Skill = prompt 模板 + 响应解析器，无需独立 Agent 实例。
这使单次证明任务的 LLM 调用从 9-12 次降至平均 2.6 次。

### 2. 技能系统 (`turing/skills/`)

| 文件 | 职责 |
|------|------|
| `skill_registry.py` | `Skill` 数据结构（name/prompt_template/use_light）+ `SkillRegistry` 注册表 |
| `math_skills.py` | 11 项数学技能定义及 prompt 模板 |
| `task_router.py` | 规则引擎：`TaskState` 共享状态、零 LLM 分类/修复/命名 |

**六项优化原则**:

| 原则 | 说明 | 实现 |
|------|------|------|
| A 大脑+小手 | 关键决策用大模型，细活用规则/小模型 | `task_router.py` |
| B 分层路由 | 简单任务走零 LLM 路径，复杂任务才升级 | `skill_based_agent.py` |
| C 共享 TaskState | 所有中间产物存一处，禁止重复读取 | `task_router.TaskState` |
| D Token 预算+早停 | 编译成功即停，不浪费 token | `_execute_proof()` |
| E 编译器即验证器 | Lean 编译器替代 Critic 辩论 | `lean_interface.py` |
| F 大小模型路由 | 命名/修复/评估等简单任务用轻量模型 | `Skill.use_light` |

### 3. LLM 客户端 (`turing/llm/`)

- **统一接口**: 同时支持 Ollama 和 OpenAI 兼容 API
- **异步通信**: 基于 `httpx` / `openai` 的 async 调用
- **大小模型切换**: 通过 `use_light` 参数决定使用主模型或轻量模型
- **Token 统计**: 每次调用记录 token 消耗

### 4. Lean 接口 (`turing/lean/`)

- **Lean 4 编译**: 将生成的代码写入临时 `.lean` 文件，调用 `lean` 编译
- **错误解析**: 结构化提取编译错误（行号/错误类型/消息）
- **Lake 管理**: 自动检测 Mathlib 可用性，管理 Lake 项目
- **超时控制**: `compile_timeout` 默认 300s（Mathlib 首次导入需要较长时间）

### 5. 三层记忆系统 (`turing/memory/`)

```
┌──────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│  Working Memory  │  │  Long-Term Memory  │  │ Persistent Memory  │
│  （工作记忆）     │  │  （长期记忆）       │  │  （持久记忆）       │
├──────────────────┤  ├────────────────────┤  ├────────────────────┤
│ 存储: 内存       │  │ 存储: ChromaDB     │  │ 存储: SQLite       │
│ 生命周期: 会话   │  │ 生命周期: 永久     │  │ 生命周期: 永久     │
│ 用途: 推理步骤   │  │ 用途: 定理/策略RAG │  │ 用途: 元认知/日志  │
│ 特性: 带压缩     │  │ 特性: 语义检索     │  │ 特性: 关系查询     │
└──────────────────┘  └────────────────────┘  └────────────────────┘
```

- **Working Memory**: 维护当前任务的推理链、假设、Lean 编译结果
- **Long-Term Memory (ChromaDB)**:
  - 嵌入模型: `all-MiniLM-L6-v2`（384 维）
  - 中文搜索增强: `_ZH_EN_KEYWORDS` 映射 + `_augment_query_for_search()`
  - 存储: 已证明定理（含 Lean 代码）、成功策略、错误模式
- **Persistent Memory (SQLite)**:
  - 任务日志: 每次 `process_task` 的输入、结果、耗时
  - 反思记录: 阶段性反思报告和改进计划
  - 经验统计: 各领域成功率、失败模式

### 6. 自我演化 (`turing/evolution/`)

```
任务完成 → ExperienceManager.extract() → Long-Term Memory
                    ↓
            每 20 任务 / 24h
                    ↓
        ReflectionEngine.reflect()
                    ↓
        弱点分析 → 改进计划 → 技能参数调整
```

### 7. 资源管理 (`turing/resources/`)

- **实时监控**: CPU / RAM / GPU 利用率（`psutil` + `GPUtil`）
- **Apple Silicon 适配**: 检测 M1–M4 共享内存架构
- **三级策略**: HIGH（≥16GB GPU, ≥32GB RAM）/ MEDIUM / LOW
- **动态并发**: 根据资源等级调整并行任务数

### 8. Web 层

分为两部分:

| 模块 | 位置 | 职责 |
|------|------|------|
| 题目抓取 | `turing/web/problem_scraper.py` | Loogle 搜索、ProofWiki 抓取 |
| Web 前端 | `web/app.py` + `web/static/` | FastAPI REST+SSE 后端 + 浏览器 UI |

---

## 证明流水线详解

一个典型的证明任务 `"证明: 对任意自然数 a 和 b，a + b = b + a"` 的处理流程:

1. **KB 快速路径 (Step -1)**: 查询 ChromaDB 是否已有相似定理的 Lean 代码
   - 使用 `_augment_query_for_search()` 扩展中文关键词为英文
   - 若找到相似度 ≥ 0.5 的结果且含 `lean_code`，直接编译验证
   - 命中: 0 LLM 调用，直接完成

2. **LLM 直接证明 (Step 0)**: 用 1 次 LLM 调用尝试生成 Lean 代码
   - 拼接 KB 参考上下文 + 任务描述
   - 成功: 1 LLM 调用即完成
   - 失败: 记录错误信息，降级到完整流程

3. **规则分类**: TaskRouter 尝试零 LLM 分类
   - 关键词匹配: "任意" → `∀`、"加法" → `nat/int`
   - 高置信度: 跳过 LLM 分类
   - 低置信度: 调用 `analyze_and_plan` 技能 (1 LLM)

4. **难度评估 → 证明生成**:
   - trivial: 规则生成代码 (0 LLM)
   - easy/medium/hard: `lean_prove` 技能 (1 LLM)

5. **编译验证 + 修复循环**:
   - Lean 编译 → 成功则早停
   - 失败: 先尝试规则修复 (0 LLM)，再尝试 `lean_fix` 技能 (light LLM)
   - 最多 `max_retries` 次

6. **命名 + 存储**:
   - `name_and_review` 技能 (light LLM)
   - 存入 ChromaDB + SQLite + theorem_library.json

---

## 配置系统

配置通过 `config.yaml` 加载，映射到层次化 dataclass:

```
TuringConfig
├── SystemConfig      (name, version, log_level, data_dir)
├── LLMConfig         (provider, model, light_model, temperature, ...)
├── LeanConfig        (executable, project_dir, compile_timeout, ...)
├── MemoryConfig
│   ├── WorkingMemoryConfig
│   ├── LongTermMemoryConfig
│   └── PersistentMemoryConfig
├── AgentsConfig      (mode: skill|multi, max_concurrent, ...)
├── ResourcesConfig   (gpu_high, ram_high, check_interval, ...)
├── EvolutionConfig   (reflection_task_interval, reflection_time_interval)
└── WebConfig         (enabled, timeout, user_agent)
```

使用 `get_config()` 获取全局单例。

---

## Prover Tactic 优先级

Lean 证明策略按以下优先级选择:

```
第一优先：一行自动化 — simp → omega → ring → norm_num → decide → exact? → aesop
第二优先：直引 Mathlib — exact Nat.add_comm a b
第三优先：simp + 引理   — simp [Nat.add_comm]
第四优先：手动归纳      — induction（仅最后手段）
```

---

## 依赖关系图

```
main.py / web/app.py
    │
    ▼
turing.agents.skill_based_agent
    ├── turing.skills (skill_registry, math_skills, task_router)
    ├── turing.llm.llm_client
    ├── turing.lean.lean_interface
    ├── turing.memory (working, long_term, persistent)
    ├── turing.evolution (experience, reflection)
    └── turing.resources.resource_manager

turing.config ← 被所有模块引用（全局单例）
turing.utils  ← 日志配置，被入口脚本调用
```
