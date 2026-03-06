# Turing — 数学研究智能体系统

> **Turing v2.1.0** — 基于本地 **Qwen3-coder:30b** (via Ollama) 的自主数学研究智能体。  
> 通过 **Lean 4 + Mathlib** 定理证明器进行形式化验证，具备技能驱动推理、三层记忆和自我演化能力。

**English** | [中文文档见下方](#快速开始)

## Highlights

- **Skill-based single-agent** architecture: 93% success rate, avg 2.6 LLM calls/task (↓75% vs v1)
- **Lean 4 + Mathlib** formal verification as the ultimate validator
- **3-tier memory**: Working (session) / Long-Term (ChromaDB RAG) / Persistent (SQLite metacognition)
- **Self-evolution**: experience extraction → periodic reflection → improvement plans
- **Local-first**: runs entirely on local Ollama, no cloud API needed
- **Web UI**: FastAPI + vanilla HTML/CSS/JS, real-time SSE streaming

---

## v2 架构概览

v2 采用**技能驱动单智能体**架构，单次证明任务 LLM 调用从 9-12 次降至平均 **2.6 次**（↓75%），
成功率保持 **93%**。新增 **LLM 直接证明**快速路径：1 次 LLM 调用即编译通过则直接完成，
无需启动完整智能体流水线。

```
                        ┌─────────────────────────────────┐
                        │      SkillBasedTuringAgent      │
                        │   process_task()  统一入口       │
                        ├─────────────────────────────────┤
               ┌────────┤    Step 0: LLM 直接证明 (1 LLM) ├────────┐
               │        │    1次调用 → 编译通过 → 直接完成  │        │
               │        └───────────┬─────────────────────┘        │
               │                    │ 未通过 ↓ 降级                  │
               │        ┌───────────┴─────────────────────┐        │
               │        │        TaskRouter (规则引擎)     │        │
               │        │  classify / fix / name / trivial │        │
               │        └───────────┬─────────────────────┘        │
               │                    │ 复杂任务回落                    │
               ▼                    ▼                              ▼
        ┌──────────┐      ┌─────────────────┐          ┌────────────────┐
        │ 规则路径  │      │   SkillRegistry │          │  Lean 编译器    │
        │ (0 LLM)  │      │   11 项数学技能  │          │  (终极验证器)   │
        └──────────┘      │ 大模型·小模型路由 │          └────────────────┘
                          └─────────────────┘
               ┌──────────────────┬──────────────────┐
               ▼                  ▼                  ▼
        ┌──────────┐      ┌──────────────┐    ┌──────────────┐
        │ Working  │      │  Long-Term   │    │  Persistent  │
        │ Memory   │      │ Memory (RAG) │    │ Memory (SQL) │
        │ (会话级)  │      │  (ChromaDB)  │    │  (元认知)     │
        └──────────┘      └──────────────┘    └──────────────┘
```

### 六项优化原则

| 原则 | 说明 | 实现位置 |
|------|------|----------|
| **A** 大脑+小手 | 关键决策用大模型，细活用规则/小模型 | `task_router.py` |
| **B** 分层路由 | 简单任务走零 LLM 路径，复杂任务才升级 | `skill_based_agent.py` |
| **C** 共享 TaskState | 所有中间产物存一处，禁止重复读取 | `task_router.TaskState` |
| **D** Token 预算+早停 | 编译成功即停，不浪费 token | `_execute_proof()` |
| **E** 编译器即验证器 | Lean 编译器替代 Critic 辩论 | `lean_interface.py` |
| **F** 大小模型路由 | 命名/修复/评估等简单任务用轻量模型 | `Skill.use_light` |

---

## 快速开始

### 1. 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.11 | `asyncio` 特性依赖 |
| [Ollama](https://ollama.ai/) | latest | 本地 LLM 推理引擎 |
| [Lean 4](https://leanprover.github.io/) | ≥ 4.x | 定理证明器 |
| Mathlib | latest | Lean 数学库（推荐） |

### 2. 安装

```bash
git clone https://github.com/yourname/Math_Lean_Agent.git
cd Math_Lean_Agent
pip install -r requirements.txt

# 确保 Ollama 运行中
ollama serve &
ollama pull qwen3-coder:30b

# (可选) Lean 项目初始化 — Mathlib 首次下载需 ~10 分钟
cd lean_workspace && lake update && cd ..
```

### 3. 配置

编辑 `config.yaml`（唯一配置入口）:

```yaml
llm:
  provider: ollama
  model: qwen3-coder:30b            # 主推理模型
  light_model: qwen3-coder:30b      # 轻量任务（可换更小模型）

lean:
  project_dir: ./lean_workspace
  compile_timeout: 300               # Mathlib 导入需 ≥ 300s

agents:
  mode: "skill"                      # skill（推荐）| multi（legacy）
```

### 4. 运行

```bash
# 交互模式
python main.py

# 单任务证明
python main.py --task "证明: 对任意自然数 n, n + 0 = n"

# 训练模式（自主训练 60 分钟）
python main.py --train 60

# 自主学习循环
python main.py --loop

# ★ Web 前端（浏览器可视化界面）
python -m web.app                    # 默认 http://localhost:8000
python -m web.app --port 9000        # 自定义端口

# 全分支演化基准测试（推荐 skill 模式）
TOKENIZERS_PARALLELISM=false python run_evolution.py --mode skill --phases 3

# Legacy 多智能体模式（对比用）
TOKENIZERS_PARALLELISM=false python run_evolution.py --mode multi --phases 3

# 系统状态
python main.py --status
```

---

## 使用示例

### 交互模式

```
$ python main.py

  Turing 数学研究智能体 — 交互模式
  输入数学任务，或使用以下命令:
    /train [分钟]   进入训练模式
    /status         显示系统状态
    /reflect        触发反思
    /evaluate       系统全面评估
    /evolve         评估 + 演化流程
    /optimize       系统优化
    /quit           退出

Turing> 证明: 对任意自然数 a 和 b，a + b = b + a

--- 任务结果 ---
状态: ✓ 成功
LLM 调用: 2 次                        ← v2 技能模式
Lean 代码:
  import Mathlib
  theorem add_comm_nat (a b : Nat) : a + b = b + a := by
    exact Nat.add_comm a b
审查: 编译通过（Lean 编译器验证）
命名: add_comm_nat (加法交换律)
---
```

### 单任务模式（JSON 输出）

```bash
$ python main.py --task "证明: 对任意命题 P，¬¬P → P"

{
  "success": true,
  "lean_code": "import Mathlib\n\ntheorem double_neg_elim (P : Prop) : ¬¬P → P := by\n  exact not_not.mp",
  "theorem_naming": {
    "theorem_name": "double_neg_elim",
    "chinese_name": "双重否定消去",
    "is_novel": false
  },
  "llm_calls": 3
}
```

### 全分支演化

```bash
$ TOKENIZERS_PARALLELISM=false python run_evolution.py --mode skill --phases 3

  [Phase 1/3] 5 tasks
  ────────────────────────
  #01 ✓ (LLM: 2) 证明: 对任意自然数 n，n + 0 = n
  #02 ✓ (LLM: 3) 证明: 对任意整数 a, b，a + b = b + a
  #03 ✓ (LLM: 2) 证明: 空集是任意集合的子集
  #04 ✓ (LLM: 4) 证明: 对任意命题 P 和 Q，P ∧ Q → Q ∧ P
  #05 ✓ (LLM: 3) 证明: ∀ n : ℕ, 0 ≤ n

  [Phase 2/3] 5 tasks
  ...

  ══════════════════════════
  演化报告
  ──────────────────────────
  总任务: 15 | 成功: 14 (93%)
  总 LLM 调用: 39 (平均 2.6/任务)
  用时: 720s (12.0 min)
  ══════════════════════════
```

---

## 基准测试对比

| 指标 | v1 多智能体 | v2 技能驱动 | 变化 |
|------|------------|------------|------|
| 每任务 LLM 调用 | 9-12 次 | **2.6 次** | **↓75%** |
| LLM 直接搞定 | — | **1 次** (快速路径) | 新增 |
| 成功率 (15 任务) | — | **93%** | — |
| 架构开销 | 8 个 Agent 实例 | 1 个 Agent + 规则引擎 | ↓87.5% |
| 无 LLM 路径 | 无 | 规则分类/修复/命名 | 新增 |

---

## 核心特性

### 技能系统（11 项数学技能）

| 技能 | 合并自 | 用途 | 模型 |
|------|--------|------|------|
| `analyze_and_plan` | classify+plan+outline | 分类+策略+大纲 | 大 |
| `lean_prove` | Prover 首次生成 | 生成 Lean 证明代码 | 大 |
| `lean_fix` | Prover 错误修正 | 根据编译错误修复 | **小** |
| `name_and_review` | 命名+Critic | 定理命名+简评 | **小** |
| `evaluate` | Evaluator | 多维度评估 | **小** |
| `explore_conjectures` | Explorer | 猜想生成 | 大 |
| `explore_deep` | Explorer 深度 | 深度探索 | 大 |
| `decompose_task` | 任务分解 | 复杂任务拆分 | 大 |
| `evaluate_batch` | 批量评估 | 阶段性总结 | **小** |
| `generate_evolution` | 演化生成 | 根据反思生成改进 | 大 |
| `generate_training_problems` | Scout | 生成训练题 | 大 |

### 三层记忆系统

| 层级 | 存储引擎 | 功能 |
|------|---------|------|
| Working Memory | 内存 | 当前任务推理步骤、假设追踪（会话级） |
| Long-Term Memory | ChromaDB | 定理/策略/错误的语义检索 (RAG) |
| Persistent Memory | SQLite | 元认知经验、任务日志、反思记录 |

### 任务路由流水线

```
输入任务
  │
  ├── Step 0: LLM 直接证明 (1 LLM)
  │       └── 编译通过? ──→ ✓ 完成（跳过全部后续步骤）
  │                ↓ 否（降级到完整流程）
  │
  ├── 规则分类 (0 LLM) ──→ 高置信度? ──→ 跳过 LLM 分类
  │                            ↓ 否
  │                      LLM analyze_and_plan (1 次调用)
  ├── 难度评估 ──→ trivial? ──→ 规则生成代码 (0 LLM) → Lean 编译
  │                  ↓ 否
  │              easy/medium/hard → _execute_proof()
  │                  ├── lean_prove (1 LLM)
  │                  ├── Lean 编译 → 成功? → 早停 ✓
  │                  │                ↓ 否
  │                  ├── 规则修复 (0 LLM) → 重编译
  │                  │                ↓ 失败
  │                  ├── lean_fix (light LLM) × 最多 N 次
  │                  └── 成功 → name_and_review (light LLM)
  └── 存入记忆 + 经验积累
```

### Prover tactic 优先级

```
第一优先：一行自动化 — simp → omega → ring → norm_num → decide → exact? → aesop
第二优先：直引 Mathlib — exact Nat.add_comm a b
第三优先：simp + 引理   — simp [Nat.add_comm]
第四优先：手动归纳      — induction（仅最后手段）
```

### 定理命名与存储

1. 优先从 Lean 代码提取 Mathlib 定理名（`exact Xxx.yyy`）
2. 无法匹配时由 LLM 根据定理内容生成名称和描述
3. 存入 ChromaDB 长期记忆，标记 `is_novel` 标识是否为新定理
4. 定理库持久化到 `data/theorem_library.json`

### 自我演化

```
任务完成 → 经验提取 → 长期记忆存储
                ↓
         每 20 任务 / 24h
                ↓
         反思 → 弱点分析 → 改进计划 → 演化循环
```

### 资源管理

- CPU / RAM / GPU 实时监控（`psutil` + `GPUtil`）
- **Apple Silicon 原生适配**（检测 M1–M4 共享内存 GPU）
- 三级资源策略：HIGH（≥16GB GPU, ≥32GB RAM）/ MEDIUM / LOW

### Web 前端

简约的单页浏览器界面，提供 4 个视图：

| 面板 | 功能 |
|------|------|
| 🔬 **证明** | 输入定理、实时 SSE 流式输出证明结果、Lean 代码高亮 |
| 📚 **知识库** | 浏览 ChromaDB 中已证明的定理、策略、错误日志 |
| 🧠 **经验** | 各领域成功率图表、失败模式、反思记录 |
| 📈 **演化** | 全分支演化数据可视化（从 theorem_library.json） |

启动：`python -m web.app`，浏览器打开 `http://localhost:8000`。

技术栈：FastAPI + 原生 HTML/CSS/JS（零构建步骤、零 npm 依赖）。

---

## 演化成绩

### v1 全分支演化（50 个定理，10 个数学分支）

| 分支 | 领域 | 成功率 |
|------|------|--------|
| 自然数论 | number_theory | █████ 5/5 (100%) |
| 整数与整除性 | number_theory | █████ 5/5 (100%) |
| 命题逻辑 | logic | █████ 5/5 (100%) |
| 集合论 | set_theory | ████░ 4/5 (80%) |
| 代数结构 | algebra | ███░░ 3/5 (60%) |
| 序与格 | order_theory | ████░ 4/5 (80%) |
| 实分析 | analysis | ████░ 4/5 (80%) |
| 函数与映射 | functions | ████░ 4/5 (80%) |
| 组合数学 | combinatorics | ███░░ 3/5 (60%) |
| 拓扑学 | topology | ███░░ 3/5 (60%) |
| **合计** | | **40/50 (80%)** |

### v2 技能模式基准（15 任务，3 阶段）

| 指标 | 结果 |
|------|------|
| 成功率 | **14/15 (93%)** |
| 总 LLM 调用 | **39 (平均 2.6/任务)** |
| 相比 v1 LLM 调用 | **↓75%** |

已命名定理库保存在 `data/theorem_library.json`。

---

## 项目结构

```
Math_Lean_Agent/
├── main.py                          # CLI 入口（交互/单任务/训练/状态）
├── run_evolution.py                 # 全分支演化基准测试
├── config.yaml                      # 系统配置（唯一入口）
├── requirements.txt                 # Python 依赖
├── ARCHITECTURE.md                  # 系统架构详解
│
├── turing/                          # 核心 Python 包
│   ├── __init__.py                  # 包元信息（v2.1.0）
│   ├── config.py                    # 配置管理（YAML → dataclass 单例）
│   ├── utils.py                     # 共享工具（日志配置等）
│   │
│   ├── agents/                      # 智能体层
│   │   ├── base_agent.py            #   智能体抽象基类
│   │   ├── skill_based_agent.py     #   ★ v2 主智能体（推荐）
│   │   ├── turing_agent.py          #   v1 多智能体调度器（legacy）
│   │   ├── agent_factory.py         #   智能体工厂（仅 multi 模式）
│   │   └── legacy/                  #   v1 子智能体（prover/critic/...）
│   │
│   ├── skills/                      # ★ v2 核心：技能系统
│   │   ├── skill_registry.py        #   Skill 数据结构 + 注册表
│   │   ├── math_skills.py           #   11 项数学技能定义
│   │   └── task_router.py           #   规则引擎（零 LLM 路由/修复/命名）
│   │
│   ├── llm/
│   │   └── llm_client.py            #   Ollama / OpenAI LLM 异步客户端
│   ├── lean/
│   │   └── lean_interface.py        #   Lean 4 编译、错误解析
│   ├── memory/
│   │   ├── working_memory.py        #   工作记忆（会话级）
│   │   ├── long_term_memory.py      #   长期记忆（ChromaDB / RAG）
│   │   └── persistent_memory.py     #   持久记忆（SQLite / 元认知）
│   ├── evolution/
│   │   ├── experience.py            #   经验提取与强化
│   │   └── reflection.py            #   阶段性反思引擎
│   ├── resources/
│   │   └── resource_manager.py      #   GPU/RAM 监控 + 并发策略
│   └── web/
│       └── problem_scraper.py       #   题目抓取 + Loogle 定理搜索
│
├── web/                              # ★ Web 前端
│   ├── app.py                        #   FastAPI 后端（REST + SSE）
│   └── static/
│       └── index.html                #   单页前端（原生 HTML/CSS/JS）
│
├── data/
│   ├── theorem_library.json          # 演化产出的定理库
│   ├── chroma_db/                    # ChromaDB 持久化（.gitignore）
│   ├── logs/                         # 运行日志（.gitignore）
│   └── prompts/
│       └── agent_prompts.yaml        # Agent prompt 模板
│
├── lean_workspace/                   # Lean 4 项目（含 Mathlib）
│   ├── lakefile.toml
│   └── ...
└── tests/                            # 测试目录
```

---

## 依赖说明

| 包 | 用途 |
|----|------|
| `httpx` + `openai` | Ollama REST API 通信 |
| `chromadb` + `sentence-transformers` | 向量数据库 RAG 检索 |
| `aiosqlite` | 异步 SQLite（持久记忆） |
| `psutil` + `GPUtil` | CPU / RAM / GPU 监控 |
| `aiohttp` + `beautifulsoup4` | 网页抓取、Loogle/ProofWiki 搜索 |
| `loguru` | 结构化日志 |
| `rich` | 终端美化 |
| `pyyaml` | 配置文件解析 |
| `fastapi` + `uvicorn` | Web 前端 API 服务器 |

---

## 许可证

MIT License © 2026 Jansen Yu
