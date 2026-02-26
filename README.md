# Turing — 数学研究智能体系统

基于本地 **Qwen3-coder:30b** (via Ollama) 的自主数学研究智能体。  
通过 **Lean 4 + Mathlib** 定理证明器进行形式化验证，具备多智能体协作、自我演化和主动学习能力。

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                   Turing  (主智能体)                      │
│        任务分析 → 计划 → 调度 → 验证 → 评估 → 演化        │
├──────────────────────────────────────────────────────────┤
│ Prover   Explorer  Critic  Librarian  Scout  Evaluator  │
│ 形式证明  探索猜想  审查校验 知识整理  题目搜寻 多维评估    │
│                  Architect   DynamicAgent                │
│                  系统优化     运行时生成                   │
├──────────────────────────────────────────────────────────┤
│ Working Memory  │ Long-Term Memory  │ Persistent Memory  │
│ (会话上下文)     │ (ChromaDB / RAG)  │ (SQLite / 元认知)   │
├──────────────────────────────────────────────────────────┤
│ LLM Client (Ollama) │ Lean 4 Interface │ Resource Manager │
│                     │ + Mathlib        │ (Apple Silicon)  │
└──────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.11 | 异步特性依赖 |
| [Ollama](https://ollama.ai/) | latest | 本地 LLM 推理引擎 |
| [Lean 4](https://leanprover.github.io/) | ≥ 4.x | 定理证明器 |
| Mathlib | latest | Lean 数学库（可选，推荐） |

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

编辑 `config.yaml` 调整系统参数：

```yaml
llm:
  provider: ollama
  model: qwen3-coder:30b
  base_url: http://localhost:11434

lean:
  project_dir: ./lean_workspace
  compile_timeout: 300      # Mathlib 导入较慢，建议 ≥ 300

memory:
  long_term:
    similarity_threshold: 0.92  # 知识去重阈值
```

### 4. 运行

```bash
# 交互模式 — 直接输入数学任务
python main.py

# 单任务模式 — 证明一个定理
python main.py --task "证明: 对任意自然数 n, n + 0 = n"

# 训练模式 — 自主训练 60 分钟
python main.py --train 60

# 自主循环 — 持续自我学习
python main.py --loop

# 系统状态检查
python main.py --status

# 全分支数学演化 — 50 个定理，覆盖 10 个数学分支
TOKENIZERS_PARALLELISM=false python run_evolution.py
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
Lean 代码:
  import Mathlib
  theorem main_theorem (a b : Nat) : a + b = b + a := by
    exact Nat.add_comm a b
审查: 评分 9/10 — 通过
评估: 证明简洁直接，使用了 Mathlib 标准定理
---
```

### 单任务模式（JSON 输出）

```bash
$ python main.py --task "证明: 对任意命题 P，¬¬P → P"

{
  "success": true,
  "lean_code": "import Mathlib\n\ntheorem main_theorem (P : Prop) : ¬¬P → P := by\n  exact not_not.mp",
  "theorem_naming": {
    "theorem_name": "double_negation_elimination",
    "chinese_name": "双重否定消去",
    "is_novel": false
  }
}
```

### 全分支演化

```bash
$ TOKENIZERS_PARALLELISM=false python run_evolution.py

  🔬 第 1/10 期: 自然数论 [number_theory]
  📝 证明: 对任意自然数 n，n + 0 = n...
  → [✓] Nat.add_zero (4 行 Lean)
  ...
  📊 自然数论报告: 5/5 (100%) | 120s

  🏆 最终报告
  总任务: 50
  总成功: 40 (80%)
  总用时: 7195s (119.9min)
  💾 定理库已保存到: ./data/theorem_library.json
```

---

## 核心特性

### 三层记忆系统

| 层级 | 存储引擎 | 功能 | 实现 |
|------|---------|------|------|
| Working Memory | 内存 | 当前任务的推理步骤、假设追踪 | `working_memory.py` |
| Long-Term Memory | ChromaDB | 定理/策略/错误的语义检索 (RAG) | `long_term_memory.py` |
| Persistent Memory | SQLite | 元认知经验、任务日志、反思记录 | `persistent_memory.py` |

### 多智能体协作

| 智能体 | 职责 | 关键能力 |
|--------|------|----------|
| **Prover** | 自然语言 → Lean 4 证明 | 编译–修正循环，tactic 优先级策略 |
| **Explorer** | 模式发现、猜想生成 | 深度优先搜索，概念发散 |
| **Critic** | 多维度证明审查 | 逻辑/前提/边界/泛化 5 维评分 |
| **Librarian** | 知识库整理 | 关联发现，标签优化 |
| **Scout** | 训练题目搜寻 | 70/20/10 策略（成长区/薄弱区/探索） |
| **Architect** | 系统评估与优化 | Prompt 变更、新智能体设计 |
| **Evaluator** | 多维度结果评估 | 6 项技能，驱动演化方案生成 |

### Prover tactic 优先级

```
第一优先：一行自动化 — simp → omega → ring → norm_num → decide → exact? → aesop
第二优先：直引 Mathlib   — exact Nat.add_comm a b
第三优先：simp + 引理     — simp [Nat.add_comm]
第四优先：手动归纳        — induction（仅最后手段）
```

### 定理命名与存储

每个成功证明的定理会：
1. 搜索 Loogle (Mathlib 搜索引擎) 和 ProofWiki 匹配标准名称
2. 从 Lean 代码中提取 `exact Xxx.yyy` 引用的 Mathlib 定理名
3. 无法匹配时由 LLM 根据定理内容生成名称和描述
4. 存入 ChromaDB 长期记忆，标记 `is_novel` 标识是否为新定理

### 自我演化

```
任务输入 → 分类 → 记忆检索 → 计划 → 调度子智能体 → Lean 验证 → 积累
              ↓
        Evaluator 评估（10 维度评分）
              ↓
    评分 ≥ 6  → 继续
    评分 < 6  → 触发演化（Prompt 变更 / 策略调整 / 新智能体）
              ↓
        每 20 任务 / 24h → 反思 → 技能评估 → 演化循环
```

### 资源管理

- CPU / RAM / GPU 实时监控（`psutil` + `GPUtil`）
- **Apple Silicon 原生适配**（检测 M1–M4 共享内存 GPU）
- 三级资源策略：HIGH（≥16GB GPU, ≥32GB RAM）/ MEDIUM / LOW
- 智能体并发数动态控制

---

## 演化成绩

### 全分支演化（50 个定理，10 个数学分支）

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

已命名定理库保存在 `data/theorem_library.json`。

---

## 项目结构

```
Math_Lean_Agent/
├── main.py                        # CLI 入口（交互/单任务/训练/状态）
├── run_evolution.py               # 全分支数学演化脚本
├── config.yaml                    # 系统配置
├── requirements.txt               # Python 依赖
├── turing_system_prompt.md        # 系统设计文档
├── turing/
│   ├── __init__.py                # 包信息（版本号）
│   ├── config.py                  # 配置管理（YAML → dataclass 单例）
│   ├── utils.py                   # 共享工具（日志配置等）
│   ├── llm/
│   │   └── llm_client.py         # Ollama / OpenAI LLM 客户端
│   ├── lean/
│   │   └── lean_interface.py     # Lean 4 编译、错误解析、Mathlib 搜索
│   ├── memory/
│   │   ├── working_memory.py     # 工作记忆（会话级）
│   │   ├── long_term_memory.py   # 长期记忆（ChromaDB / RAG）
│   │   └── persistent_memory.py  # 持久记忆（SQLite / 元认知）
│   ├── agents/
│   │   ├── base_agent.py         # 智能体抽象基类
│   │   ├── agent_factory.py      # 智能体工厂（版本控制 + 回滚）
│   │   ├── turing_agent.py       # 主智能体（中央调度器）
│   │   ├── prover.py             # Lean 4 证明专家
│   │   ├── explorer.py           # 探索者（猜想生成）
│   │   ├── critic.py             # 审查官（证明评审）
│   │   ├── librarian.py          # 知识管理员
│   │   ├── scout.py              # 题目侦察兵
│   │   ├── architect.py          # 系统架构师
│   │   ├── evaluator.py          # 多维评估智能体
│   │   └── dynamic_agent.py      # 运行时动态生成智能体
│   ├── evolution/
│   │   ├── experience.py         # 经验强化管理
│   │   └── reflection.py         # 阶段性反思引擎
│   ├── resources/
│   │   └── resource_manager.py   # 资源监控 + 策略
│   └── web/
│       └── problem_scraper.py    # 题目抓取 + 定理名称搜索
├── data/
│   ├── prompts/
│   │   └── agent_prompts.yaml    # Prompt 模板参考文档
│   ├── theorem_library.json      # 演化产出的定理库
│   ├── chroma_db/                # ChromaDB 持久化目录
│   ├── logs/                     # 运行日志
│   └── snapshots/                # 系统快照
├── lean_workspace/                # Lean 4 项目（含 Mathlib）
│   ├── lakefile.toml
│   └── ...
└── tests/                         # 测试目录（待完善）
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

---

## 许可证

MIT License © 2026 Jansen Yu
