# DeerFlow Agent Routing POC — 复现指南

本文档供不熟悉本仓库的同事在新机器上从头复现 POC 结果。

---

## 1. 环境要求

| 工具 | 最低版本 | 验证命令 |
|---|---|---|
| Python | ≥ 3.12 | `python --version` |
| uv | 任意 | `uv --version` |
| Git | 任意 | `git --version` |

以下两个 API Key 二选一即可：

- **DeepSeek API Key**（推荐）：同时用于被测 Agent 和质量 Judge
- **OpenAI API Key**：仅用于质量 Judge（被测 Agent 需要另外配置）

---

## 2. 克隆 & 安装

```bash
git clone <repo-url> deer-flow
cd deer-flow
cd backend
uv sync --group dev
```

验证安装：

```bash
uv run pytest tests/skill_eval -q
```

预期输出：

```text
121 passed in ~15s
```

---

## 3. 配置

### 3.1 被测 Agent（DeerFlow）

需要 DeerFlow 的 `config.yaml` 中配好一个模型。用你已有的配置文件，或者参考 `config.example.yaml` 新建。

确认模型可用：

```bash
uv run python -c "
from deerflow.client import DeerFlowClient
c = DeerFlowClient()
print([m.get('name') for m in c.list_models().get('models',[])])
"
```

记下你要用的模型名，例如 `deepseek-v4-flash`。

### 3.2 质量 Judge（Inspect AI）

设置 API Key 环境变量。两种方式：

**方式 A：DeepSeek API**

```bash
export DEEPSEEK_API_KEY=sk-your-key-here
```

**方式 B：OpenAI API**

```bash
export OPENAI_API_KEY=sk-your-key-here
```

**不需要同时设置两个。**

---

## 4. 运行

均在 `backend/` 目录下执行。

### 4.0 评估模式：`--mode routing|quick|full`

三种评估模式完全独立，`--mode` 可重复指定任意组合；**不指定 = 三个全跑**。路由评估不再必跑。

| 模式 | 评估对象 | 单条成本 | 适用场景 |
|---|---|---|---|
| `routing` | 20 条 case × 3 epochs 的路由准确率（ smoke 为固定 3 条） | 低（路由确定即停） | 路由 benchmark |
| `quick` | Skill 加载成功后的**第一条 AI 文本输出**（Skill 的影响集中在这一次输出上） | 低（加载 Skill + 一轮输出即停） | 快速迭代、大批量质量评估 |
| `full` | Agent 跑完整个任务后的**最终输出**（smoke 下始终不可用） | 高（完整任务执行） | 端到端质量确认 |

示例：

```bash
uv run python -m skill_eval.poc --mode routing                # 只跑路由
uv run python -m skill_eval.poc --mode quick                  # 只跑 quick 质量（不跑路由 benchmark）
uv run python -m skill_eval.poc --mode routing --mode quick   # 路由 + quick
```

quality case 共 4 条，其中 3 条期望路由到具体 Skill（quick/full 都会评估），1 条期望 `none`（quick 模式下标记为 `not_applicable_none_case`，不参与 quick 质量门槛）。quick/full 的运行不依赖路由 benchmark——每条 run 内部仍会观察路由用于判定。

### 4.1 Smoke（快速验证，推荐先跑）

3 条路由 case + 3 条 quick 质量评估（默认模式组合；smoke 下 full 模式始终跳过），约 2–5 分钟。

**用 DeepSeek：**

```bash
DEEPSEEK_API_KEY=sk-your-key \
AGENT_MODEL=deepseek-v4-flash \
JUDGE_MODEL=openai-api/deepseek/deepseek-reasoner \
uv run python -m skill_eval.poc --smoke
```

**用 OpenAI：**

```bash
OPENAI_API_KEY=sk-your-key \
AGENT_MODEL=deepseek-v4-flash \
JUDGE_MODEL=openai/gpt-4o \
uv run python -m skill_eval.poc --smoke
```

> 注意：如果你用 Fish shell，上面 `KEY=VAL cmd` 语法不生效。改用：
> ```fish
> env KEY=VAL uv run python -m skill_eval.poc --smoke
> ```

预期输出：3 条全部通过，exit code 0。

> 只想跑路由评估：`uv run python -m skill_eval.poc --smoke --mode routing`。

### 4.2 全量

20 条 × 3 epochs = 60 次路由 + 4 条 quick 质量评估 + 4 条 full 质量评估。quick 部分约几分钟，full 部分约 10–30 分钟。

```bash
DEEPSEEK_API_KEY=sk-your-key \
AGENT_MODEL=deepseek-v4-flash \
JUDGE_MODEL=openai-api/deepseek/deepseek-reasoner \
uv run python -m skill_eval.poc
```

快速迭代时只跑 quick 质量评估（连 60 次路由 benchmark 也跳过，成本最低）：

```bash
uv run python -m skill_eval.poc --mode quick
```

预期输出：每次运行结果不同（LLM 有随机性），exit code：

| 退出码 | 含义 |
|---|---:|
| 0 | 所有 acceptance 通过 |
| 1 | 指标未达阈值 |
| 2 | 评测本身无效（基础设施故障 / judge 失败 / quick_turn_missing） |

---

## 5. 结果在哪

```text
backend/eval-results/<run-id>/
├── summary.json    # 机器可读的完整结果
├── summary.md      # 人类可读报告
└── traces/         # 每次 Agent 运行的完整事件流（JSONL）
```

每次运行产生一个新的 `<run-id>` 目录。

### 5.1 summary.json 关键字段

`schema_version` 为 `deerflow.agent-routing-poc.v3`。

```json
{
  "modes": ["routing", "quick", "full"],
  "routing": {
    "planned_runs": 60,
    "valid_runs": 58,
    "valid_run_rate": 0.967,
    "confusion": {
      "systematic-literature-review": {
        "systematic-literature-review": 20, "academic-paper-review": 2, "none": 2
      },
      ...
    },
    "macro_precision": 0.85,
    "macro_recall": 0.83,
    "macro_f1": 0.84,
    "stability_rate": 0.85,
    "results": [
      {
        "case_id": "slr-implicit-rlhf-001",
        "epoch": 1,
        "expected_route": "systematic-literature-review",
        "observed_route": "systematic-literature-review",
        "evidence": [
          {"kind": "load_requested", "skill": "systematic-literature-review", "tool_call_id": "..."},
          {"kind": "loaded", "skill": "systematic-literature-review", "tool_call_id": "..."}
        ]
      },
      ...
    ]
  },
  "quick_results": [
    {
      "case_id": "slr-implicit-rlhf-001",
      "judgment": {
        "turn_quality": 3,
        "fatal_error": false,
        "rationale": "...",
        "evidence_references": ["tool_chain[0]", "quick_turn"]
      },
      "category": null,
      "turn_quality": 3,
      "quality_passed": true
    },
    ...
  ],
  "quick_metrics": {
    "planned_runs": 4,
    "judged_cases": 3,
    "passed_cases": 3,
    "pass_rate": 1.0,
    "mean_turn_quality": 3.33,
    "turn_quality_distribution": {"0": 0, "1": 0, "2": 0, "3": 2, "4": 1},
    "failure_buckets": {
      "infrastructure_error": 0,
      "judge_failure": 0,
      "quick_turn_missing": 0,
      "route_mismatch": 0,
      "not_applicable_none_case": 1
    }
  },
  "quality_results": [
    {
      "case_id": "slr-implicit-rlhf-001",
      "judgment": {
        "recommended_route": "systematic-literature-review",
        "route_quality": 3, "process_quality": 3, "output_quality": 3,
        "fatal_error": false,
        "reasons": ["..."],
        "evidence": ["tool_chain[0]", "final_answer"]
      },
      "quality_passed": true
    },
    ...
  ],
  "acceptance": [
    {"name": "valid routing run rate", "actual": 0.967, "threshold": ">= 0.95", "passed": true},
    {"name": "macro routing precision", "actual": 0.85, "threshold": ">= 0.80", "passed": true},
    {"name": "macro routing recall", "actual": 0.83, "threshold": ">= 0.80", "passed": true},
    {"name": "quick quality pass rate of judgeable cases", "actual": 1.0, "threshold": ">= 75% of judgeable", "passed": true},
    {"name": "quality cases passing all dimension thresholds", "actual": 4, "threshold": ">= 3 of 4", "passed": true}
  ],
  "errors": []
}
```

### 5.2 各字段含义

**routing 指标：**

| 字段 | 含义 | 通过线 |
|---|---|---|
| `valid_run_rate` | 成功完成的比例（排除基础设施失败） | ≥ 0.95 |
| `macro_precision` | 三个类别 precision 等权平均 | ≥ 0.80 |
| `macro_recall` | 三个类别 recall 等权平均 | ≥ 0.80 |
| `stability_rate` | 同一 case 三次运行结果一致的比例 | 报告，非硬门槛 |

**quick_results / quick_metrics（快速模式，首轮输出评估）：**

| 字段 | 含义 | 通过线 |
|---|---|---|
| `turn_quality` | Judge 对 Skill 加载后第一条 AI 文本输出的评分 | ≥ 3/4 |
| `fatal_error` | 该轮输出是否不可恢复 | False |
| `category` | 未评估原因（见下表），评估成功为 null | — |
| `quick_metrics.pass_rate` | 通过数 / 实际可判 case 数（judged） | ≥ 0.75 |
| `turn_quality_distribution` | 0–4 分各档 case 数 | 报告 |
| `failure_buckets` | 五类失败各自的 case 数 | 报告 |

quick 失败分类（互斥，体现在 `category` / `failure_buckets`）：

| category | 含义 |
|---|---|
| `infrastructure_error` | Agent 运行或路由观察本身坏掉 |
| `route_mismatch` | 实际路由 ≠ 期望路由（路由分由 routing 评估负责，不重复扣分） |
| `quick_turn_missing` | 路由命中但到流结束也没捕到文本轮 |
| `judge_failure` | Judge 输出修复一次后仍无法解析/校验 |
| `not_applicable_none_case` | 期望路由为 none，quick 质量评估不适用 |

**quality_results（完整模式，最终输出评估）：**

| 字段 | 含义 | 通过线 |
|---|---|---|
| `route_quality` | Judge 认为 Agent 选的 Skill 是否合理 | ≥ 3/4 |
| `process_quality` | 工具链是否连贯、错误是否处理 | ≥ 3/4 |
| `output_quality` | 最终答案 / artifact 质量 | ≥ 3/4 |
| `fatal_error` | 是否不可恢复 | False |

> 注：两种模式的 Judge 证据包都不再包含完整消息历史（message trace），过程证据以 `tool_chain[B]`（按并发批分组的工具调用链）形式提供——这也是 `evidence` / `evidence_references` 中出现 `tool_chain[0]` 的原因。

---

## 6. 常见问题

### "Invalid Inspect judge model"

确保 `JUDGE_MODEL` 格式正确。DeepSeek 用 `openai-api/deepseek/deepseek-reasoner`（推荐，schema/证据引用遵循稳定），OpenAI 用 `openai/gpt-4o`。

> 注意：不建议用 `deepseek-chat` 当 judge——实测它反复不遵守「必须引用证据」的校验，会产生 `judge_failure` 让整轮评测无效（exit 2）。

### "Required environment variable(s) missing"

两个变量都必须设置：`AGENT_MODEL` 和 `JUDGE_MODEL`。

### "quality cases missing expected_output"

`backend/cases/literature_skill_routing.jsonl` 文件不完整。确认 4 条 quality case 都包含 `expected_output` 字段。

### API 调用失败 / 401

检查 API Key 是否正确、是否有额度。

### 某条 Agent 跑超时 / recursion limit

DeerFlow 的 Agent 循环有 100 步限制。复杂的 skill 工作流可能触发。这不影响评测框架本身——会被记录为 infrastructure failure 并纳入 `valid_run_rate`。

---

## 7. 数据集说明

`backend/cases/literature_skill_routing.jsonl` — 20 条，JSONL 格式。

```json
{
  "id": "slr-implicit-rlhf-001",
  "input": "What does the literature say about RLHF? ...",
  "expected_route": "systematic-literature-review",
  "rationale": "The literature plus a three-paper synthesis requires cross-paper review.",
  "tags": ["implicit", "multi-paper", "quality"],
  "expected_output": "Should include: (1) identification of at least 3 ..."
}
```

- `expected_route` 是标准路由答案
- `tags` 中带 `quality` 的 4 条进入质量评估：quick 模式评估 Skill 加载后的首轮输出（期望 `none` 的 1 条不适用），full 模式评估完整任务后的最终输出
- `expected_output` 是给 Judge 的质量参考标准

---

## 8. 快速诊断命令

```bash
# 只跑路由评测，跳过质量 Judge
uv run pytest tests/skill_eval/test_routing_eval.py -q

# 只跑 Judge 单元测试（含 quick judge）
uv run pytest tests/skill_eval/test_judge.py -q

# 只跑 quick scorer / quick eval task 单元测试
uv run pytest tests/skill_eval/test_quick_scorer.py tests/skill_eval/test_quick_eval.py -q

# 验证数据集格式
uv run python -c "
from skill_eval.dataset_loader import read_routing_cases, validate_poc_suite
cases = read_routing_cases('cases/literature_skill_routing.jsonl')
validate_poc_suite(cases)
print('OK: 20 cases, tags valid, quality cases have expected_output')
"
```
