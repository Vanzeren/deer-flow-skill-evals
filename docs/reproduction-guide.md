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
83 passed in ~15s
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

### 4.1 Smoke（快速验证，推荐先跑）

3 条路由 case，不跑质量 Judge，约 20 秒。

**用 DeepSeek：**

```bash
DEEPSEEK_API_KEY=sk-your-key \
AGENT_MODEL=deepseek-v4-flash \
JUDGE_MODEL=openai-api/deepseek/deepseek-chat \
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

### 4.2 全量

20 条 × 3 epochs = 60 次路由 + 4 条质量 Judge，约 10–30 分钟。

```bash
DEEPSEEK_API_KEY=sk-your-key \
AGENT_MODEL=deepseek-v4-flash \
JUDGE_MODEL=openai-api/deepseek/deepseek-chat \
uv run python -m skill_eval.poc
```

预期输出：每次运行结果不同（LLM 有随机性），exit code：

| 退出码 | 含义 |
|---|---:|
| 0 | 所有 acceptance 通过 |
| 1 | 指标未达阈值 |
| 2 | 评测本身无效（基础设施故障） |

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

```json
{
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
  "quality_results": [
    {
      "case_id": "slr-implicit-rlhf-001",
      "judgment": {
        "recommended_route": "systematic-literature-review",
        "route_quality": 3, "process_quality": 3, "output_quality": 3,
        "fatal_error": false,
        "reasons": ["..."],
        "evidence": ["message[1]", "final_answer"]
      },
      "quality_passed": true
    },
    ...
  ],
  "acceptance": [
    {"name": "valid routing run rate", "actual": 0.967, "threshold": ">= 0.95", "passed": true},
    {"name": "macro routing precision", "actual": 0.85, "threshold": ">= 0.80", "passed": true},
    {"name": "macro routing recall", "actual": 0.83, "threshold": ">= 0.80", "passed": true}
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

**quality_results：**

| 字段 | 含义 | 通过线 |
|---|---|---|
| `route_quality` | Judge 认为 Agent 选的 Skill 是否合理 | ≥ 3/4 |
| `process_quality` | 工具链是否连贯、错误是否处理 | ≥ 3/4 |
| `output_quality` | 最终答案 / artifact 质量 | ≥ 3/4 |
| `fatal_error` | 是否不可恢复 | False |

---

## 6. 常见问题

### "Invalid Inspect judge model"

确保 `JUDGE_MODEL` 格式正确。DeepSeek 用 `openai-api/deepseek/deepseek-chat`，OpenAI 用 `openai/gpt-4o`。

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
- `tags` 中带 `quality` 的 4 条会跑完整 Agent + Judge
- `expected_output` 是给 Judge 的质量参考标准

---

## 8. 快速诊断命令

```bash
# 只跑路由评测，跳过质量 Judge
uv run pytest tests/skill_eval/test_routing_eval.py -q

# 只跑 Judge 单元测试
uv run pytest tests/skill_eval/test_judge.py -q

# 验证数据集格式
uv run python -c "
from skill_eval.dataset_loader import read_routing_cases, validate_poc_suite
cases = read_routing_cases('cases/literature_skill_routing.jsonl')
validate_poc_suite(cases)
print('OK: 20 cases, tags valid, quality cases have expected_output')
"
```
