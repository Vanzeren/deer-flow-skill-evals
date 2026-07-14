# skill_eval 核心代码原理

三层：**数据契约 → 执行引擎 → 评测 & 报告**。

---

## 一、数据契约

### `case_schema.py` — 每条评测用例长什么样

```python
class RoutingCase:
    id: str              # slr-implicit-rlhf-001
    input: str           # 给 Agent 的原话，不能包含 skill 名
    expected_route:      # 标准路由
        "systematic-literature-review"
        | "academic-paper-review"
        | "none"
    rationale: str       # 为什么这样标
    tags: list[str]      # explicit / implicit / keyword-collision / quality / ...
    expected_output: str | None   # 质量评测参考答案
```

约束：
- `extra="forbid"` — 多传字段直接报错
- input 不能包含 candidate skill 名（防止泄漏标准答案）
- input 不能以 `/` 开头（防止 slash command 强制激活）
- tags 去重、strip 空白
- 质量 case（tag 含 `quality`）必须有 `expected_output`

### `trace_schema.py` — Agent 跑完后留下的证据

```python
AgentTrace:
    messages:    list[dict]          # 完整对话历史
    tool_calls:  list[ToolCall]      # {id, name, args, result, error}
    artifacts:   list[AgentArtifact] # {path, mime_type, content}
    final_answer: str                # 最终文本
    errors:      list[str]           # 异常消息
    success:     bool
    latency_ms / input_tokens / output_tokens
```

设计要点：**不存原始事件流**（太大），提取紧凑结构。每个 `tool_call` 同时记录请求和返回，同一个 `tool_call_id` 关联。

---

## 二、执行引擎

### `agent_runner.py` — 抽象接口（38 行）

```python
class AgentRunRequest:
    case_id / user_input / mode / model_name / timeout_seconds / sandbox

class AgentRunResult:
    final_answer / success / trace / route_observation / thread_id

class AgentRunner(ABC):
    async def run(request: AgentRunRequest) -> AgentRunResult
```

只有一个 `run()` 方法。任何后端实现这个接口就能接入——真实 DeerFlow、mock、未来其他 agent 框架。

### `adapters/deerflow.py` — 真实 DeerFlow 实现

**子进程隔离。** 每次 `run()` spawn 一个独立 `multiprocessing.Process`，在里面初始化 `DeerFlowClient` → `dispatch()` → 通过 `Connection`(pipe) 传回结果。为什么不用协程？`DeerFlowClient` 内部有同步阻塞调用（LangGraph 图执行），直接 await 会卡死事件循环。

**防死锁。** 父进程先 `recv()` 读完 pipe 所有数据，再 `join()` 子进程。如果反过来（先 join 再 recv）：子进程写满 pipe 缓冲区后阻塞 → 父进程等子进程退出 → 互相等待 → 死锁。

**退出检测。** 子进程返回 payload 后给 5 秒退出宽限。超时未退出 → `terminate()` + 返回 infrastructure failure。防止"数据传回来了但进程泄漏"被伪装成成功。

**路由观察。** 子进程里嵌一个 `RoutingObserver`，逐事件 `feed()`，路由确定或 stream 结束后停止。

**沙箱。** 用 `LocalSandboxProvider` + `allow_host_bash=true`。需改环境变量时通过 `push_current_app_config` / `pop_current_app_config` 做临时 monkey-patch，跑完恢复。

### `routing.py` — 路由观察器（173 行）

```python
RoutingObserver(candidates=("systematic-literature-review", "academic-paper-review"))
    feed(event: StreamEvent) -> bool    # True = 路由已确定，可提前停止
    finalize(stream_completed, latency_ms) -> RouteObservation
```

核心逻辑三步：

1. **`_feed_ai()`** — 拦截 AI 发起的 `tool_calls`
   - `describe_skill(name)` → 只记录，不算路由
   - `read_file(".../skills/public/<candidate>/SKILL.md")` → 记录 `load_requested`

2. **`_feed_tool()`** — 匹配同一个 `tool_call_id` 的 tool result
   - 成功（不以 `Error:` 开头）→ `loaded`
   - 失败 → `load_failed`

3. **同批次决定。** 同一个 `batch_id` 内所有 load 完成后判断：
   - 只成功加载一个 → 路由确定
   - 成功加载多个 → `ambiguous`
   - 都没加载 → `none`

路径校验用 `PurePosixPath` 严格匹配 `/skills/public/<candidate>/SKILL.md`，排除误识别。

`RouteEvidence` 记录完整的证据链：`{id, kind, skill, tool_call_id, detail}`，每个证据可追溯到具体工具调用。

---

## 三、评测 & 报告

### `inspect_scorer.py` — 两个 scorer

**`routing_scorer`**（45 行）：纯规则，不调 LLM。

```
读 RouteObservation
  → completion=False / observed=None → NOANSWER + infrastructure_error
  → observed == expected → CORRECT
  → observed != expected → INCORRECT
```

**`quality_judge_scorer`**（55 行）：先检查基础设施，再调 Judge。

```
读 AgentTrace + RouteObservation
  → success=False / completion=False → NOANSWER + infrastructure_error
  → build_judge_evidence(trace, observation, skills)
  → judge_quality(bundle, model)
  → route_quality ≥ 3 ∧ process_quality ≥ 3 ∧ output_quality ≥ 3 → 通过
```

一个关键设计：调用 `dict.fromkeys(infrastructure_errors)` 去重，避免同一个错误在多个字段重复出现导致 Judge 被误导。

### `judge.py` — LLM Judge 引擎

**`build_judge_evidence()`** — 把 `AgentTrace` 拆成结构化证据：

```
message[N]      ← trace.messages        (每条 AI/tool 消息的完整 JSON)
tool_call[N]    ← trace.tool_calls       ({name, args})
tool_result[N]  ← trace.tool_calls       ({result, error})
error[N]        ← trace.errors
artifact[...]   ← trace.artifacts        ({path, mime_type, content})
final_answer    ← trace.final_answer
```

容量控制：单条 ≤ 12KB，总包 ≤ 80KB。超出截断并标记 `truncated=true`，Judge 知道信息不完整。

**`build_judge_prompt()`** — 纯 user message（无 system prompt），包含：
1. 任务说明：只评估可观测行为，不推断未发生的步骤
2. JSON schema（返回格式）
3. 两个 skill 的 route rubric + 通用 process rubric
4. 评分锚点（0–4）
5. 完整 evidence bundle JSON
6. 如有 `expected_output`：对比参考

只用 user message 的原因：部分模型不支持或不重视 system role，减少差异。

**`judge_quality()`** — 调 LLM → 解析 JSON → 校验引用：

```
get_model → model.generate(prompt)
  → _strip_fences(completion)          ← 剥离 ```json ```
  → QualityJudgment.model_validate_json
  → 失败 → repair_prompt → 重试一次
  → _validate_evidence_references()    ← 引用必须真实存在
```

`_strip_fences()` 处理 DeepSeek 等模型爱包 markdown fence 的问题。

`_validate_evidence_references()` 两层校验：
- 每条引用必须在 bundle 中真实存在
- 必须同时包含 trace 证据（message/tool_call/tool_result/error）和 output 证据（artifact/final_answer）

**评分维度：**

| 维度 | 评估内容 | 通过线 |
|---|---|---|
| route_quality | Skill 选择是否合理 | ≥ 3 |
| process_quality | 工具链是否连贯、错误是否处理 | ≥ 3 |
| output_quality | 最终答案/artifact 质量 | ≥ 3 |
| fatal_error | 是否不可恢复 | False |

### `report.py` — 指标聚合 & 输出

`extract_routing_results()` — 从 Inspect EvalLog 提取每条 case 各 epoch 的结果。

`summarize_routing()` — 40 行出全部指标：
- confusion matrix、per-class precision/recall/F1
- macro average（等权，避免类别不平衡）
- stability（同一 case 多次运行是否一致）
- valid_run_rate（排除基础设施失败）

`extract_quality_results()` — 从 Inspect log 提取 quality scorer 输出，区分 `infrastructure_error` 和 `judge_failure`。

`render_poc_markdown()` — 生成人类可读的 summary.md 和 summary.json。

### `poc.py` — 一键入口

```python
PocConfig.from_env()       # 读 AGENT_MODEL / JUDGE_MODEL 环境变量
preflight(config)           # 校验 case 文件、skill 文件、模型可用、config 路径
run_poc(config)             # routing eval（60 runs）→ quality eval（4 runs）
exit_code_for(summary)      # 0=全过 / 1=指标未达标 / 2=评测无效
```

---

## 核心设计原则

| 原则 | 体现位置 |
|---|---|
| **证据优于声明** | routing.py 看真实工具调用流不看模型文本；judge.py 只看 trace 不推断 |
| **失败独立分类** | inspect_scorer.py / report.py：infrastructure、judge、routing 三种失败分开 |
| **子进程隔离** | deerflow.py：每次 run 独立进程，崩溃不影响主流程 |
| **防死锁** | deerflow.py：先读 pipe 再 join，pipe 缓冲区满不死锁 |
| **退出宽限** | deerflow.py：子进程返回后 5s 未退出 → terminate + infrastructure failure |
| **容量控制** | judge.py：evidence 单条 12KB、总包 80KB，截断标记 truncated |
| **防作弊** | case_schema.py / dataset_loader.py：禁止 skill 名在 input、禁止 slash、分布强制校验 |
| **可重现** | poc.py preflight 记录 config SHA-256、skill SHA-256、版本号 |

---

## 文件清单

| 文件 | 行数 | 职责 |
|---|---|---|
| `case_schema.py` | ~40 | 数据契约 |
| `trace_schema.py` | ~60 | Agent 执行证据结构 |
| `agent_runner.py` | ~35 | 抽象执行接口 |
| `adapters/deerflow.py` | ~520 | 真实 DeerFlow 子进程执行 |
| `routing.py` | ~175 | 路由观察器 |
| `inspect_scorer.py` | ~100 | 两个 scorer |
| `judge.py` | ~335 | LLM Judge 引擎 |
| `inspect_solver.py` | ~90 | Inspect Task 组装 |
| `report.py` | ~350 | 指标聚合 & Markdown/JSON 输出 |
| `poc.py` | ~385 | 一键入口 |
| `dataset_loader.py` | ~65 | 数据加载 & 校验 |
