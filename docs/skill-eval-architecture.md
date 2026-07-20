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
    messages:    list[dict]          # 完整对话历史（采集保留，不进 Judge）
    tool_calls:  list[ToolCall]      # {id, name, args, result, error}
    tool_call_chain: list[list[str]] # 工具调用链：外层=按时间序的调用批次，
                                     # 内层=同一 AI 消息并发发起的 tool_call id
    quick_turn:  QuickTurnCapture | None  # quick 模式捕获的首轮输出
                                          # {message_id, skill, content}
    artifacts:   list[AgentArtifact] # {path, mime_type, content}
    final_answer: str                # 最终文本
    errors:      list[str]           # 异常消息
    success:     bool
    latency_ms / input_tokens / output_tokens
```

设计要点：**不存原始事件流**（太大），提取紧凑结构。每个 `tool_call` 同时记录请求和返回，同一个 `tool_call_id` 关联。`tool_call_chain` 由 adapter 从已采集的 per-message 数据派生，零额外采集成本；`messages` 继续采集用于调试/追溯，但不再进入 Judge 证据包。

---

## 二、执行引擎

### `agent_runner.py` — 抽象接口（37 行）

```python
type RunMode = "routing_probe" | "quick" | "full"
    # routing_probe: 路由确定即停，只做路由评估
    # quick:         路由命中后继续到 Skill 加载后的第一条 AI 文本输出即停
    # full:          跑完整个任务，评估最终输出

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

**quick 捕获。** quick 模式下 `_QuickTurnWatcher` 在路由判定为具体 candidate 后开始监视：第一条**新出现的、累积文本非空的** AI 消息为目标轮；当事件流进入下一条消息或 stream end 时捕获完成并提前停止。`ambiguous` 立即停；`none` 退化为 routing_probe 行为；超时无文本轮 → `quick_turn=None`（不算 infrastructure failure）。

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

### `inspect_scorer.py` — 三个 scorer

**`routing_scorer`**：纯规则，不调 LLM。

```
读 RouteObservation
  → completion=False / observed=None → NOANSWER + infrastructure_error
  → observed == expected → CORRECT
  → observed != expected → INCORRECT
```

**`quality_judge_scorer`**（full 模式）：先检查基础设施，再调 Judge。

```
读 AgentTrace + RouteObservation
  → success=False / completion=False → NOANSWER + infrastructure_error
  → build_judge_evidence(..., target="final_output")
  → judge_quality(bundle, model)
  → route_quality ≥ 3 ∧ process_quality ≥ 3 ∧ output_quality ≥ 3 → 通过
```

**`quick_turn_scorer`**（quick 模式）：按序检查，失败分类互斥。

```
读 AgentTrace + RouteObservation + case
  → 基础设施失败                → NOANSWER + infrastructure_error
  → expected_route == "none"     → NOANSWER + not_applicable_none_case
  → observed != expected         → NOANSWER + route_mismatch（路由分归 routing_scorer，不重复扣分）
  → trace.quick_turn is None     → NOANSWER + quick_turn_missing
  → build_judge_evidence(..., target="quick_turn")
  → judge_quick_turn(bundle, model)
  → turn_quality ≥ 3 ∧ ¬fatal_error → CORRECT
```

一个关键设计：调用 `dict.fromkeys(infrastructure_errors)` 去重，避免同一个错误在多个字段重复出现导致 Judge 被误导。

### `judge.py` — LLM Judge 引擎

**`build_judge_evidence()`** — 把 `AgentTrace` 拆成结构化证据（**不再包含 message trace**）：

```
tool_chain[B]   ← trace.tool_call_chain  (第 B 个并发批，展开为 {id, name, args, result, error})
error[N]        ← trace.errors
artifact[...]   ← trace.artifacts        ({path, mime_type, content})
final_answer    ← trace.final_answer      (仅 target="final_output")
quick_turn      ← trace.quick_turn        (仅 target="quick_turn")
```

设计动机：完整消息历史体量太大，Judge 实际上跑不完；过程证据改由按并发批分组的工具调用链承载。quick target 下会排除被捕获轮自身发起的调用批。

容量控制：单条 ≤ 12KB，总包 ≤ 80KB。超出截断并标记 `truncated=true`，Judge 知道信息不完整。

**`build_judge_prompt()`** — 纯 user message（无 system prompt），包含：
1. 任务说明：只评估可观测行为，不推断未发生的步骤
2. JSON schema（返回格式）
3. 两个 skill 的 route rubric + 通用 process rubric
4. 评分锚点（0–4）
5. 完整 evidence bundle JSON
6. 如有 `expected_output`：对比参考

只用 user message 的原因：部分模型不支持或不重视 system role，减少差异。

**`judge_quality()` / `judge_quick_turn()`** — 调 LLM → 解析 JSON → 校验引用：

```
get_model → model.generate(prompt)
  → _strip_fences(completion)          ← 剥离 ```json ```
  → QualityJudgment / QuickJudgment.model_validate_json
  → 失败 → repair_prompt → 重试一次
  → _validate_evidence_references()    ← 引用必须真实存在
```

`_strip_fences()` 处理 DeepSeek 等模型爱包 markdown fence 的问题。

`_validate_evidence_references()` 两层校验：
- 每条引用必须在 bundle 中真实存在
- bundle 含过程证据（tool_chain/error）时必须引用至少一条；且必须始终引用至少一条输出证据（artifact/final_answer/quick_turn）

**评分维度（full 模式 `QualityJudgment`）：**

| 维度 | 评估内容 | 通过线 |
|---|---|---|
| route_quality | Skill 选择是否合理 | ≥ 3 |
| process_quality | 工具链是否连贯、错误是否处理 | ≥ 3 |
| output_quality | 最终答案/artifact 质量 | ≥ 3 |
| fatal_error | 是否不可恢复 | False |

**评分维度（quick 模式 `QuickJudgment`）：**

| 维度 | 评估内容 | 通过线 |
|---|---|---|
| turn_quality | Skill 加载后首轮输出是否体现该 Skill 的流程/格式约束、是否回应用户请求 | ≥ 3 |
| fatal_error | 是否不可恢复 | False |

### `report.py` — 指标聚合 & 输出

`extract_routing_results()` — 从 Inspect EvalLog 提取每条 case 各 epoch 的结果。

`summarize_routing()` — 40 行出全部指标：
- confusion matrix、per-class precision/recall/F1
- macro average（等权，避免类别不平衡）
- stability（同一 case 多次运行是否一致）
- valid_run_rate（排除基础设施失败）

`extract_quality_results()` — 从 Inspect log 提取 quality scorer 输出，区分 `infrastructure_error` 和 `judge_failure`。

`extract_quick_results()` / `summarize_quick()` — 提取 quick_turn_scorer 输出并聚合：judged 数、pass rate、turn_quality 均值与 0–4 分布、五类失败桶（infrastructure_error / judge_failure / quick_turn_missing / route_mismatch / not_applicable_none_case）。

`render_poc_markdown()` — 生成人类可读的 summary.md 和 summary.json（含 quick 质量小节；不内嵌 messages）。

### `poc.py` — 一键入口

```python
PocConfig.from_env()       # 读 AGENT_MODEL / JUDGE_MODEL 环境变量
preflight(config)           # 校验 case 文件、skill 文件、模型可用、config 路径
run_poc(config)             # routing eval（60 runs）→ quick/full quality eval
                            # 由 --quality-mode quick|full|both 选择（默认 both）
exit_code_for(summary)      # 0=全过 / 1=指标未达标 / 2=评测无效
```

quick 质量门槛：通过数 / 实际可判（judged）case 数 ≥ 75%；`quick_turn_missing`、`judge_failure`、infrastructure 失败出现即判评测无效（exit 2）。

---

## 核心设计原则

| 原则 | 体现位置 |
|---|---|
| **证据优于声明** | routing.py 看真实工具调用流不看模型文本；judge.py 只看 trace 不推断 |
| **一次运行两个事实** | quick 模式复用 routing stream，路由判定与首轮捕获来自同一条轨迹 |
| **失败独立分类** | inspect_scorer.py / report.py：infrastructure、judge、routing、quick_turn_missing 分开 |
| **子进程隔离** | deerflow.py：每次 run 独立进程，崩溃不影响主流程 |
| **防死锁** | deerflow.py：先读 pipe 再 join，pipe 缓冲区满不死锁 |
| **退出宽限** | deerflow.py：子进程返回后 5s 未退出 → terminate + infrastructure failure |
| **容量控制** | judge.py：evidence 单条 12KB、总包 80KB，截断标记 truncated；消息历史不进 Judge |
| **防作弊** | case_schema.py / dataset_loader.py：禁止 skill 名在 input、禁止 slash、分布强制校验 |
| **可重现** | poc.py preflight 记录 config SHA-256、skill SHA-256、版本号 |

---

## 文件清单

| 文件 | 行数 | 职责 |
|---|---|---|
| `case_schema.py` | ~70 | 数据契约 |
| `trace_schema.py` | ~45 | Agent 执行证据结构（含 tool_call_chain / quick_turn） |
| `agent_runner.py` | ~37 | 抽象执行接口（routing_probe / quick / full） |
| `adapters/deerflow.py` | ~590 | 真实 DeerFlow 子进程执行 + quick 捕获 |
| `routing.py` | ~176 | 路由观察器 |
| `inspect_scorer.py` | ~183 | 三个 scorer |
| `judge.py` | ~420 | LLM Judge 引擎（full + quick） |
| `inspect_solver.py` | ~32 | Inspect solver（透传 mode） |
| `evals/skills_routing_eval.py` | ~46 | 路由评估 Task |
| `evals/skills_quick_eval.py` | ~55 | quick 质量评估 Task |
| `evals/skills_quality_eval.py` | ~46 | full 质量评估 Task |
| `report.py` | ~495 | 指标聚合 & Markdown/JSON 输出 |
| `poc.py` | ~475 | 一键入口（--quality-mode） |
| `dataset_loader.py` | ~68 | 数据加载 & 校验 |
