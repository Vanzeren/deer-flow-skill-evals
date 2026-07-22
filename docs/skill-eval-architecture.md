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
                                          # {message_id, skill, content, tool_calls}
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
    # quick:         路由命中后继续到 Skill 加载后的第一条 AI 输出即停
    #                （文本或 tool_call 任一出现均算输出）
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

**quick 捕获。** quick 模式下 `_QuickTurnWatcher` 在路由判定为具体 candidate 后开始监视：第一条**新出现且包含非空文本或至少一个 `tool_call`** 的 AI 消息为目标轮；当事件流进入下一条消息或 stream end 时捕获完成并提前停止。捕获结果同时保留 `content` 和该消息的结构化 `tool_calls`（名称、参数及已观察到的结果/错误），因此纯工具调用轮同样进入 Judge。`ambiguous` 立即停；`none` 退化为 routing_probe 行为；流结束仍没有任何文本或工具调用 → `quick_turn=None`（不算 infrastructure failure）。

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
quick_turn      ← trace.quick_turn        (仅 target="quick_turn"，JSON 包含 content + tool_calls)
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

`render_poc_markdown()` — 生成人类可读的 summary.md 和 summary.json（含 quick 质量小节；不内嵌 messages；routing 未运行时路由小节显示 Skipped）。schema_version 为 `deerflow.agent-routing-poc.v3`。

### `poc.py` — 一键入口

```python
PocConfig.from_env()       # 读 AGENT_MODEL / JUDGE_MODEL 环境变量
preflight(config)           # 校验 case 文件、skill 文件、模型可用、config 路径
run_poc(config)             # 按 config.modes 跑 routing / quick / full 任意组合
                            # 由可重复的 --mode 选择（默认三个全跑；
                            # routing 不跑时 summary.routing=None）
exit_code_for(summary)      # 0=全过 / 1=指标未达标 / 2=评测无效
```

quick 质量门槛：通过数 / 实际可判（judged）case 数 ≥ 75%；`quick_turn_missing`、`judge_failure`、infrastructure 失败出现即判评测无效（exit 2）。

---

## 四、如何接入新的 Agent Runtime

这一节是当前实现的对接指南。早期的
[`2026-07-07-skill-eval-deerflow-adapter.md`](superpowers/specs/2026-07-07-skill-eval-deerflow-adapter.md)
是历史设计稿，其中的进程模型、事件类型和 trace 字段已经过时，不应作为实现依据。

### 4.1 接入边界

新 Runtime 只需要实现 `AgentRunner`，将自己的事件流归一化为稳定的
`AgentRunResult`：

```python
class AgentRunner(Protocol):
    async def run(self, request: AgentRunRequest) -> AgentRunResult: ...
```

必须返回的事实分为三组：

| 事实 | 字段 | 要求 |
|---|---|---|
| 执行结果 | `success`、`thread_id`、`final_answer`、`errors` | 运行失败也要返回可解析的部分 trace，不要只抛异常 |
| 路由结果 | `route_observation` | 基于真实 Skill 加载工具调用，不读取模型的自我声明 |
| 可评分证据 | `trace.tool_calls`、`tool_call_chain`、`quick_turn`、`artifacts` | 保留稳定 ID，让 Judge 能引用并追溯到原始事件 |

Generic scorer 只能读取这些归一化字段。Runtime 私有对象、LangGraph 消息或供应商响应
不得泄漏到 scorer；需要深挖时通过 `raw_trace_ref` 找原始 JSONL。

### 4.2 推荐实现顺序

1. **先冻结输入输出契约。** 用 `AgentRunRequest.model_validate()` 和
   `AgentRunResult.model_validate()` 覆盖子进程或网络边界，尽早暴露 schema 漂移。
2. **实现纯事件 adapter。** `feed(event)` 只积累数据，`build()` 只生成
   `AgentTrace`；不要在 scorer 中解析 Runtime 原生事件。
3. **按 ID 合并消息。** 流式供应商会把同一条 AI 消息拆成多个 delta，文本需要累加，
   tool call 需要按 `tool_call_id` 去重并关联结果。
4. **再实现三种停止策略。** `routing_probe`、`quick`、`full` 共用同一条采集链，
   只改变何时安全关闭 stream。
5. **最后加进程隔离和超时。** 先用合成事件把 trace 转换测准，再接真实模型；否则
   schema、模型和多进程问题会混在一起。

### 4.3 DeerFlow 当前事件映射

`DeerFlowClient.stream()` 当前主要产生以下事件：

| 事件 | 处理方式 |
|---|---|
| `messages-tuple` / AI | 按 message ID 累加 `content`；从 `tool_calls` 建立调用记录和同批次调用链 |
| `messages-tuple` / tool | 用 `tool_call_id` 回填 `result` / `error`；`Error:` 前缀也视为错误 |
| `values` | 从完整消息快照补齐流式 delta 中缺失的 tool args，并收集 artifacts |
| `end` | 记录 usage；只有看到它才能证明自然完成 |

不要照早期设计稿等待独立的 `tool_call` / `tool_result` 事件。真实调用和返回都在
`messages-tuple` 中，`values` 还是补齐参数的重要兜底。

### 4.4 三种模式的停止边界

| 模式 | 何时停止 | 容易误判的边界 |
|---|---|---|
| `routing_probe` | 某批候选 Skill 加载结果全部落定后立即关闭 stream | `none` 没有正向加载证据，只能等正常 `end` |
| `quick` | 路由命中后，第一条新 AI 消息包含非空文本或至少一个 `tool_call`，且该消息到达边界后停止 | 纯 `tool_call` 也是输出，不能等文本；要把该轮调用嵌入 `quick_turn` |
| `full` | 正常 `end` | 空 final answer 只有在存在 artifact 时才可能是有效产出 |

路由必须等待 Skill `read_file(.../SKILL.md)` 的 ToolMessage 成功返回。只看到请求就停止，
会把权限错误、路径错误或工具失败误记成成功路由。同一 AI 消息并发加载多个候选 Skill
时，必须等整批返回后判为 `ambiguous`，不能采用“第一个返回者获胜”。

### 4.5 Quick 输出和 Judge 证据

`QuickTurnCapture` 是一条完整的 assistant 输出，而不是文本别名：

```json
{
  "message_id": "m2",
  "skill": "academic-paper-review",
  "content": "",
  "tool_calls": [
    {"id": "t2", "name": "web_search", "args": {"query": "..."}}
  ]
}
```

文本和工具调用任一存在即可评估。`build_judge_evidence(target="quick_turn")` 会：

- 从过程 `tool_chain` 中排除 quick turn 自己的调用批，避免同一调用重复计为过程证据；
- 把 `content + tool_calls` 一起序列化进 `quick_turn` 输出证据；
- 仍保留此前的 Skill 加载批作为过程证据。

Judge 的引用校验要求：有过程证据时至少引用一个 `tool_chain` / `error`，并始终引用一个
输出证据。真实 smoke 中出现过 Judge 只引用 `quick_turn`，导致
`judgment must cite tool chain or error evidence`。这属于 `judge_failure`，不是 adapter 丢失
quick turn。排查时先看 `quick_turn_missing`，再看 `judge_failure`，不要只看退出码 2。

### 4.6 子进程和超时

真实 DeerFlow runner 使用 `multiprocessing.get_context("spawn")` 隔离每个样本。这里有三个
已经验证过的死锁/泄漏风险：

- **先 `recv()`，再 `join()`。** 大 trace 可能塞满 pipe；先等子进程退出会形成互等。
- **收到结果不等于子进程已退出。** 返回 payload 后只给有限退出宽限，仍存活就 terminate，
  并记为 infrastructure failure。
- **原始 trace 边跑边写。** 如果只在 `build()` 时落盘，超时杀进程后恰好没有任何现场。

取消、超时、stream 异常和 `close()` 异常都必须回收子进程/生成器。不要把 partial trace
包装成成功；也不要因为缺少 `end` 又追加一个虚假的重复错误。

### 4.7 配置、成本与产物

- 命令从 `backend/` 运行；`AGENT_MODEL` 和 `JUDGE_MODEL` 都是必填项，前者还必须出现在
  DeerFlow `config.yaml` 的模型列表中。
- API Key 只放环境变量或受支持的 secret 配置中，绝不能硬编码进临时运行脚本。
- `config.yaml` 版本过旧会产生 upgrade 警告；先区分警告和真正的 preflight failure。
- 全量 routing 是 20 cases × 3 epochs = 60 个独立子进程。开发阶段先跑
  `--smoke --mode routing`（3 次），需要稳定性数据时才跑 60 次。
- quick smoke 的退出码 2 可能来自 Judge 引用失败；它不等价于 Agent 或 adapter 失败。
- `backend/eval-results/` 和 `backend/logs/` 是本地产物，均不应提交；前者已由根
  `.gitignore` 明确忽略。

### 4.8 验证清单

接入完成前按以下顺序验证，失败时更容易定位层级：

```bash
cd backend

# 1. 合成事件：不访问模型
uv run pytest tests/skill_eval/test_deerflow_adapter.py -q

# 2. Trace → Judge/Scorer 契约
uv run pytest tests/skill_eval/test_judge.py tests/skill_eval/test_quick_scorer.py -q

# 3. 全部 skill eval 回归
uv run pytest tests/skill_eval -q

# 4. 真实模型最小验证（3 个样本）
uv run python -m skill_eval.poc --smoke --mode routing
uv run python -m skill_eval.poc --smoke --mode quick
```

真实 quick smoke 至少检查以下字段，而不只看总退出码：

- 命中 Skill 的 case：`quick_turn != null`；
- 纯工具输出：`quick_turn.content == ""` 且 `quick_turn.tool_calls` 非空；
- `quick_turn_missing == 0`；
- `judge_failure` 与 `infrastructure_error` 分开统计；
- `raw_trace_ref` 指向存在的 JSONL，超时样本也能保留已产生的事件。

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
