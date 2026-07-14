# Judge 设计：用 LLM 评估 Agent 执行质量

## 问题

路由评测只能回答"Agent 选了哪个 Skill、选得对不对"，无法回答"选对之后执行得怎么样"。需要一个独立 Judge 来评估：

- 工具选择和使用是否合理
- 过程质量是否达标
- 最终输出是否可用

## 设计原则

1. **不信任 Agent 的自我声明。** Judge 只看 trace 里的可观测证据——真实工具调用名、参数、返回结果、最终答案。
2. **不推断未发生的步骤。** 如果 trace 里没有某一步，不能假设 Agent 私下做了。
3. **引用必须可追溯。** Judge 给出的每条理由和每条引用都必须指向 bundle 中真实存在的证据 ID。
4. **路由评测和过程评测分离。** 路由用确定性规则，质量用 LLM。

---

## 输入：JudgeEvidenceBundle

Judge 收到的是一整个证据包：

```
JudgeEvidenceBundle
├── user_input         原始用户请求
├── candidate_skills   两个候选 Skill 的 description（来自 SKILL.md frontmatter）
├── observed_route     Agent 实际加载了哪个 Skill（或 "unavailable"）
└── evidence[]         可观测执行证据，每项包含 id、kind、content、是否截断
```

evidence 来自 `AgentTrace`，按类别提取：

| kind | 来源 | 示例 |
|---|---|---|
| `message` | trace.messages | 每条 AI/tool 消息的完整 JSON |
| `tool_call` | trace.tool_calls | `{"name":"read_file", "args":{"path":"..."}}` |
| `tool_result` | trace.tool_calls | 工具调用的返回内容 |
| `error` | trace.errors | `"recursion limit reached"` |
| `artifact` | trace.artifacts | 文件路径、SHA-256、首尾内容 |
| `final_answer` | trace.final_answer | 模型最终产出的文本 |

**容量控制：**
- 单条 ≤ 12KB，超出截断
- 总 bundle ≤ 80KB，超出按顺序截断
- 截断会标记 `truncated=true`，Judge 知道信息不完整

---

## 评分维度

Judge 对三个维度分别打 0–4 分：

### Route Quality
```
评估 Agent 选这个 Skill 是否合理

- 单篇论文 → academic-paper-review
- 多篇综述 → systematic-literature-review
- 都不需要 → none

Judge 独立判断 recommended_route，不受 Agent 实际选择或数据集 label 影响。
```

### Process Quality
```
评估可观测的执行过程

- tool choice and ordering are coherent
- tool errors are handled rather than ignored
- repeated or unused calls are penalized
- final claims are supported by retrieved evidence
- final output agrees with the retained trace and artifacts
```

### Output Quality
```
评估最终答案的质量

- 是否完成、准确、格式正确
- 是否有 fatal error（致命错误导致输出不可用）
```

### 评分锚点
```
0: No evaluable result or completely wrong
1: Severe omissions or largely unusable
2: Partially satisfies with material problems
3: Satisfies with sound evidence and no major defect
4: Excellent, well-supported, efficient, and complete
```

---

## 输出：QualityJudgment

```json
{
  "recommended_route": "academic-paper-review",
  "route_quality": 3,
  "process_quality": 3,
  "output_quality": 0,
  "overall_quality": 2,
  "fatal_error": false,
  "reasons": [
    "agent loaded academic-paper-review correctly",
    "tool calls coherent, adapted to failures",
    "final_answer is empty, no review produced"
  ],
  "evidence": [
    "message[1]",
    "tool_call[0]",
    "final_answer"
  ]
}
```

### 校验规则

Judge 返回后有两层校验：

1. **Schema 校验：** JSON 必须符合 `QualityJudgment` schema，否则自动发 repair prompt 重试一次
2. **引用校验：** `evidence` 里的每个 ID 必须在 bundle 中真实存在，且必须同时包含 trace 证据（message/tool_call/tool_result/error）和输出证据（artifact/final_answer），缺失任一类型则拒绝

### 通过条件

```python
quality_passed = (
    not judgment.fatal_error
    and judgment.route_quality >= 3
    and judgment.process_quality >= 3
    and judgment.output_quality >= 3
)
```

三个维度全部 ≥ 3 才算通过。

---

## Judge Prompt 结构

```
System prompt（无）

User prompt:
  1. 任务说明："Evaluate only the observable behavior..."
  2. JSON schema（要求返回的格式）
  3. 三个 Skill 的 rubric（systematic-review / paper-review / none）
  4. 通用过程 rubric
  5. 评分锚点
  6. Evidence bundle（JSON，包含全部 4 类输入）
```

没有 system prompt，所有约束都在一条 user message 里，避免不同模型对 system role 的处理差异。

---

## 与路由评测的对比

| | 路由评测 | 质量 Judge |
|---|---|---|
| 回答什么问题 | Agent 选了哪个 Skill？ | 选对之后执行得好吗？ |
| 怎么判断 | 确定性规则（`observed == expected`） | LLM 读 trace 打分 |
| 证据 | 只关心 `read_file(SKILL.md)` | 全部消息、工具调用、结果、最终答案 |
| 调用 LLM | 不调用 | 每条 case 调用一次 |
| 失败含义 | 选错 Skill | 过程或输出质量不达标 |

---

## 实际运行示例

```
Case: "Review this paper: https://arxiv.org/abs/2310.06825"
Expected: academic-paper-review

Agent 执行:
  ✓ read_file(".../academic-paper-review/SKILL.md") → loaded
  ✓ web_fetch(paper_url) → Jina 401
  ✓ web_search(paper title) → 找到 ar5iv
  ✓ bash curl(ar5iv HTML) → 提取全文
  ✗ recursion limit → 未产出 final_answer

Judge 判决:
  recommended_route:  academic-paper-review
  route_quality:      3  ← "correct skill for single paper"
  process_quality:    3  ← "coherent, adapted to failures"
  output_quality:     0  ← "final_answer is empty"
  overall_quality:    2
  fatal_error:        False
  QUALITY PASSED:     False
```
