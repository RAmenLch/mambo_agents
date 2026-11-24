# SubAgent 内部错误包装与续连设计方案

> 状态：设计定稿（待实施）
> 前置依赖：《工具异常内化中间件设计方案》**已落地**
> （`middleware/tool_call_error.py`、`utils/cancellation.py`，v0.4.2b5）
> 目标仓库：`mambo_agents`
> 适用范围：同步子代理（`SubAgentMiddleware` / `task` 工具）

---

## 1. 概述

同步子代理在运行期发生内部异常时，当前会直接穿透 `task()` 冒泡到主图，导致主 Agent 当前
轮次整体失败，下游不得不做复杂的异常清理与外层恢复，且已完成进度无法接续。

本方案做两件事：

1. **错误内化**：子代理内部运行期异常被包装为结构化 `ToolMessage(status="error")`
   返回给主 Agent，不再击穿主图；
2. **链式续连**：每次失败在 `store` 中追加一个**增量快照节点**并返回该节点的 `node_id`；
   主 Agent（或下游）凭 `node_id` 再次调用 `task(...)` 时，沿链拼出完整历史，
   **新建一个全新子图**注入历史继续执行。

### 1.1 设计原则

| # | 原则 |
|---|---|
| P1 | **不触碰 LangGraph 内部机制**：不使用 `checkpoint_ns` 稳定化、`CONFIG_KEY_RESUMING`、`checkpoint_map`、`ReplayState` 等私有约定。续连只依赖 `store` 存取与子图正常 `stream`。 |
| P2 | **超步不得中途打断**：工具执行异常由工具层中间件收敛（`ToolCallErrorMiddleware`，**已落地**，主图位于 `BackendToolsMiddleware` 内层、子代理位于 middleware 链首），保证每一超步完整提交，历史永远 `AIMessage` / `ToolMessage` 成对。 |
| P3 | **控制流绝不吞掉**：中断信号与取消信号原样透传。 |
| P4 | **不依赖 LLM 自觉**：配额、错误分级、并发互斥全部在代码层强制，提示词只做引导。 |
| P5 | **快照生命周期跟随 `thread_id`**：所有快照挂在 `(thread_id, …)` 命名空间下，由下游按会话统一清理，本模块不提供删除接口、不做主动清理。 |
| P6 | **链式增量存储**：每个节点只保存本次执行新增的消息，历史沿链拼接，避免重复存储。 |

---

## 2. 目标与非目标

### 2.1 目标

| # | 目标 |
|---|---|
| G1 | 子代理内部异常不击穿主图，转为 `ToolMessage(status="error")` |
| G2 | 失败结果中给出 `node_id`，可自主续跑，已完成步骤不重做、副作用不重复 |
| G3 | 续连时允许携带新指令（纠偏 / 补充约束） |
| G4 | `GraphBubbleUp` / `GraphInterrupt` / `ParentCommand` 与 `asyncio.CancelledError` / `KeyboardInterrupt` / `SystemExit` 绝不吞掉 |
| G5 | 下游无需改动即可兼容，同时获得结构化元数据与查询能力 |
| G6 | 实现只依赖稳定公开 API，不受 LangGraph 版本升级冲击 |
| G7 | 同一条 `AIMessage` 被重复执行（历史重放、`/retry` 从 checkpoint 恢复）不产生重复快照节点（“重新生成”属新调用，见 §4.2） |

### 2.2 非目标

| # | 不做 | 说明 |
|---|---|---|
| N1 | 精确到"超步"的续跑 | 采用对话级重放；靠"工具结果已在历史中"保证副作用不重复 |
| N2 | 恢复非 `messages` 通道 | 子图自身的 `todos` / `plans` / `structured_response` / 中间件状态一律视为重置 |
| N3 | 恢复 `interrupt()` 挂起态 | 续连（注入历史新建子图）不恢复挂起态，视作放弃原审核。注意：**原生的 `interrupt()` + `Command(resume=...)` 本身就是支持的**，见 §4.7 |
| N4 | 跨 `thread_id` 共享同一条链 | 快照按 `(thread_id, …)` 命名空间隔离，不同会话互不可见；**同一会话内的树形分叉是允许的**（不做分支校验，见 §4.5） |
| N5 | 无限次自动续连 | 由配额与熔断约束（§6：链深 + 分叉数两个维度） |
| N6 | 快照的主动清理与压缩 | 单个节点不可删；清理只发生在"同一 `tool_call_id` 重新执行"（§4.3.1），其余跟随 `thread_id` |

---

## 3. 前置依赖

本方案的正确性依赖两个前提。其中“工具执行异常内化”由独立中间件承担，
机制与配置见《工具异常内化中间件设计方案》；本节只声明依赖关系与子代理侧的装配要求。

### 3.1 依赖：工具执行异常内化中间件

**要求**：工具执行期异常不得中止超步，必须以 `ToolMessage(status="error")` 的形式
成为该次工具调用的结果。

**与本方案的关系**：本方案保存的 `delta_messages` 取自“最后一个已提交超步”的状态。
若超步因工具异常而中止，该超步内已成功的兄弟工具结果不会进入状态，
`delta` 就会缺失这部分内容，续连后的历史不自洽。

**装配现状（已落地）**：

| 位置 | 挂载点 | 位置选择理由 |
|---|---|---|
| 主图 | `graph.py` `create_mambo_agent()` 的 `mw` 列表，`BackendToolsMiddleware` **之后**（内层） | `BackendToolsMiddleware` 会在工具**成功之后**做大结果驱逐；若本中间件在其外层，驱逐失败会把已成功的工具结果替换成 error `ToolMessage`，模型会重做已产生副作用的操作 |
| 子代理 | `subagents.py` / `async_subagents.py` 的子代理 middleware **链首** | 子代理栈没有 `BackendToolsMiddleware`，不存在上述顾虑；放链首可连带兜住用户自定义中间件抛的异常 |

- 未生效时本方案仍可运行，但 delta 可能缺失兄弟工具结果（见 §9）；
- 该中间件**达到连续失败阈值不会重新上抛**（只升级日志与 `escalated` 标记），
  因此不会反过来破坏 P2。

### 3.2 子代理侧中间件链（现状与待补）

**现状（已核实代码）**：子代理 middleware 列表为

```text
ToolCallErrorMiddleware → *spec.middleware → (可选) HumanInTheLoopMiddleware
```

见 `middleware/subagents.py:755`、`middleware/async_subagents.py:949`。

**`PatchToolCallsMiddleware` / `ReorderToolMessagesMiddleware` 当前并未挂在子代理栈上**
（它们只存在于主图 `graph.py` 的链尾）。因此：

- 初版本文档写的
  “`ToolCallErrorMiddleware → PatchToolCallsMiddleware → ReorderToolMessagesMiddleware`”
  与代码不符，已删除；
- 若判定子代理也需要“悬空调用修补 + 消息重排”这层兜底，须作为**新增项**落地
  （见 §10.2 第 5 条），并评估两点影响：
  1. `PatchToolCallsMiddleware` 用 `Overwrite` 重写整个 `messages`，在子代理内每轮
     `before_agent` 都会全量扫描——成本随历史增长；
  2. 它依赖 `runtime.store`（§7.4）才能给 `task` 悬空调用补 `continue_id`，
     挂载后需同步确认 store 在子代理运行时可用。

对 P2 而言**不构成阻塞**：每超步不被打断由 `ToolCallErrorMiddleware` 单独保证
（`ToolNode` 的 `wrap_tool_call` 链上没有任何其它中间件，因此不存在“漏挂”问题）。

---

## 4. 核心机制

### 4.1 为什么“注入历史重放”等价于续跑

续跑的唯一实质目的，是**不重复执行已经做过的事**。本方案免费获得这一点：

- 历史中已存在 `AIMessage(tool_calls=[grep])` + `ToolMessage("grep 结果…")`；
- 新子图第一次 `model` 调用看到这段历史，不会重新 `grep`；
- 有副作用的操作同理：模型看到 `write` 的成功回执，不会再写一次。

模型视角与真断点续跑几乎等价；差异仅在于图的 `step` 从 0 开始，
以及非消息通道被重置（§2.2 N2）。

### 4.2 链式快照模型

#### store 布局

```text
namespace: (thread_id, "mambo_subagent_history")
key:       <node_id>
```

命名空间顶层即 `thread_id`，与该会话的其它数据（如 `(thread_id, "mambo_fs")`）
处于同一前缀下，便于下游按会话整体清理。

#### 节点结构

```python
class SubAgentSnapshotNode(BaseModel):
    node_id: str                      # = 产生本节点的那次 task 调用的 tool_call_id
    parent_id: str | None             # 根节点为 None
    chain: list[str]                  # 祖先 node_id 有序列表，不含自身（天然支持树/分叉）
    depth: int                        # = len(chain) + 1，即“这份进度是第几手”
    subagent_type: str
    description: str                  # 本次执行携带的指令原文（根节点为首个任务描述）
    status: Literal["failed", "cancelled"]
    error_type: str
    error_message: str                # 截断后错误详情（≤1000 字符）
    retryable: bool
    available: bool                   # False → 不下发 continue_id（不可重试/超限）
    delta_messages: list[dict]        # ★ 仅本次执行新增的消息（messages_to_dict 序列化）
    steps: int                        # 本次执行完成的超步数（= values chunk 数）
    injected_prefix_len: int          # 本次注入的历史长度（调试用）
    created_at: str
```

#### `node_id` 生成规则

`node_id = runtime.tool_call_id`，服务端取值，**不接受 LLM 传入**。

这条规则 + 下面的“入场清理”共同保证一个不变式：

> **`store` 中某个 `node_id` 下的记录，永远只代表“该次 tool_call 最近一次执行”的结果。**

- 父图历史重放、`/retry`（从 checkpoint 恢复）重跑同一超步时，消息中的
  `tool_call_id` 保持不变 → `node_id` 相同；
- 若改用随机 ID，每次重跑都会追加节点，链会无限膨胀。

**注意“重新生成”不属于同一调用**：mambochat 里除了
`POST /messages/{id}/retry`（恢复执行，id 不变），还有“编辑消息 / 重新生成 AI 回复”
这类接口（`chat_interaction.py:406`、`:731`）—— 它们会重新调用模型产出**新的
`AIMessage`**，`tool_call_id` 也随之变新，因此属于新调用、理应产生新节点。
去重只在“同一条 `AIMessage` 被重复执行”这一层成立。

#### delta 计算规则（按消息 id 做差集，不用下标）

续连时注入的是 `[*history, HumanMessage(续接指令)]`，跑完后：

```python
injected_ids = {m.id for m in injected if m.id}
delta = [m for m in full_messages if m.id not in injected_ids]
```

**不能用 `full_messages[len(injected) - 1:]` 这种下标切片**：子代理一旦挂了
摘要中间件（`MamboSummarizationMiddleware`，方案自己在 §9 推荐子代理使用）或
任何用 `Overwrite` 重写 `messages` 的中间件，列表长度会变，下标切片会切空或错位，
**进度被静默丢弃而凭证看起来完全正常**。

- 消息 id 缺失时（`id is None`）退化处理：先按下标定位，再按对象 `is` 身份做差；
- 续接指令本身会作为新消息出现在 `full_messages` 中，因此天然进入 delta，
  链的拼接结果完整、可读；
- `injected_prefix_len = len(injected)` 仅用于排查。

#### 链的读取

```python
node = store.get(ns, node_id)
nodes = [store.get(ns, nid) for nid in node.chain] + [node]     # 按祖先顺序
history = [m for n in nodes for m in n.delta_messages]
```

- 任何一个祖先节点缺失即为 `chain_broken`，**必须显式报错**，
  不得静默拼出残缺历史（半截上下文比报错更危险）。
- 常见深度 ≤ 3，逐次 `get` 的成本可接受。
- **节点一旦写入就不再修改、不删除、不压缩**；清理只发生在“同一
  `tool_call_id` 重新执行”这一条路径上（见 §4.3 入场清理），
  其余交由 `thread_id` 生命周期整体清理。
- 一个节点可以有**多个子节点**（从同一断点分不同方向续连），`chain` 记录祖先路径，
  因此链天然是一棵树，不需要额外字段。

#### 关于“封链标记 / 已成功即失效”：已废弃

初版方案设计了 `status="sealed"` + `already_done` 校验，用来阻止“续连成功后旧
`node_id` 再被使用”。**该机制已删除**，原因：

1. **父节点的 `delta` 从不过期**：它是一份“某时刻的进度快照”，记录的是当时真发生过的事，
   本身没有被篡改；从它再开一条新分支（换个思路继续）是**合法用法**；
2. 封链若写回同一 key（初版写法），会把祖先的 `delta` 抹掉，而 `chain_broken`
   只检查“节点是否存在” → 会静默产出半截历史，正好违反本节第一性要求；
3. “同一调用重复执行”导致的陈旧记录，由 §4.3 的入场清理一次性解决，
   不需要额外的状态机状态。

### 4.3 失败路径

```text
0. 入场清理：store.delete(ns, runtime.tool_call_id)      ← §4.3.1，在校验之后、执行之前
1. task(description, subagent_type)
2. 校验类型；subagent_state = {"messages": [HumanMessage(description)]}
3. 子图 stream（保持 LangGraph 默认 checkpoint_ns，不做 override）
4. 每收到 "values" chunk → 记录 final_state（= 最后一个已提交超步）
5. 若抛 GraphBubbleUp 或非 Exception 的 BaseException → 直接 raise（§5.1）
6. 若抛普通 Exception 且 catch_errors=True：
   a. 取 final_state["messages"]；为空则回退为 [HumanMessage(description)]
   b. 计算 error_type / retryable / steps / available
   c. 组装节点（node_id = runtime.tool_call_id，parent/chain 由续连上下文提供）
   d. store.put(namespace, node_id, node)          ← 先落盘
   e. store.get 回读校验：写失败/写被取消 → available=False（不下发 continue_id）
   f. 复用 _return_command_with_state_update 的 passthrough 语义返回 Command，
      而不是只塞 messages —— 见 §4.3.2
```

#### 4.3.1 入场清理（替代封链机制）

`task` 进入执行前，**删掉自己这个 `tool_call_id` 对应的记录**：

```python
store.delete(_snapshot_namespace(thread_id), runtime.tool_call_id)
```

- 位置：**参数校验之后、开始执行之前**。若放在校验前，遇到 `not_found` /
  `type_mismatch` / `busy` 这类“只返回错误、不写节点”的分支时，会把上一次的记录
  删掉却不写新的，凭空丢一份凭证；
- 只删自己这一个 key，**绝不触碰 `chain` 上的祖先**（拼历史还要用它们）；
- 效果：本次成功 → key 下无残留；本次失败 → 写入全新记录。三件事一次解决：
  幂等（重跑不叠加）、覆盖（旧 delta 自然失效）、清理（陈旧记录不残留）。

#### 4.3.2 失败路径必须 passthrough 通道更新

成功路径走 `_return_command_with_state_update()`（`subagents.py:381`），它会把子图
final_state 中除 `_EXCLUDED_STATE_KEYS` 之外的键一起透传回主图。失败路径若只手写
`{"messages": [ToolMessage(...)]}`，会丢掉这部分通道更新 → 与成功路径行为不一致。

- 失败路径复用同一 helper（或抽出公共函数），只把 `messages` 换成本次的 error
  `ToolMessage`；
- 同样要过取消守卫（§5.3），且**不得**把 `HumanMessage(description)` 的原文覆盖掉。

**错误文本（可续连）**

```text
Subagent 'researcher' failed with TimeoutError: Request timed out.
Progress before failure: 3 step(s) in this run.
Continuation saved: node_id="call_bbb"

To continue without repeating completed steps, call:
  task(subagent_type="researcher", continue_id="call_bbb")
If the error looks persistent (configuration / permission / repeated failures),
do not keep retrying — report the error and node_id to the user.
```

**错误文本（不可续连）**

```text
Subagent 'researcher' failed with BadRequestError: 400 invalid message structure.
This error is not retryable. Do not retry with continue_id; report it to the user.
```

### 4.4 续连路径

```text
1. task(subagent_type, continue_id="call_bbb", description="" | "补充指令")
2. 读节点 → 按 §4.5 校验；任一失败 → 返回明确 error ToolMessage
   （校验不通过时【不做】入场清理，本节点记录保持原样）
3. 入场清理：store.delete(ns, runtime.tool_call_id)   ← 删自己，不删 continue_id
4. 沿 chain 拼历史：
     nodes   = [node(chain[0]), …, node(chain[-1]), node(continue_id)]
     history = [m for n in nodes for m in n.delta_messages]
     inject  = [*history, HumanMessage(续接指令)]
5. 续接指令（固定措辞，抑制重复副作用）：
     "上次执行因 <error_type> 中断。以上为已完成步骤的记录，请勿重复执行；
      从断点继续完成任务：<description 或节点原始描述>"
6. 新建子图执行：
     subagent_state = {"messages": inject, **父图非排除通道}
     · 不使用 Overwrite
     · 不 override checkpoint_ns（沿用继承值；挂起/失败重跑时子图会从自己的存档继续）
7. 成功 → 返回正常 ToolMessage（**不写、不改任何节点**：父节点的 delta 仍然有效，
         可随时从它再开新分支）
   失败 → 回到 §4.3，新节点 node_id = 本次 tool_call_id，
          parent_id = continue_id，chain = [*parent.chain, parent.node_id]
```

**关于“只传 messages”的取舍（已按现状拍定）**：现有
`_validate_and_prepare_state()`（`subagents.py:496-515`）会把父图 state 中
**非** `_EXCLUDED_STATE_KEYS` 的通道复制给子代理，并把父图 `configurable`
整体透传（含 `thread_id` / `checkpoint_ns`）。续连**沿用该行为**，理由：

- 与新建任务保持一致，避免两套 state 构造逻辑；
- `_EXCLUDED_STATE_KEYS` 已排除 `messages` / `todos` / `plans` /
  `structured_response` 等会话级通道，续连不会把主图的 todo 混进子图；
- 逐字「只传 messages」会让子代理自定义中间件依赖的通道消失，反而更不安全。

### 4.5 续连校验状态机

| 校验项 | 状态 | 处理 |
|---|---|---|
| 既无 `continue_id` 又无 `description` | `missing_description` | error：新建任务必须提供 `description` |
| 节点不存在 | `not_found` | error：未找到该 `continue_id`，请核对 `continue_id` 与 `subagent_type` |
| `node.subagent_type != subagent_type` | `type_mismatch` | error：类型不匹配，拒绝跨类型复用 |
| 链上任一祖先节点缺失 | `chain_broken` | error：续连链不完整，请从最近的可用 `node_id` 续连或重新开始 |
| `node.available is False` | `not_available` | error：该断点不可续连（不可重试或已超限），请换策略 |
| `depth > max_continue_attempts` | `exhausted` | error：已达续连上限，请换策略或上报用户 |
| 该节点已被续连次数 ≥ `max_children` | `branched_out` | error：该断点已从不同方向续连过 N 次，仍失败请换子代理或重新开始 |
| 该节点已有执行中的子任务 | `busy` | error：该任务正在执行中，拒绝并发 |
| 全部通过 | `available` | 正常续连 |

**设计说明**：

- **没有 `already_done`**：父节点的 delta 从不过期，从同一个断点再开新分支是合法用法
  （见 §4.2 末段）；
- **没有 `branch_mismatch`**：初版用根节点的 `root_checkpoint_id`
  比对父分支，既依赖 `configurable.checkpoint_id`（语义未验证、retry 后可能变），
  又与“允许分叉”自相矛盾，**已删除**。链之间的隔离只靠 `thread_id` 命名空间；
- **`branched_out` 是替代品**：真正需要防的是“同一断点被反复续连、每次都失败”
  （`depth` 管不到，因为每个子节点的 `depth` 相同）。`max_children` 默认 `2`，
  用 `store.search(ns)` 客户端过滤 `parent_id == node_id` 统计（节点不可变，不在父节点
  上维护计数）；
- `depth` 校验的对象是**被续连节点的 `depth`**，不是新节点的。

### 4.6 并发互斥与执行次序

`create_agent` 每个 tool call 一个独立 Pregel task 并行执行。若 LLM 一次输出两个相同
`continue_id`，两个子图会并发基于同一父节点续跑，产生分叉。

- `task()` 内维护进程级在飞集合，**键为父 `node_id`**（新建任务无父，天然隔离）；
- **必须“锁内 check-and-add”**：先占位再校验，校验失败回滚 —— 否则
  “校验 → 入集合”之间有窗口，两个并发请求会同时通过；
- 退出（含异常）必须在 `finally` 中 discard。

**完整执行次序**（sync / async 一致）：

```text
参数校验（§4.5 状态机）→ 在飞集合占位 → 入场清理(delete 自己)
→ 拼历史 → 子图 stream → 成功:直接返回 / 失败:写节点后返回
→ finally: 在飞集合 discard
```

## 4.7 子代理的 checkpoint：共享父图、按 ns 恢复（实测结论）

这一节是实施阶段实测出来的事实，**直接影响续连的正确性**，必须先理解它。

### 4.7.1 事实

`SubAgent` 由 `_langchain_create_agent(...)` 创建，**没有自己的 checkpointer**
（`subagents.py:757-770`，`subagent.checkpointer is None`）。但 `task` 工具在节点内调用
子图时，运行期 config 从父节点继承了一整套隐藏键，其中包含父图的 checkpointer：

```text
config["configurable"] = {
    "thread_id": "<父线程>",
    "__pregel_checkpointer": <父图的 checkpointer>,
    "checkpoint_ns": "call_task:<task_id>",      ← 节点名:任务id
    "checkpoint_map": {...}, "checkpoint_id": None,
    "__pregel_scratchpad": PregelScratchpad(...),
    "__pregel_resuming": True (仅 resume 时),
    ...
}
```

实测（主图 `InMemorySaver` + 子图 `compile()` 无 checkpointer + 子图内部 `interrupt()`）：

| 观测 | 结果 |
|---|---|
| 第一次 invoke | 主图返回 `__interrupt__`，子图的 `work` 节点执行 1 次 |
| `Command(resume="yes")` | 子图 **`work` 仍未重跑（计数仍为 1）**，只有承载 `interrupt()` 的那个节点带 `__pregel_resuming=True` 重跑 |
| 结论 | **子代理的 `interrupt()` + 审核恢复是原生支持的，且是原地恢复** |

即：子图通过继承的 checkpointer 把自己的超步写进父线程下、`checkpoint_ns` 为
`<节点名>:<task_id>` 的命名空间，resume 时从那里恢复，已完成超步不重跑。

### 4.7.2 对本方案的直接影响

1. **`/retry`（父图 `invoke(None)` 恢复）时，`task_id` 不变 → `checkpoint_ns` 不变 →
   子图从自己的存档继续**。实测（子图三步 `n1→n2→n3`，第 3 步抛异常）：

   ```text
   第一次执行后：n1=1, n2=1, n3=1（失败）
   /retry 之后：  n1=1, n2=1, n3=2   ← 只有失败那一步重跑
   ```

   **已完成的超步不会重做**。父图只重跑“未提交的那个 tools 超步”，
   而子图内部按自己的 checkpoint 只重跑“未完成的那一步”。

2. **重跑时子图会忽略本次注入的 `HumanMessage`**：实测第二次执行时
   config 里带 `__pregel_resuming=True`，`task` 工具新建的那条
   `HumanMessage(description)` 被丢弃，子图历史仍只有 1 条。因此
   **不会出现“两条相同指令”**（§4.7.3 记录了这条错误推断的推翻过程）。

3. ~~续连必须 override `checkpoint_ns`~~、~~注入消息必须带确定性 id~~ ——
   两条都**撤销**：真实路径（挂起 resume / 失败后 `/retry`）都不会重复注入。
   只有在“同一个 ns 被非 resume 方式重复使用”时才会；当前的调用方式下不存在这种路径。

4. 副作用：子代理的每个超步都会在**父线程**里留下嵌套 checkpoint。
   长会话 + 多子代理时存储会明显增长——这是既有行为（不是本方案引入），
   但续连会增加超步数，值得在容量评估时考虑。

### 4.7.3 曾经怀疑的“重复指令”问题：已排除（保留结论与证据）

> 这一段保留下来是为了避免后人（包括我）重复同一个错误推断。

**曾经的说法**：`task` 每次调用都新建 `HumanMessage(description)`，同一超步重跑会让
子图历史里出现两条相同指令。**实测证明：真实路径下不会。**

| 实验设置 | 结果 |
|---|---|
| 手工把子图的 `checkpoint_ns` 固定，直接调两次 `subagent.invoke(fresh HumanMessage)` | 出现 2 条相同指令 ← **这是人造场景** |
| 真实路径 A：`interrupt()` 挂起 → `Command(resume=...)` | 1 条（挂起型 resume 忽略新输入） |
| 真实路径 B：父图失败 → `invoke(None)` 恢复（= 下游 `/retry`） | 1 条（config 带 `__pregel_resuming=True`，新输入被丢弃） |

结论：两条真实路径都不会重复，**不需要给注入消息加确定性 id**。
若将来出现“同一 ns 以非 resume 方式被复用”的调用形态（例如自己拼 config 手动重放），
才需要 `id=f"task:{tool_call_id}"` 这种防御。

---

## 5. 异常分类与透传边界

### 5.1 透传 / 捕获边界

| 异常类别 | 典型来源 | 处理策略 |
|---|---|---|
| `GraphBubbleUp`（含 `GraphInterrupt`、`ParentCommand`） | 子代理内部工具审核（`AutoSecurityReviewMiddleware` / `HumanInTheLoopMiddleware`）、`ask_user` | **直接 raise 透传**，冒泡到主图暂停，等待 `Command(resume=...)` |
| `asyncio.CancelledError`、`KeyboardInterrupt`、`SystemExit`、`GeneratorExit`（非 `Exception` 的 `BaseException`） | 前端「停止生成」、进程退出 | **不捕获，自然透传**（可选写 `status="cancelled"` 节点，见 §6.3） |
| **取消信号的包装变体**：被工具或 HTTP 客户端转成普通 `Exception` 的取消 | 同上 | **上抛**，靠 `is_cancellation_related(exc)` 守卫识别（见 §5.3） |
| 普通 `Exception` 且 `catch_errors=True` | LLM 超时 / 429 / 5xx / 断流、子图级异常、`GraphRecursionError` | 捕获并包装为 `ToolMessage(status="error")` |
| 普通 `Exception` 且 `catch_errors=False` | 同上 | 直接 raise 透传，保留调试直抛能力 |

> **工具执行异常不会到达 `task()`**：子代理自己的 `ToolNode` 外面已挂
> `ToolCallErrorMiddleware`（§3.2），工具异常在工具层就被内化成
> `ToolMessage(status="error")`，子代理继续跑。因此本表 `catch_errors` 分支捕获的是
> **LLM 调用异常与图级异常**，而不是工具异常——这也意味着“一次工具失败”不应产生
> 续连节点，只有真正无法继续的失败才写节点。

### 5.2 错误可重试性分级

决定是否下发 `continue_id`。

| 类别 | `retryable` | 是否给 `continue_id` |
|---|---|---|
| `TimeoutError` / 连接错误 / 断流 / 429 / 5xx | `True` | ✅ |
| 上下文超限类报错 | `False` | ❌ |
| `400 BadRequest`（含消息结构非法） | `False` | ❌ |
| `401` / `403` / 权限、配置类错误 | `False` | ❌ |
| `GraphRecursionError` | `False` | ❌（续连只会白送新一轮预算） |
| Pydantic 参数校验失败 / `ToolInvocationError` | `False` | ❌ |
| 其他未识别异常 | `False` | ❌（**已订正**：原写 `True`，会诱导模型在未知失败上反复续连） |

**实现方式（直接复用，不另写一套）**：

```python
from mambo_agents.middleware.tool_call_error import _is_retryable
```

判定逻辑见《工具异常内化中间件设计方案》§4.3，要点：

1. **按 HTTP 状态码数值判断**（`status_code` / `response.status_code`），不做类名子串
   匹配——实测 `httpx.HTTPStatusError(429)` 与 `openai.RateLimitError(429)` 语义相同，
   旧的字符串判据会给出相反结论；
2. 沿**整条异常链**（含 `__cause__` / `__context__`）匹配，`APITimeoutError →
   httpx.ReadTimeout → TimeoutError` 这类包装也能识别为可重试；
3. 未知异常默认 `False`。

两处（工具层内化 / 子代理续连）必须共用同一实现，否则会出现“工具层判可重试、
续连层判不可重试”的分裂。`_is_retryable` 目前是 `tool_call_error` 的私有函数，
落地本方案时把它提升为公开工具函数（或迁到 `utils/`）即可，不要复制。

### 5.3 取消守卫

`task()` / `atask()` 的 `except Exception` 必须**先经过取消守卫再决定是否包装**：

```python
except GraphBubbleUp:
    raise
except Exception as exc:
    if is_cancellation_related(exc):        # ← 必须在 _on_failure 之前
        raise
    return self._on_failure(subagent_type, exc, ...)
```

原因：取消信号要穿过工具、HTTP 客户端、序列化层才到 `task()`，任一层把它转成普通
`Exception`，`_on_failure` 就会**写一份失败节点 + 返回 error `ToolMessage`**，
主图因而在用户「停止生成」之后继续执行，并留下一条误导性的续连凭证。

- 守卫实现位于 `mambo_agents/utils/cancellation.py`（**已落地**），与工具异常内化
  中间件共用，只依赖标准库；
- 同步路径无运行事件循环时自动退化。

#### 5.3.1 判据语义（必须按此理解，否则本方案在“超时”场景整段失效）

`is_cancellation_related(exc)` 的实际语义是：

| 判据 | 优先级 | 说明 |
|---|---|---|
| `asyncio.current_task().cancelling() > 0` | **主** | 真实取消；anyio 在 cancel scope 退出时会 `uncancel()`，deadline 型取消不残留 |
| 异常链上有 `CancelledError`，但**同时有** `TimeoutError` | 判为**超时**，非取消 | `httpcore` 用 `anyio.fail_after()` 实现超时，`httpx.ReadTimeout` / `openai.APITimeoutError` 的链上**必然**带 `CancelledError(reason: deadline exc)` |
| 异常链上只有 `CancelledError`（`raise X from CancelledError`） | 判为取消 | 工具自己把取消包了一层 |

实测证据见《工具异常内化中间件设计方案》§4.4.3。

**为什么这条对本方案致命**：若把超时误判为取消，§5.2 里唯一稳定的
“可续连”类别（`TimeoutError` / 连接错误 / 429 / 5xx）会全部走上抛路径 ——
即不写节点、不下发 `continue_id`，续连机制在最需要它的场景下完全不触发。
因此调用方**不要**自己实现“链上有 `CancelledError` 就算取消”的简化版判据。

- 该函数只在**工具自己的 task** 里判定才准确（异步 `ToolNode` 用 `asyncio.gather`，
  每个工具一个独立 Task）。`task()` 工具本身就是一个普通工具调用，因此
  `except` 里直接判定即可；不要在父图/外层再判一次。
- 同步路径（`task()` 跑在线程池里，没有事件循环）只能退化为异常链判定：
  此时若工具把取消吞掉后再抛一个与取消无关的异常，是**识别不出来**的。
  同步链路下"停止生成"通常不走 asyncio 取消，实践影响有限；若同步链路也要支持，
  需要在上层显式传递取消标志（本方案不做）。

---

## 6. 熔断与配额

### 6.1 次数配额（三个维度，缺一不可）

| 维度 | 参数 | 默认 | 判定依据 | 防的是 |
|---|---|---|---|---|
| 链深 | `max_continue_attempts` | `3` | 被续连节点的 `depth = len(chain) + 1` | 一条链越续越深、无限往下拖 |
| 分叉 | `max_children` | `2` | 该节点已有的子节点数 | 同一断点被反复从不同方向续连、每次都失败 |
| 新任务 | `max_root_failures_per_thread` | `10` | 该会话内 `parent_id is None` 且 `failed` 的节点数 | 模型绕过续连、对同一任务反复发起全新的 `task`（见 §6.3）。**只影响“要不要留快照 + 文案提示”，不限制执行** |

只做链深不够：每次“从同一个 A 分叉”产生的新节点 `depth` 都相同，链深永远不涨。

**子节点数怎么统计**：节点一旦写入就不可变（§4.2），所以**不要**在父节点上维护
`child_count` 字段；改为 `store.search(ns)` 取回后按 `parent_id == node_id` 客户端过滤计数
（一个会话的节点数是可控的）。同理 `max_root_failures_per_thread` 也走客户端过滤。

超限时不下发 `continue_id`，错误文案写死：“已达续连上限 / 该断点已续连过
N 次 / 本会话同类任务已失败 N 次，不要再次续连，请换子代理或上报用户”。

### 6.2 零进展检测

`steps` 的**精确定义**：本次执行收到的 `values` chunk 数量
（LangGraph 每提交一个超步发一次 `values`，因此 `steps` == 已提交超步数；
失败的那个超步不计入，因为它的写入没有提交）。

比较当前节点与父节点的 `steps`：若父节点已推进若干超步、本次续连 `steps == 0`，
说明原地打转 → 提前熔断，不必等满配额。**不要**用 `len(messages)` 或 AIMessage 数量
代替，否则摘要中间件一压缩就会误判。

### 6.3 新建任务（无 `continue_id`）的快照上限

上面两个维度只约束“续连”，**根节点（新建任务）不受任何约束**：模型完全可以对同一个
失败任务反复发起**全新** `task(description=...)`（新 `tool_call_id`、新 `ns`），
每次都写一个根节点，`depth` 恒为 1、`max_children` 也管不到。

**注意：这不是执行次数限制**。任务该跑还是跑，`recursion_limit` 才是防跑飞的闸门；
这里只约束“**同一个会话里堆多少份失败快照**”：

- `max_root_failures_per_thread`（默认 `10`）：按 `(thread_id)` 统计
  `parent_id is None` 且 `status="failed"` 的节点数，达到上限后**该次失败不再写节点**
  （仍照常返回 error `ToolMessage`），并在文案里说明“本会话同类任务已失败 N 次，
  快照已不再保存，请换子代理或上报用户”；
- 目的只有两个：store 不被无限堆、模型得到"别再这样试"的明确信号；
- 该计数不需要额外存储，用 `store.search` 客户端过滤即可（节点数不多）。

### 6.4 取消节点

用户点「停止生成」时 `CancelledError` 默认透传、不写节点。

若确需“停止后仍能续跑”，可在 `task()` 的 `finally` 中补写一条 `status="cancelled"` 节点，
但必须注意两点：

1. **取消路径上的 `await` 会被立刻再次取消**（任务已处于 cancelling 状态），
   store 写入、checkpoint 落盘都可能静默失败。要么用 `asyncio.shield(...)` 包住，
   要么放弃落盘（推荐后者：取消即不产生续连凭证，语义最清晰）。
2. 写节点前同样要过取消守卫（§5.3），避免把“取消”误记成“失败”。

---

## 7. 下游对接能力

### 7.1 结构化元数据

不依赖正则解析自然语言。

```python
ToolMessage(
    content=<人读文本>,
    name="task",
    tool_call_id=runtime.tool_call_id,
    status="error",
    additional_kwargs={
        "subagent_continuation": {
            "continue_id": "call_bbb",
            "subagent_type": "researcher",
            "status": "failed",
            "depth": 2,
            "max_attempts": 3,
            "max_children": 2,
            "child_count": 0,        # 仅供下游展示，由 search 统计得出，不落库
            "retryable": True,
            "error_type": "TimeoutError",
            "steps": 2,
            "available": True,      # False → 不可续连（不可重试 / exhausted / branched_out / 落盘失败）
            "reason": None,         # available=False 时给出状态机里的原因码
        }
    },
)
```

`status` 只有 `failed` / `cancelled` 两种（没有 `sealed`，见 §4.2 末段）。

### 7.2 自定义事件

- 失败当时发 `type="subagent_continuation"`，字段同 §7.1 + `tool_call_id` / `timestamp`；
- 成功续连时，在常规 `subagent_event` 中补充 `root_node_id`，
  供下游把一条链上的多次调用归为同一条轨迹（UI 归组、跨轮次追溯）。

### 7.3 查询 API

```python
async def aget_subagent_node(store, thread_id, node_id) -> SubAgentSnapshotNode | None
async def aget_subagent_chain(store, thread_id, node_id) -> list[SubAgentSnapshotNode]
async def alist_subagent_roots(store, thread_id) -> list[SubAgentSnapshotNode]
```

- `aget_subagent_chain`：返回从根到指定节点的完整链，供下游渲染“执行轨迹”
  （因为节点永不修改、永不删除，轨迹永远完整）；
- `alist_subagent_roots`：返回该会话所有链的根节点，供页面刷新 / 服务重启后
  列出“有哪些子任务、各自续跑了多少次、当前是否可继续”；
- **不要依赖 `store.search(filter=...)`**：各后端（InMemory / Sqlite / Postgres）
  对 filter 的支持不一致。`search(namespace_prefix)` 取回后用 Python 侧过滤
  （一个会话的节点数是可控的），必要时再优化。

不提供按节点删除的接口：节点一旦写入即不可变（§4.2），生命周期完全跟随 `thread_id`，
下游删除会话时按 `(thread_id, "mambo_subagent_history")` 前缀整体清理即可。
初版的 `purge_sealed()` 也随 `sealed` 概念一并删除。

### 7.4 与悬空调用修补联动

当修补的悬空工具调用是 `task`，且 store 中存在该 `tool_call_id` 的节点时，
在合成 `ToolMessage` 的 `additional_kwargs` 中附带 `subagent_continuation`
（`available=True`），并在文本中提示可续连。
覆盖“用户点停止后直接发新消息（未点 /retry）”这一路径的信息丢失。

读取方式：`PatchToolCallsMiddleware.before_agent(state, runtime)` 中的 `runtime.store`。

**坑（必须处理）**：`before_agent` 是同步方法，而异步图里对 `BaseStore` 调**同步**
`get()` 在部分后端（Postgres / 异步适配层）不可用或会阻塞事件循环。

- 需要同时实现 `abefore_agent(state, runtime)`，异步路径走 `await store.aget(...)`；
- 读不到 store（或读失败）时**静默跳过**注入即可，修补本身不能因此失败；
- 只有该 `node_id` 的节点存在且 `available=True` 时才注入，否则维持原样的
  “was cancelled” 文本。

---

## 8. 边界场景矩阵

| 场景 | 预期行为 |
|---|---|
| 子代理第 2 步 LLM 超时 | 超步 1 已提交 → 节点含第 1 步结果；`continue_id` 可用；续连后第 1 步不重跑 |
| 子代理第 2 步 LLM 超时后用户点 `/retry` | 仅当异常未被内化（`catch_errors=False`）、整轮 FAILED 时才走这条路：子图按 ns 从自身存档**只重跑失败那一步**（实测 `n1=1,n2=1,n3=2`），`task` 注入的消息被 `__pregel_resuming` 丢弃，不会重复。走内化路径时整轮不 FAILED，只能用 `continue_id` |
| 并行两个工具，A 成功 B 抛异常 | 前置依赖生效后超步正常提交（B 为 error ToolMessage）→ 无孤立 `tool_call`，delta 自洽。**已实测**：无中间件时 checkpoint 只剩 `AIMessage`，兄弟结果丢失 |
| 子代理内部工具超时（`httpx` / vendor SDK） | 工具层内化为 error `ToolMessage`，子代理继续；**不写续连节点**、`task()` 看不到该异常 |
| 子代理内部工具连续失败 | 工具层仍只内化（阈值仅升级日志 + `escalated=True`），不会上抛到 `task()` |
| 子代理内部工具抛代码缺陷（`TypeError` / `InvalidUpdateError` …） | 工具层**不内化**，原样上抛 → 到 `task()` 的 `except`；`retryable=False`，不下发 `continue_id` |
| 子代理自身 LLM 超时 / 429 / 5xx | 到 `task()` 的 `except`，`retryable=True` → 写节点 + 下发 `continue_id` |
| 第二次续连再次失败 | 新节点 `node_id` = 本次 `tool_call_id`，`chain` 追加父节点；三次续连历史 = 三个 delta 拼接 |
| 同一超步被 `/retry` 重跑（恢复执行） | `tool_call_id` 不变 → 入场清理先删旧记录 → 成功则不残留、再失败则写全新节点 |
| 「编辑消息 / 重新生成 AI 回复」后再次调用 `task` | 这是**新调用**（新 `AIMessage`、新 `tool_call_id`）→ 产生新节点，不做去重 |
| 从同一断点连开两条不同方向的续连 | 允许（树形分叉）；两条分别写各自的子节点，父节点不变 |
| 同一断点被续连到 `max_children`(2) 次仍在失败 | `branched_out` error，提示换子代理或重新开始 |
| 连续失败 3 次 | `depth` 逐次递增；达上限后不再给 `continue_id` |
| 子代理首次第 1 步即失败 | `final_state` 为空 → 节点 delta 回退为 `[HumanMessage(description)]`；仍可续连 |
| 400 消息结构非法 / 401 / 权限错 | `retryable=False`，不给 `continue_id`，文案禁止重试 |
| `GraphRecursionError` | `retryable=False`，不续连 |
| 主 Agent 传错 `subagent_type` | `type_mismatch` error（**不做**入场清理，节点保持原样） |
| 主 Agent 传已耗尽的 `continue_id` | `exhausted` error |
| 主 Agent 传 `available=False` 的 `continue_id` | `not_available` error，文案说明原因码 |
| 链中间节点被外部删除 | `chain_broken` error，明确提示从最近可用节点续连 |
| 同一次回答吐两个相同 `continue_id` | 第二个返回 `busy` error（锁内 check-and-add） |
| 父图回退到另一分支后凭旧 `continue_id` 续连 | **允许**（不再做分支校验，§4.5） |
| 子代理内部触发 `interrupt()` 审核 | `GraphInterrupt` 原样透传，主图产出 `__interrupt__`；`Command(resume=...)` 后**原地恢复，已完成超步不重跑**（实测，见 §4.7） |
| 续连调用本身被 `/retry` 重跑 | 子图从自身存档续跑；注入的历史保留原始 message id，按 id 去重，不会翻倍 |
| 用户「停止生成」 | `CancelledError` 透传不吞；可选写 `cancelled` 节点 |
| 用户「停止生成」时工具把取消包成普通异常 | 取消守卫 `is_cancellation_related` 识别 → 上抛，**不写节点、不下发 `continue_id`**，主图停止后续生成 |
| 取消发生在 `finally` 落盘时 | 任务已处于 cancelling，`await` 会被再次取消 → 默认放弃落盘（§6.3） |
| 子代理是 `CompiledSubAgent`（`RunnableLambda`） | 正常参与错误包装；快照机制与图类型无关 |
| 子代理编译时 `checkpointer=False` | 无影响（本方案不依赖子图 checkpointer） |
| `catch_errors=False` | 异常原样抛出，不写节点 |

---

## 9. 边界与已知限制

| 限制 | 说明 | 缓解 |
|---|---|---|
| 历史每次唤起都重发给模型 | 与在图层面做断点续跑的成本相同（后者同样要重发同一段历史） | 子代理挂 `MamboSummarizationMiddleware` |
| 非消息通道被重置 | 见 §2.2 N2 | 文档写明；必要时扩展 `extra_state` 字段 |
| 子代理的 checkpoint 写在父线程里 | 子图继承父图 checkpointer，ns 为 `<节点名>:<task_id>`（§4.7）。长会话 + 多子代理时存储增长 | 既有行为（非本方案引入）；续连会增加超步数，容量评估时计入 |
| 重跑未提交超步时，子代理会**再跑一轮** | 该 `task` 调用的结果从未提交，重跑超步必然重新进入；但子图会从自身存档继续，**只重跑失败那一步**（实测 `n1=1,n2=1,n3=2`）。这是 LangGraph 的超步语义，不是本方案引入 | 工具异常内化已把“工具失败导致超步中止”堵住；剩余触发源只有未内化的异常 |
| ~~注入消息须带确定性 id~~ **已排除** | 真实路径（挂起 resume / 失败后 `/retry`）下新输入都会被丢弃或按 id 去重，不会出现两条相同指令（§4.7.3） | 无需处理；仅在“手工复用同一 ns、非 resume 方式重放”时才需要加 `id` |
| ~~`checkpoint_ns` 必须 override~~ **已排除** | 同上；非 `messages` 通道沿用旧存档反而是想要的行为 | 需要隔离子图内部 `todos` 等通道时再考虑 |
| ~~`interrupt()` 无法原地恢复~~ **已排除** | 子图继承父图 checkpointer，`Command(resume=...)` 是原地恢复，已完成超步不重跑（§4.7.1 实测） | 无需处理 |
| 崩溃那个超步内的已完成任务会重跑 | `/retry` 恢复重跑的是“未提交的超步”，该超步里已成功的任务会重复执行；`GraphInterrupt` 恢复同理。续连机制只能保证“已提交的历史”不重做 | 工具异常内化已把“工具失败导致超步中止”这条路堵住；剩余风险来自 LLM 错误与审核中断 |
| 工具失败不再打断 → 模型可能反复重试工具 | 工具层内化后，模型可能反复调同一个坏工具 | **不做额外限制**：`recursion_limit` 天然兜住跑飞（子代理继承主图 config）。**明确不使用 `ToolCallLimitMiddleware`** —— 按工具调用次数限流会误伤正常的多步任务，且 `task` 的重试已由 §6 的三维配额管住，语义上也不该用"调用次数"来表达 |
| 链的中间节点不可单独删除 | 删任一段都会导致后代链无法拼接（`chain_broken`） | 生命周期跟随 `thread_id`，整链随会话清理 |
| `store` 需持久化 | 默认 `InMemoryStore` 重启即失（与 checkpointer 同性质） | 生产使用 `SqliteStore` / `PostgresStore` |
| `delta_messages` 是序列化 dict | 跨 LangChain 版本格式可能变；`ToolMessage.artifact` 等字段易丢 | 一律走 `messages_to_dict` / `messages_from_dict`，不手写 JSON |
| 依赖工具异常内化中间件（§3.1） | **已落地**（v0.4.2b5）。若该中间件被关闭/未挂载，超步会中断，delta 可能缺兄弟工具结果 | 子代理侧由 `_get_subagents()` 强制挂载到链首，用户无法通过 `SubAgent.middleware` 移除 |
| `retryable` 判据两处漂移 | 工具层内化与子代理续连都要判可重试性 | 共用 `tool_call_error._is_retryable`，落地本方案时提升为公开函数（§5.2） |
| 模型可能重复只读操作 | 如已读过的文件因摘要被压缩掉而重读 | 可接受 |
| 不带 `tool_call_id` 调用 `task` | 现有代码直接 `raise ValueError`；内化后会被包装成 error `ToolMessage`，但**无法生成 `continue_id`** | 改为“返回不可续连的 error 消息”（`available=False`），语义更直白 |

---

## 10. 模块清单与实现要点

### 10.1 新增 `mambo_agents/src/mambo_agents/middleware/subagent_snapshot.py`

- `SubAgentSnapshotNode` 模型（§4.2）
- `SUBAGENT_HISTORY_KEY = "mambo_subagent_history"`
- `put_node` / `get_node` / `get_chain` / `list_roots`，均提供 sync 与 async 版本
- `_snapshot_namespace(thread_id) -> tuple[str, str]`
- `_build_node(...)`：从执行上下文与结果组装节点（含 delta 计算：按 message id 差集）
- `_classify_error_retryable(exc) -> bool`（§5.2）—— **不要另写一套**：
  直接复用 `mambo_agents.middleware.tool_call_error._is_retryable`
  （落地本方案时把它提升为公开函数，两个调用方共用）
- `_format_subagent_error(...)` / `_build_continuation_prompt(...)`
- `validate_continuation(...)`：§4.5 状态机
- 复用 `mambo_agents/utils/cancellation.py` 的 `is_cancellation_related`（§5.3）
- §7.3 的公开查询 API

### 10.2 `middleware/subagents.py`

1. `SubAgentConfig` 新增：
   ```python
   catch_errors: bool | None = None
   max_continue_attempts: int | None = None
   ```
2. `TaskToolSchema`：
   - `description: str = Field(default="", ...)`（新建必填、续连可选）
   - `continue_id: str | None = Field(default=None, ...)`
   - ⚠️ 两个字段都**必须写进 schema**：`infer_schema=False` + `extra="allow"` 下，
     未声明的参数会在 `_parse_input()` 阶段被丢掉（该类的 docstring 已踩过这个坑）；
   - 代码里必须校验"无 `continue_id` 且 `description` 为空" → `missing_description`。
3. 提示词（`TASK_TOOL_DESCRIPTION` / `TASK_SYSTEM_PROMPT`）：
   - 修正 “Each agent invocation is stateless...” 的表述；
   - 新增 `Error recovery & continuing` 章节：临时性错误可携 `continue_id` 续跑；
     `available=false` 或已达上限时禁止再次续连，直接上报用户。
4. `_build_task_tool(...)`：
   - 新增参数 `catch_errors: bool = True`、`max_continue_attempts: int = 3`、
     `max_children: int = 2`、`max_root_failures_per_thread: int = 10`；
   - `task` / `atask` 新增 `continue_id` 参数；
   - `_load_continuation(...)`：读链 + 校验 + 拼历史 + 组续接指令；
   - `_reset_own_node(...)`：入场清理，`store.delete(ns, runtime.tool_call_id)`
     （§4.3.1，位置在校验之后、执行之前）；
   - `_on_failure(...)`：组节点 → store 写 → 回读校验 → 返回 error `Command`
     （复用 `_return_command_with_state_update` 的 passthrough，§4.3.2）；
   - 执行主体包 `try / except GraphBubbleUp: raise / except Exception as exc:`，
     其中 `except Exception` 首行插入取消守卫 `if is_cancellation_related(exc): raise`
     （§5.3）；
   - 并发互斥：`finally` 里 discard，且"锁内 check-and-add"（§4.6）；
   - **没有** `_seal_node`（封链机制已删除，§4.2 末段）。
5. `SubAgentMiddleware.__init__`：
   - 透传 `catch_errors` / `max_continue_attempts`；
   - `_get_subagents()` 的中间件挂载**已在 v0.4.2b5 落地**为
     `[ToolCallErrorMiddleware(), *spec.middleware]`（可选再追加
     `HumanInTheLoopMiddleware`），本方案**无需改动**；
   - 若确认子代理也需要悬空修补 / 消息重排（§3.2 的“待补”项），再追加：
     ```python
     middleware = [
         ToolCallErrorMiddleware(),
         *spec.middleware,
         PatchToolCallsMiddleware(),
         ReorderToolMessagesMiddleware(),
     ]
     ```
     注意 `PatchToolCallsMiddleware` 依赖 `runtime.store`（§7.4），且每轮
     `before_agent` 会全量扫描 `messages`，子代理历史较长时需评估开销。

### 10.3 `graph.py`

- 解析 `SubAgentConfig` 时透传 `catch_errors`、`max_continue_attempts`。

### 10.4 `middleware/patch_tool_calls.py`

- 修补悬空 `task` 调用时，若 store 中存在该 `tool_call_id` 的节点，
  注入 `subagent_continuation` 元数据（§7.4）。

### 10.5 导出与文档

- `mambo_agents/__init__.py` 导出 `SubAgentSnapshotNode` 与 §7.3 的查询 API；
- `docs/usage.md` / `docs/usage.cn.md`：补 `catch_errors`、`continue_id`、
  链式快照与查询 API、控制流透传说明、按 `thread_id` 清理的方式。

---

## 11. 测试计划

**前置依赖**

1. `test_subagent_middleware_order`：断言子代理 middleware 链首为
   `ToolCallErrorMiddleware`，其后为用户 `spec.middleware`
   （现状**不含** Patch / Reorder，见 §3.2；若日后补挂再扩展断言）。
   （工具异常内化本身的行为测试见 `tests/test_tool_call_error_middleware.py`。）
- （附加）`test_subagent_tool_error_does_not_produce_snapshot`：子代理内部工具抛
  `RuntimeError` → 子代理继续、主图不崩、store 中**没有**续连节点。

**错误包装与节点写入**

2. `test_subagent_error_wrapped_as_tool_message`（sync & async）：主图不崩，
   `status == "error"`、`name == "task"`、`additional_kwargs` 字段齐全。
3. `test_node_written_before_error_message`：store 中节点存在，`depth=1`、`parent_id=None`、
   `delta_messages` 非空、`steps` 正确。
4. `test_first_step_failure_falls_back_to_human_message`：首个超步即失败 →
   `delta_messages == [HumanMessage(description)]`。

**链式续连**

5. `test_continue_replays_history_without_reexecuting`（核心）：第 1 步工具计数器；
   第 2 步超时；凭 `continue_id` 续连成功 → 断言计数器仍为 1。
6. `test_chain_accumulates_deltas`：连续 2 次失败 → 第 2 个节点 `chain == [root_id]`、
   `parent_id == root_id`；续连时注入历史 = 两个 delta 顺序拼接。
7. `test_delta_uses_message_id_diff`：挂一个会 `Overwrite` 重写 `messages` 的中间件
   （模拟 `MamboSummarizationMiddleware`）→ delta 仍只含本次新增消息，不丢进度、
   不重复包含注入历史。**（对应 §4.2 的下标切片坑，必须有用例）**
8. `test_continue_supports_new_instruction`：续连带新 `description` →
   新指令出现在注入历史末尾。
9. `test_continue_uses_fresh_namespace`：续连前后子图 `checkpoint_ns` 不同。
10. `test_entry_delete_makes_retry_idempotent`：同一 `tool_call_id` 连续执行两次
    （模拟 `/retry` 恢复）→ 成功后 store 无残留；再失败则只有一条记录且内容是新的。
11. `test_entry_delete_keeps_ancestors`：续连失败后再从同一父节点续连 →
    父节点记录未被删除、delta 完整。
12. `test_branching_from_same_node_allowed`：从同一 `continue_id` 开两条不同指令的
    续连 → 两个子节点并存（`max_children` 内），父节点不变。
13. `test_branched_out_limit`：同一断点续连超过 `max_children` → `branched_out` error。

**状态机与配额**

14. `test_continue_not_found` / `test_continue_type_mismatch` /
    `test_continue_chain_broken` / `test_continue_not_available` /
    `test_continue_missing_description` / `test_continue_busy`。
15. `test_continue_exhausted`：`depth > max_continue_attempts` → `available=False`。
16. `test_non_retryable_errors_withhold_continue_id`：400 / 401 / `GraphRecursionError`。
17. `test_zero_progress_early_stop`：续连 `steps == 0` → 提前熔断。
18. `test_validation_failure_keeps_node`：校验失败（`not_found` / `busy`）时
    **不做入场清理**，原节点记录完好。
19. `test_write_failure_withholds_continue_id`：模拟 `store.put` 抛错 →
    `available=False`、不下发 `continue_id`。

**控制流**

20. `test_graph_interrupt_propagates_and_resumes`：子代理 `interrupt()` → 主图产出
    `__interrupt__`；`Command(resume=...)` 后**原地恢复**（实测，见 §4.7），
    已完成超步不重跑；用例按此断言。
21. `test_cancelled_error_not_swallowed`：`CancelledError` 原样抛出。
22. `test_wrapped_cancellation_not_persisted`：工具把 `CancelledError` 包成
    `RuntimeError` → 取消守卫识别（§5.3）→ 上抛，**不写节点、不下发 `continue_id`**。
23. `test_catch_errors_false_reraises`。

**下游对接**

24. `test_continuation_event_emitted`：失败时发出 `subagent_continuation` 事件。
25. `test_patch_task_call_injects_continue_id`：悬空 `task` 修补时附带 `continue_id`
    （sync 与 async 两条路径各一例，§7.4 的 store 读取差异）。
26. `test_query_apis`：`get_node` / `get_chain` / `list_roots`，含客户端过滤。
27. `test_continue_across_store_reload`：持久化 store 模拟重启后可续连。

运行方式：

```cmd
set PYTHONPATH=C:\Users\87485\PycharmProjects\mambo_agents\src
.venv\Scripts\python.exe -m pytest tests/test_subagents.py -q
```

---

## 12. 实施顺序

0. ~~先落地《工具异常内化中间件设计方案》，完成主图与子代理两处装配（§3.1），
   其中包含取消守卫 `utils/cancellation.py`（§5.3）。~~
   **已完成（v0.4.2b5）**：`middleware/tool_call_error.py`、`utils/cancellation.py`、
   主图与子代理两处装配、`tests/test_tool_call_error_middleware.py`（22 项）。
1. 子代理 middleware 链顺序核对（§3.2 现状：链首 `ToolCallErrorMiddleware`）→ 测试 1。
2. `subagent_snapshot.py`：模型 / 命名空间 / 读写 / 入场清理 / delta（id 差集）/
   错误分级 / 格式化 → 测试 3、4、7、16。
   `_classify_error_retryable` 复用 `tool_call_error._is_retryable`（§5.2），不新写。
3. `subagents.py`：schema（含 `continue_id`、`description` 可选 + 校验）/
   提示词 / `_load_continuation` / `_on_failure` / 状态机 / 入场清理 /
   并发互斥（锁内 check-and-add）/ 双层配额（depth + children）/ 取消守卫接入
   → 测试 2、5–6、8–15、17–19、21–23。
4. `graph.py` 透传配置 → 集成用例。
5. `patch_tool_calls.py` 联动（含 `abefore_agent`）+ 自定义事件 → 测试 24、25。
6. 导出查询 API + 文档 → 测试 26、27。
7. （可选）判断是否需要给子代理补挂 Patch / Reorder（§3.2 待补项），
   若补挂再补一条链顺序断言。
8. 全量回归（`tests/` 全跑，`PYTHONPATH` 必须指向本仓库 `src`）。

---

## 13. 已决策（无遗留待定项）

| # | 问题 | 结论 |
|---|---|---|
| ~~D1~~ | ~~子代理内部 `interrupt()` 无法原地 resume~~ | **已实测排除**：子图继承父图 checkpointer（`checkpoint_ns = <节点名>:<task_id>`），`interrupt()` + `Command(resume=...)` 是**原地恢复**，已完成超步不重跑。见 §4.7。无需任何改动 |
| ~~D2~~ | ~~工具层内化后模型可能对坏工具反复重试，是否需要限流~~ | **不引入 `ToolCallLimitMiddleware`**。理由：① `recursion_limit` 已经天然兜住跑飞；② 按“工具调用次数”限流会误伤正常的多步任务（子代理本来就该调几十次工具）；③ `task` 的重试已由 §6 的三维配额表达，用“调用次数”表达它语义上是错的 |
