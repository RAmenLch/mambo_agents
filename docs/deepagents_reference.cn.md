> 📖 [English](deepagents_reference.md) | **中文**

# 关于 deepagents 的参考声明

## 1. 参考申明

**Mambo Agents** 项目承认并感谢 [deepagents](https://github.com/langchain-ai/deepagents)（v0.5.7，LangChain 团队）的开源工作。本项目在其基础架构之上进行了大量重构和扩展，核心灵感来源于 deepagents 的 **中间件管道（Middleware Pipeline）** 和 **后端抽象（Backend Protocol）** 两大设计范式。

具体而言，我们参考了 deepagents 的以下架构思想：

| 架构层面 | 参考来源 | 说明 |
|----------|----------|------|
| 中间件管道模式 | `deepagents.graph.create_deep_agent()` | 通过 `AgentMiddleware` 栈串行拦截 Agent 生命周期 |
| 后端抽象协议 | `deepagents.backends.protocol.BackendProtocol` | `ls/read/write/edit/grep/glob` 六核心文件操作 |
| 状态后端 | `deepagents.backends.state.StateBackend` | 内存文件存储，通过 LangGraph 状态通道管理 |
| 文件系统后端 | `deepagents.backends.filesystem.FilesystemBackend` | 真实文件系统封装 |
| 技能系统 | `deepagents.middleware.skills.SkillsMiddleware` | 渐进式技能披露（SKILL.md） |
| 子代理系统 | `deepagents.middleware.subagents.SubAgentMiddleware` | 同步/异步子代理委托 |
| 对话摘要 | `deepagents.middleware.summarization.SummarizationMiddleware` | 上下文窗口压缩 |
| 工具调用修复 | `deepagents.middleware.patch_tool_calls.PatchToolCallsMiddleware` | 悬空工具调用修补 |

> **重要提示**：本项目并非 deepagents 的 fork 或分支，而是在其架构思想指导下独立实现的 Agent 框架。代码层面仅在早期原型阶段参考了部分实现，当前版本的所有代码均为独立编写。

---

## 2. 优化方向与取舍

以下差异不代表 deepagents 的设计是"错误"的，而是两个项目在**不同场景和目标**下的取舍。

### 2.1 多后端与命令执行

| 对比维度 | deepagents | Mambo Agents | 取舍说明 |
|----------|------------|--------------|----------|
| **本地文件+执行** | `LocalShellBackend`（`FilesystemBackend` + `SandboxBackendProtocol`）<br>• 通过继承 `SandboxBackendProtocol` 获得 `execute()`<br>• 使用 `subprocess.run(shell=True)` 直连宿主机，无隔离 | `LocalBackend`（自带 `execute()`）<br>• `execute()` 是后端自身方法，无需额外协议层<br>• 同样直连宿主机，但内聚在单一类中 | Mambo 不做 LocalShellBackend/FilesystemBackend 两套类分离，本地文件操作和本地 shell 执行天然一体 |
| **远程文件+执行** | `LangSmithSandbox`（继承 `BaseSandbox`）<br>• 对接 LangSmith 云端容器服务，实现真正的进程隔离<br>• 文件操作全部委托给 `execute()`，通过 SDK 收发命令 | `SshBackend`（基于 paramiko）<br>• `execute()` 通过 SSH 信道在远端执行<br>• 文件操作直接操作远端文件系统（ls/read/write 等不走 execute，有原生实现） | deepagents 的远程方案绑定特定云服务（LangSmith），适合容器化配置好的环境；Mambo 的 SSH 方案更通用，任何有 sshd 的机器都可用 |
| **execute 架构理念** | `execute()` 需要通过 `SandboxBackendProtocol` 获得<br>• 协议层次：`BackendProtocol`（纯文件）→ `SandboxBackendProtocol`（+ execute）→ `BaseSandbox`（便利封装）<br>• 只有声明为 "Sandbox" 的后端才能执行命令 | `execute()` 是后端自带的可选能力<br>• 协议层次：只有 `BackendProtocol`，任何后端想加 execute 直接在自家类上实现即可<br>• `LocalBackend`、`SshBackend` 各怀 execute，`StateBackend` 没有 — 按需启用，不强制分类 | Mambo 追求更通用、更灵活的 execute 启用方式：不引入独立的 Sandbox 协议层，execute 就是一个普通的方法，由各后端自行决定是否提供 |
| **跨会话持久化** | ✅ `StoreBackend`（基于 LangGraph BaseStore） | ❌ 未实现 StoreBackend | 当前阶段聚焦单会话场景，未来按需添加 |

### 2.2 中间件栈差异

| 对比维度 | deepagents | Mambo Agents | 取舍说明 |
|----------|------------|--------------|----------|
| **安全审查** | `HumanInTheLoopMiddleware`（简单人工审批） | `AutoSecurityReviewMiddleware`（AI 预审 + 人工兜底） | Mambo 增加 AI 自动审查层，减少人工审批打断频率 |
| **任务规划** | `TodoListMiddleware`（简单 TODO 管理） | `MamboPlanMiddleware`（结构化计划 + 摘要集成钩子） | Mambo 的计划与摘要系统深度耦合，防止压缩丢失计划状态 |
| **多模型兼容** | ❌ 无工具消息重排序 | ✅ `ReorderToolMessagesMiddleware` | 某些多模态模型对工具消息顺序敏感，Mambo 显式处理 |
| **工具扩展性** | `FilesystemMiddleware`（六核心工具 + 后端额外工具 + 多模态 read） | ✅ `BackendToolsMiddleware`（六核心工具 + 后端额外工具自动注入）<br>• `include_line_numbers` 参数：Mambo 的 `read` 默认不返回行号，模型可自行决定在需要定位行号时传 `True`（deepagents 总是无条件加行号，不可关闭）<br>• `build_tool_descriptions()` 提取所有工具描述映射，供 `AutoSecurityReviewMiddleware` 安全审查时理解工具用途 | 基础工具层面的差异化：`include_line_numbers` 的可控性减少了无关噪音；`build_tool_descriptions()` 使工具描述可被其他中间件消费 |
| **大结果驱逐** | ✅ `FilesystemMiddleware` 内置驱逐（保存到 `/large_tool_results/`）<br>• 驱逐时机：`wrap_model_call`（预模型调用时批量处理消息历史） | ✅ `BackendToolsMiddleware` 驱逐（保存到 `/.mambo/large_tool_results/`）<br>• 驱逐时机：`wrap_tool_call`（工具返回结果后即时拦截，而非等到下一轮模型调用） | 双端均实现大结果驱逐（包括多模态块保留和 Command 多消息场景），核心差异仅在于驱逐时机：Mambo 在工具返回后立即生效，deepagents 在模型调用前批量处理 |
| **大文件读取限制与摘要** | ✅ `read_file` 内置 `_truncate()`：超过字符阈值后附加静态 `READ_FILE_TRUNCATION_MSG`（固定模板消息，无文件类型区分能力） | ✅ `BackendProtocol` 内置 `max_read_chars` + 可插拔 `ReadSummarizer` 回调<br>• 回调签名 `(file_path, content, max_chars) -> str`，可基于文件后缀（`.py`/`.json`/`.yaml` 等）生成差异化指导摘要<br>• 默认摘要器引导模型用 `offset`+`limit` 分段读取；用户注入自定义回调即可按文件类型定制策略<br>• 二进制/多模态文件永不截断<br>• 附赠 `read_summarizers` 子包，含 9 种预置摘要器（Python / JS / TS / Java / C / C++ / Go / Rust / Markdown / JSON），提取结构大纲 + 精确行号 | 两方均有字符级读取上限和截断提示；Mambo 的核心差异在于**可插拔回调**体系 — 将大文件内容替换为按文件类型定制的"指导性摘要"，帮助模型做出更正确的下一步决策，而非仅给出固定截断警告 |
| **记忆系统** | ✅ `MemoryMiddleware`（AGENTS.md） | ✅ `MamboMemoryMiddleware`（AGENTS.md） | Mambo 参考 deepagents 的 `MemoryMiddleware` 设计（`before_agent` 通过 `backend.download_files` 加载 AGENTS.md 到 state、`wrap_model_call` 注入 `<agent_memory>` 系统提示、多 sources 合并加载），在此基础上增加可选自定义 `format_prompt` 格式化回调 |
| **Profile 系统** | ✅ `ProviderProfile` + `HarnessProfile`（模型/提供商调优） | ❌ 未实现 | Mambo 暂不聚焦模型级细粒度调优，由用户自行配置 |
| **Anthropic 缓存** | ✅ `AnthropicPromptCachingMiddleware` | ❌ 未实现 | Mambo 暂不绑定特定提供商优化 |
| **子代理对话透出** | 子代理结果通过 `ToolMessage` 返回 | ✅ 子代理通过 `Command` 返回<br>• `_return_command_with_state_update()` 将子代理的非排除状态键（如文件系统状态）透明传递回父代理<br>• 三级流式事件粒度（`messages`/`updates`/`values`），通过 `subagent_event` 自定义事件推送子代理内部执行过程<br>• 并行子代理通过 `tool_call_id` 区分各自的事件流 | 子代理不仅返回结果，还将其对话上下文中的状态信息抛回父代理，使父代理可以了解子代理执行期间创建/修改了哪些文件等上下文 |

### 2.3 架构设计取舍

| 对比维度 | deepagents | Mambo Agents | 取舍说明 |
|----------|------------|--------------|----------|
| **类型安全** | 部分使用 TypedDict | ✅ 全链路 Pydantic 类型（无 Dict/Any 鸭子类型） | 严格类型控制是 Mambo 的核心编码规范 |
| **路由后端** | `CompositeBackend`（default + routes 字典，任意前缀 → 任意后端）<br>• 路由完全自由：`/memories/` → StoreBackend、`/cache/` → StateBackend 等任意组合<br>• `ls("/")` 透明聚合所有路由后端<br>• `execute()` 始终委托 default 后端，通过 `SandboxBackendProtocol` 类型判定 | `HybridWorkspaceBackend`（1 真实 + N 虚拟，统一 `/.mambo/` 前缀）<br><br>**硬性约束：**<br>• 虚拟 workspace 前缀固定 `/.mambo/`，不可自定义为其他路由路径<br>• 虚拟 workspace 仅开放 6 核心文件工具 <br>• System prompt 显式告知 AI 以上约束<br><br>**放宽之处：**<br>• 内置 `copy` 工具，支持跨后端（虚拟 ↔ 真实）的单文件搬运<br>• 支持真实后端灵活的提供工具,而不仅限于execute | `CompositeBackend` 的任意前缀路由提供了最大灵活度，但 AI 可能会做出 `execute`（如 `cat /memories/...`）绕过路由层的决定,导致比较严重的幻觉；`HybridWorkspaceBackend` 用固定前缀 + 显式工具白名单的提示词限制AI操作，并且放宽对真实后端的tools限制 |

---

## 3. 功能差异对照表

以下逐一列出两个项目在同等能力层面上的对应关系：

### 3.1 后端（Backend）

| 功能 | deepagents | Mambo Agents |
|------|------------|--------------|
| 协议定义 | `BackendProtocol` + `SandboxBackendProtocol`（independent execute layer） | `BackendProtocol`（execute is also a backend method，no independent protocol layer） |
| 内存存储 | `StateBackend` | `StateBackend`（reconstructed） |
| 本地 execute | `LocalShellBackend`（`FilesystemBackend` + `SandboxBackendProtocol`，two layers of parent classes） | `LocalBackend`（single class with built-in `execute()`，`tree`，`delete`） |
| 远程 execute | `LangSmithSandbox`（cloud container service，all file operation are delegated to `execute()`） | `SshBackend`（native SSH，`execute()` via SSH channel，file operation has native implementations） |
| 路径路由 | `CompositeBackend`（灵活多后端路由，任意前缀 → 任意后端；`execute()` 可能绕过路由层） | `HybridWorkspaceBackend`（1 真实 + N 虚拟路由：`/.mambo/<name>/` → 内存，其余 → 真实后端；System prompt 显式告知 AI workspace 语义） |
| 跨会话存储 | `StoreBackend` | ❌ |

### 3.2 中间件（Middleware）

| 功能 | deepagents | Mambo Agents |
|------|------------|--------------|
| 文件工具注入 | `FilesystemMiddleware`（六核心 + 额外工具 + 大结果驱逐 + 多模态 read） | `BackendToolsMiddleware`（六核心 + 额外工具 + 大结果驱逐 + 多模态 read；`include_line_numbers` 可选、`build_tool_descriptions()` 外部消费） |
| 大文件读取限制与摘要 | `read_file` 静态截断消息（固定模板，无法按文件类型区分） | ✅ `max_read_chars` + `ReadSummarizer`（可插拔回调，按文件后缀生成差异化指导摘要） |
| 技能披露 | `SkillsMiddleware` | `SkillsMiddleware`（重构） |
| 同步子代理 | `SubAgentMiddleware` | `SubAgentMiddleware`（重构；`subagent_event` 流式事件、三级粒度、状态透传） |
| 异步子代理 | `AsyncSubAgentMiddleware` | `AsyncSubAgentMiddleware`（重构） |
| 对话摘要 | `SummarizationMiddleware` | `MamboSummarizationMiddleware`（扩展）<br>• **链式摘要**：在多轮摘要中保留前次摘要内容，使用 `CHAINED_SUMMARY_PROMPT` 要求模型合并历史摘要而非覆盖，防止跨轮次信息丢失<br>• **CJK token 计数**：自动检测中文/日文/韩文字符占比，使用与英文不同的 chars-per-token 比例估算 token 数，避免 CJK 文本的 token 严重低估<br>• **保护最近用户消息**：确保最新的 user message 不会被摘要掉（langchain 基类无此保护）<br>• **可选后端持久化**：被移除的原始消息可写入 `BackendProtocol` 的 `/conversation_history/{thread_id}.md` 文件 |
| 悬空工具修复 | `PatchToolCallsMiddleware` | `PatchToolCallsMiddleware`（保持） |
| 工具消息重排序 | ❌ | ✅ `ReorderToolMessagesMiddleware` |
| 安全审查 | `HumanInTheLoopMiddleware` | ✅ `AutoSecurityReviewMiddleware`（AI 预审 + 人工审批） |
| 任务规划 | `TodoListMiddleware` | ✅ `MamboPlanMiddleware`（结构化 + 摘要集成） |
| 记忆加载 | `MemoryMiddleware` | ✅ `MamboMemoryMiddleware` |
| 工具排除 | `_ToolExclusionMiddleware` | ❌ |
| Anthropic 缓存 | `AnthropicPromptCachingMiddleware` | ❌ |

### 3.3 核心入口

| 功能 | deepagents | Mambo Agents |
|------|------------|--------------|
| 主构造函数 | `create_deep_agent()` | `create_mambo_agent()` |
| Profile 系统 | ✅ ProviderProfile + HarnessProfile | ❌ |
| 子代理类型 | `SubAgent` / `CompiledSubAgent` / `AsyncSubAgent` | `SubAgent` / `CompiledSubAgent` / `AsyncSubAgent` |

---

## 结语

deepagents 是 Agent 框架领域的优秀开源项目，其**中间件管道 + 后端协议**的架构设计为 Mambo Agents 提供了清晰的蓝图。Mambo Agents 在此基础上进行了以下方向的重构与扩展：强化了安全审查、增加了远程 SSH 操作能力、引入了严格的类型系统，并针对大结果处理、多模型兼容性、计划-摘要协同等问题进行了专项优化。

我们始终怀着感激之情，认可 deepagents 团队对 Agent 基础设施领域的贡献。本项目的所有独特扩展均为面向自身需求的有意选择，而非对 deepagents 设计的否定。
