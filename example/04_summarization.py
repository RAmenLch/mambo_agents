"""
============================================================
Mambo Agents - 示例 04：超长对话管理（Summarization）
============================================================

演示对话摘要功能 - 自动压缩长对话历史，防止超出 LLM 上下文窗口。

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/04_summarization.py
============================================================
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import SummarizationMode, create_mambo_agent
from mambo_agents.middleware.summarization import SummarizationEvent
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def demo_basic_summarization():
    """演示基本摘要配置和触发效果。"""
    print("=" * 60)
    print("Part 1：基本摘要配置")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 配置摘要：当累计超过 1000 tokens 时触发（为了快速演示，设得较低）
    agent = create_mambo_agent(
        model,
        summarization={
            "trigger": ("tokens", 1000),          # 累计超过 1000 tokens 时触发
            "keep": ("messages", 10),               # 保留最后 10 条消息不压缩
            "model": model,                         # 使用相同的模型做摘要
            "trim_tokens_to_summarize": 2000,       # 摘要时回顾 2000 tokens
        },
    )

    # 同一会话中发送多条消息，观察摘要效果
    config = {"configurable": {"thread_id": "summarization-demo"}}

    topics = [
        "介绍一下 Python 的类型提示（type hints）",
        "类型提示在大型项目中有哪些最佳实践？",
        "Python 3.12 有哪些新特性？",
        "async/await 和传统多线程有什么区别？",
        "请总结一下我们之前讨论过的所有话题",
    ]

    for i, topic in enumerate(topics):
        print(f"\n--- 第 {i + 1} 轮对话 ---")
        result = agent.invoke(
            {"messages": [HumanMessage(topic)]},
            config=config,
        )
        # 只打印最后一条消息的前 300 个字符
        last_msg = result["messages"][-1]
        print(f"[回复] {last_msg.content[:300]}...")
        print(f"[当前消息总数] {len(result['messages'])}")

    print("\n提示：如果对话很长，摘要中间件会自动压缩历史消息。")
    print("你可以设置更低的 trigger 值来快速看到摘要效果。")


def demo_summarization_event():
    """演示如何通过 stream_mode='values' 获取 SummarizationEvent。
    
    SummarizationEvent 包含以下字段：
    - cutoff_index (int): state['messages'] 中的绝对索引，此索引之前的消息已被摘要
    - summary_message (HumanMessage): LLM 生成的摘要消息，包含 SESSION INTENT / SUMMARY / ARTIFACTS 三个部分
    - file_path (str | None): 如果开启了 offload_to_backend，被驱逐消息的存储路径
    - last_summarized_message (AnyMessage | None): 被摘要区域中最后一条真实消息（不含摘要标记消息）
    """
    print("\n" + "=" * 60)
    print("Part 2：获取 SummarizationEvent 摘要事件")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 使用更低的 trigger 确保快速触发摘要
    agent = create_mambo_agent(
        model,
        summarization={
            "trigger": ("tokens", 500),           # 很低的值，确保快速触发
            "keep": ("messages", 4),               # 保留最后 4 条
            "model": model,
            "trim_tokens_to_summarize": 1000,
        },
    )

    config = {"configurable": {"thread_id": "summarization-event-demo"}}

    # 发送多条消息触发摘要
    messages = [
        "Python 的类型系统是什么？",
        "PEP 484 定义了哪些内容？",
        "什么是泛型？",
    ]

    for i, msg in enumerate(messages):
        print(f"\n--- 第 {i + 1} 轮 ---")
        result = agent.invoke(
            {"messages": [HumanMessage(msg)]},
            config=config,
        )
        last_msg = result["messages"][-1]
        print(f"[回复] {last_msg.content[:200]}...")

    # 现在通过 stream_mode="values" 获取完整的 state，检查摘要事件
    print("\n--- 检查摘要事件 ---")
    for event in agent.stream(
        {"messages": [HumanMessage("请总结我们讨论过的所有内容")]},
        config=config,
        stream_mode="values",
    ):
        # 每次 state 更新都会触发，我们只关心最后一次
        pass

    # 从最终的 state 中获取 _summarization_event
    sum_event: SummarizationEvent | None = event.get("_summarization_event")
    if sum_event:
        print(f"\n✓ 检测到摘要事件！")
        print(f"  cutoff_index: {sum_event['cutoff_index']}")
        print(f"  summary_message 长度: {len(sum_event['summary_message'].content)} 字符")
        print(f"  file_path: {sum_event.get('file_path')}")
        print(f"  last_summarized_message 类型: {type(sum_event.get('last_summarized_message')).__name__}")
        print(f"\n摘要内容预览（前 500 字符）:")
        print("-" * 40)
        print(sum_event['summary_message'].content[:500])
        print("-" * 40)
        print(f"\n摘要事件字段说明：")
        print(f"  • cutoff_index = {sum_event['cutoff_index']}")
        print(f"    含义: state['messages'] 中前 {sum_event['cutoff_index']} 条消息已被摘要替换")
        print(f"  • summary_message: 摘要 HumanMessage，包含以下三个部分：")
        print(f"    - SESSION INTENT: 会话的总体目标")
        print(f"    - SUMMARY: 关键决策、结论和策略")
        print(f"    - ARTIFACTS: 创建/修改的文件及变更描述")
        print(f"  • file_path: 若启用 offload_to_backend，被驱逐消息的备份路径")
        print(f"  • last_summarized_message: 被压缩区域中最后一条真实消息")
    else:
        print("\n✗ 未检测到摘要事件（可能未触发阈值）")
        print("  提示：降低 trigger 值或增加更多轮对话来触发摘要")


def demo_stream_values_summarization():
    """演示实时监听摘要事件——通过 stream_mode='values' 在每轮调用中检测
    _summarization_event 的变化。
    """
    print("\n" + "=" * 60)
    print("Part 3：实时监听摘要事件")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        summarization={
            "trigger": ("tokens", 500),
            "keep": ("messages", 3),
            "model": model,
            "trim_tokens_to_summarize": 1000,
        },
    )

    config = {"configurable": {"thread_id": "stream-values-demo"}}

    # 记录上一轮的事件，用于对比
    last_cutoff = -1

    topics = [
        "什么是 Python 装饰器？",
        "装饰器和闭包有什么关系？",
        "Python 中的元类是什么？",
        "请总结我们讨论过的 Python 高级特性",
    ]

    for i, topic in enumerate(topics):
        print(f"\n--- 第 {i + 1} 轮：{topic[:30]}... ---")

        last_state = None
        for state in agent.stream(
            {"messages": [HumanMessage(topic)]},
            config=config,
            stream_mode="values",
        ):
            last_state = state

        if last_state:
            sum_event = last_state.get("_summarization_event")
            if sum_event and sum_event.get("cutoff_index", -1) != last_cutoff:
                last_cutoff = sum_event["cutoff_index"]
                print(f"  ⚡ 摘要事件触发！")
                print(f"     cutoff_index={sum_event['cutoff_index']} — "
                      f"前 {sum_event['cutoff_index']} 条消息已被压缩")
                print(f"     摘要长度: {len(sum_event['summary_message'].content)} 字符")
            else:
                print(f"  （本轮未触发新的摘要）")


def demo_per_model_call_mode():
    """演示 SummarizationMode.PER_MODEL_CALL - 每次模型调用前都检查摘要。

    与默认的 PER_ASTREAM（每次执行前只摘要一次）不同，
    PER_MODEL_CALL 在单次执行过程中多次调用模型时也会检查阈值，
    适合单轮任务特别长、中间可能超窗的场景。
    """
    print("\n" + "=" * 60)
    print("Part 4：SummarizationMode.PER_MODEL_CALL")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        summarization={
            "mode": SummarizationMode.PER_MODEL_CALL,  # 每次模型调用前检查
            "trigger": ("tokens", 500),
            "keep": ("messages", 3),
            "model": model,
        },
    )

    config = {"configurable": {"thread_id": "per-model-call-demo"}}

    topics = [
        "什么是 Python 的 GIL？",
        "GIL 对多线程性能有什么影响？",
        "async/await 能绕开 GIL 吗？",
    ]

    for i, topic in enumerate(topics):
        print(f"\n--- 第 {i + 1} 轮：{topic[:30]}... ---")
        result = agent.invoke(
            {"messages": [HumanMessage(topic)]},
            config=config,
        )
        last_msg = result["messages"][-1]
        print(f"[回复] {last_msg.content[:200]}...")

    print("\n提示：与 Part 1-3（默认 PER_ASTREAM）对比——本模式在单次执行的")
    print("多次模型调用之间也会触发摘要，适合长任务防超窗。")


def main():
    demo_basic_summarization()
    demo_summarization_event()
    demo_stream_values_summarization()
    demo_per_model_call_mode()


if __name__ == "__main__":
    main()
