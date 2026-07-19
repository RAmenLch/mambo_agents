"""
============================================================
Mambo Agents - 示例 02：自定义工具（Custom Tools）
============================================================

演示如何给 Agent 添加自定义工具。

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/02_custom_tools.py
============================================================
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from mambo_agents import create_mambo_agent
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。

    Args:
        city: 城市名称，如 "北京"、"上海"
    """
    # 模拟天气查询
    weather_data = {
        "北京": "晴朗，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 65%",
        "深圳": "雷阵雨，30°C，湿度 80%",
    }
    return weather_data.get(city, f"{city}：晴转多云，22°C，湿度 55%")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    """
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


def main():
    print("=" * 60)
    print("示例：给 Agent 添加自定义工具（天气查询 + 计算器）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 将自定义工具挂载到 Agent
    agent = create_mambo_agent(
        model,
        tools=[get_weather, calculate],
    )

    # 测试天气工具
    result = agent.invoke({
        "messages": [HumanMessage("查一下北京和深圳的天气")]
    }, config={"configurable": {"thread_id": "session-1"}})
    print("[天气查询]", result["messages"][-1].content[:300], "...")
    print()

    # 测试计算工具
    result = agent.invoke({
        "messages": [HumanMessage("帮我算一下 128 * 256 + 1024")]
    }, config={"configurable": {"thread_id": "session-1"}})
    print("[计算]", result["messages"][-1].content[:300], "...")
    print()

    # 综合使用
    result = agent.invoke({
        "messages": [HumanMessage(
            "查一下上海的天气，如果温度超过 25°C，帮我算一下 100 * 100"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})
    print("[综合]", result["messages"][-1].content[:300], "...")
    print()


if __name__ == "__main__":
    main()
