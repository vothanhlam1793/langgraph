"""Agent LangGraph toi gian, co tool va vong lap tu dong."""

from __future__ import annotations

import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv()


class AgentState(TypedDict):
    """Trang thai duoc luu rieng cho tung cuoc hoi thoai (thread)."""

    messages: Annotated[list[AnyMessage], add_messages]


@tool
def calculate(expression: str) -> str:
    """Tinh bieu thuc co ban, vi du '(24 * 3) / 2'."""
    allowed = set("0123456789+-*/(). %")
    if not expression or any(character not in allowed for character in expression):
        return "Chi chap nhan bieu thuc tinh toan co ban."

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except (ArithmeticError, SyntaxError):
        return "Khong the tinh bieu thuc nay."


tools = [calculate]


def route_after_model(state: AgentState) -> Literal["tools", "end"]:
    """Neu model yeu cau tool thi chuyen den tools, nguoc lai thi ket thuc."""
    if state["messages"][-1].tool_calls:
        return "tools"
    return "end"


def build_agent():
    model_options = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "temperature": 0,
    }
    if base_url := os.getenv("OPENAI_BASE_URL"):
        model_options["base_url"] = base_url
    model = ChatOpenAI(**model_options)
    model_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState) -> dict[str, list[AnyMessage]]:
        """Node agent: model tu tra loi hoac yeu cau goi mot hay nhieu tool."""
        system = SystemMessage(
            content=(
                "Ban la tro ly huu ich, tra loi ngan gon bang tieng Viet. "
                "Khi can tinh toan, hay dung tool calculate thay vi tu nham."
            )
        )
        return {"messages": [model_with_tools.invoke([system, *state["messages"]])]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_model, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Hay dat OPENAI_API_KEY trong .env truoc khi chay chuong trinh.")

    agent = build_agent()
    config = {"configurable": {"thread_id": "learning-session"}}
    print("LangGraph starter. Nhap 'exit' de thoat.")

    while prompt := input("\nYou: ").strip():
        if prompt.lower() in {"exit", "quit"}:
            break
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        print(f"Tro ly: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
