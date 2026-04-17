import json
import os
import subprocess
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic.types import Message, ToolParam
from dotenv import load_dotenv

# 获取.env文件中的环境变量，覆盖已有变量
load_dotenv(override=True)


# 优先使用其他模型
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = "你是当前目录【{os.getcwd()}】的智能体助手可以在当前工作空间使用相关bash命令"

TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "在当前工作空间一个hello.py文件",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]


@dataclass
class LoopState:
    """循环状态，包含当前的消息、轮数和转换原因"""

    messages: list
    turn_count: int = 1
    transition_reason: str | None = None


def run_bash(command: str) -> str:
    """
    执行一个shell命令并返回输出。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    # any() 任意一个真则为True
    if any(item in command for item in dangerous):
        return "Error: 不允许执行危险命令"

    try:
        # 使用subprocess.run执行命令，返回结果
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: 命令执行超时（120秒）"
    except Exception as e:
        return f"Error: {str(e)}"

    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(未输出任何内容)"


def extract_text(content: list | object) -> str:
    """
    从内容中提取文本，支持 list 或单个对象。
    """
    print("从内容中提取文本，支持 list 或单个对象")
    if not isinstance(content, list):
        return ""

    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def execute_tool_calls(response_content) -> list[dict]:
    """
    执行工具调用，返回工具结果的列表。

    从block.type != "tool_use" 可以看出，response_content.stop_reason == "tool_use"时，
    其response_content.content 是一个 list，包含了 type == "tool_use" 类型的 block。
    content:
        TextBlock(citations, text, type)
        ToolUseBlock(
            id,
            caller,
            input(command),
            name,
            type
            )
    """
    results = []

    for block in response_content:
        if block.type != "tool_use":
            continue

        command = block.input["command"]
        print(f"\033[33m$ {command}\033[0m")
        output = run_bash(command)
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "output": output,
            }
        )
    return results


def run_one_turn(state: LoopState) -> bool:
    """
    执行一轮对话，保存助手结果，返回是否需要继续。
    调用工具返回True, 否则返回False。
    """
    response: Message = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
    )
    # 将助手响应添加到消息列表，保存上下文
    print("response:::", response)
    print("格式化response", json.dumps(response.model_dump(), indent=2))
    state.messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        state.transition_reason = None
        return False

    results = execute_tool_calls(response.content)
    if not results:
        state.transition_reason = None
        return False

    # 工具调用成功，将结果追加到消息历史中，保存上下文
    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    # 更新继续调用原因
    state.transition_reason = "tool_result"
    # 返回True, 继续下一轮对话
    return True


def agent_loop(state: LoopState) -> None:
    """
    执行完整的对话循环。
    执行一轮时对话时，判断响应结果，如果调用工具，那么拿到工具响应结果追加到消息历史中，
    继续执行下一轮对话，直到不再调用工具则结束循环。
    """

    while run_one_turn(state):
        pass


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "quit", "exit"):
            break
        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)

        # 会话结束后拿到最后一次响应结果（LLM响应），截取最大长度度字符
        final_text = extract_text(history[-1]["content"])

        if final_text:
            print(final_text)
        print()
