import json
import os
import subprocess
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv

# 获取.env文件中的环境变量，覆盖已有变量
load_dotenv(override=True)


# 优先使用其他模型
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = "你是当前目录【{os.getcwd()}】的智能体助手可以在当前工作空间使用相关bash命令"

TOOLS: list = [
    {
        "name": "bash",
        "description": "在当前工作空间运行一个shell命令",
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
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: 命令执行超时（120秒）"
    except Exception as e:
        return f"Error: {str(e)}"

    output = (result.stdout, result.stderr).strip()
    return output[:50000] if output else "(未输出任何内容)"


def extract_text(content: list | object) -> str:
    """
    从内容中提取文本，支持 list 或单个对象。
    """
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
    """
    print("当前循环状态：", state)
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
    )
    print("response:::", json.dumps(response.model_dump(), indent=2))
    state.messages.append({"role": "assistant", "content": response.content})

    print("助手回复后状态：", state)
    if response.stop_reason != "tool_use":
        state.transition_reason = None
        return False

    results = execute_tool_calls(response.content)
    if not results:
        state.transition_reason = None
        return False

    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    return True


def agent_loop(state: LoopState) -> None:
    """
    执行完整的对话循环。
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

        final_text = extract_text(history[-1]["content"])

        if final_text:
            print(final_text)
        print("---------------------")
