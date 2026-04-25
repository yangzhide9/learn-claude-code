import os
import subprocess
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic.types import ToolParam

# 模型url
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 初始化客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 模型
MODEL = os.environ["MODEL_ID"]

# SYSTEM PROMPT
SYSTEM = f"你是当前工作目录{os.getcwd()}的 编码智能体可以使用任意 bash 命令"

# 工具列表
TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "在当前工作空间，可运行任意 shell命令",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"}
            },
            "required": ["command"],
        },
    }
]

# 超时
TIMEOUT = 120


@dataclass
class LoopState:
    """循环状态
    message: 历史消息列表
    turn_count: 循环计数
    transition_reason: 循环原因
    """

    message: list
    turn_count: int = 1
    transition_reason: str | None = None


def run_bash(command: str) -> str:
    """执行 bash 命令，返回输出结果。"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(item in command for item in dangerous):
        return "Error: 危险的命令"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"Error: 命令执行超时{TIMEOUT}"

    output = str((result.stdout + result.stderr).strip())
    return output[:50000] if output else "(指令没有任何输出)"


def extract_text(content) -> str:
    """从内容中提取文本，返回拼接后的字符串。"""
    if not isinstance(content, list):
        return ""

    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)

    return "\n".join(texts).strip()


def execute_tool_calls(response_content) -> list[dict]:
    results = []
    for block in response_content:
        if block.type != "tool_use":
            continue
        command = block.input["command"]
        print(f"\033[33m$ {command}\033[0m")
        output = run_bash(command)
        print(output[:200])
        results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            }
        )
    return results
