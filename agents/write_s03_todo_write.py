import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import ToolParam
from dotenv import load_dotenv

# 获取.env文件中的环境变量，覆盖已有变量
load_dotenv(override=True)


# 优先使用其他模型
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
PLAN_REMINDER_INTERVAL = 3

SYSTEM = f"""你是当前工作目录{WORKDIR}的 编码智能体，使用工具解决任务。
处理多步骤任务时，使用 todo 工具维护会话计划。
有多个步骤时，同一时刻只保持一个步骤处于 in_progress 状态。
随着工作推进及时刷新计划。优先用工具行动，少用文字解释。"""


@dataclass
class PlanItem:
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    # 执行了多少轮
    rounds_since_update: int = 0


class TodoManager:
    """"""

    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        """
        更新计划，将新的项目列表写入状态。
        1. 检查项目数量不超过12个
        2. 检查每个项目的状态是 pending, in_progress 或 completed
        3. 检查 in_progress 状态的项目数量不超过1个
        """
        if len(items) > 12:
            raise ValueError("保持会话计划简短（最多12个项目）")

        normalized = []
        in_progress_count = 0
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("activeForm", "")).strip()

            if not content:
                raise ValueError(f"Item {index} 必须有内容")
            if status not in ["pending", "in_progress", "completed"]:
                raise ValueError(
                    f"Item {index} 状态必须是 pending, in_progress 或 completed"
                )
            if status == "in_progress":
                in_progress_count += 1

            normalized.append(
                PlanItem(content=content, status=status, active_form=active_form)
            )

        if in_progress_count > 1:
            raise ValueError("只能有一个处于进行中的任务")

        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        """
        执行了一轮但没有更新计划，增加 rounds_since_update。
        """
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        """
        如果计划为空或 rounds_since_update 小于 PLAN_REMINDER_INTERVAL，返回 None；
        否则返回提醒文本。
        """
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>在继续之前刷新您当前的计划.</reminder>"

    def render(self) -> str:
        if not self.state.items:
            return " 还没有会话计划"

        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ] ",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        completed = sum(1 for item in self.state.items if item.status == "completed")
        lines.append(f"已完成 {completed}/{len(self.state.items)} 项任务")
        return "\n".join(lines)


TODO = TodoManager()


def safe_path(p: str) -> Path:
    """安全地解析路径，返回绝对路径"""
    # 拼接之后返回绝对路径
    path = (WORKDIR / p).resolve()
    # 检查path 是否是WORKDIR 的子路径
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"路径{path}超出了工作目录")
    return path


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


# 工具的分发映射


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容
    path: 文件路径
    limit: 最大读取字节数
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} 更多)"]
        return "\n".join(lines)[:5000]
    except Exception as e:
        return f"Error: {str(e)}"


# 新增claude api 工具json schema


def run_write(path: str, content: str) -> str:
    """将内容写入文件
    path: 文件路径
    content: 要写入的内容
    """
    try:
        fp = safe_path(path)
        # 创建父目录（如果不存在）
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)

        return f"成功将 {len(content)} 字符写入到 {path}"
    except Exception as e:
        return f"Error: {str(e)}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    编辑文件，将 old_text 替换为 new_text，仅支持首次匹配项
    path: 文件路径
    old_text: 要替换的旧文本
    new_text: 要替换的新文本
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"错误： {path}中未找到：{old_text}"
        # 仅替换第一次匹配
        content = content.replace(old_text, new_text, 1)
        fp.write_text(content)
        return f"成功将 {old_text} 替换为 {new_text} 并写入到 {path}"
    except Exception as e:
        return f"Error: {str(e)}"


# 工具的分发映射
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
}

# claude api工具的json schema
TOOLS: list[ToolParam] = [
    {
        "name": "bash",
        "description": "当前是Windows系统，在当前工作目录运行任意shell命令",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "todo",
        "description": "为多步骤工作重写当前会话计划.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous label.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        },
    },
]


def normalize_messages(messages: list) -> list:
    """遵循claude api 消息格式规范
    1. 过滤掉claude api 不认识的键“_”开头的
    2. 每个 tool_use 块必须有匹配的 tool_result (通过 tool_use_id 关联)
    3. user / assistant 消息必须严格交替
    """
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            clean["content"] = [
                {k: v for k, v in block.items() if not k.startswith("_")}
                for block in msg["content"]
                if isinstance(block, dict)
            ]
        else:
            clean["content"] = msg.get("content", "")

        cleaned.append(clean)

    # 2. 收集已存在的工具响应IDs
    existing_results = set()
    for msg in cleaned:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))

    # 3. 找到孤立的tool_use, 并添加tool_result; (用户取消、系统报错会出现)
    for msg in cleaned:
        if msg["role"] != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict):
                continue
            if (
                block.get("type") == "tool_use"
                and block.get("id") not in existing_results
            ):
                cleaned.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": "已取消作废",
                            }
                        ],
                    }
                )

    # 4. 合并连续相同的role消息（当执行工具时，会出现连续相同role的情况）
    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev_c = (
                prev["content"]
                if isinstance(prev["content"], list)
                else [{"type": "text", "text": str(prev["content"])}]
            )

            curr_c = (
                msg["content"]
                if isinstance(msg["content"], list)
                else [{"type": "text", "text": str(msg["content"])}]
            )
            # 合并相同role的消息内容,更新前一条content
            prev["content"] = prev_c + curr_c
        else:
            # 不同role的消息，直接追加
            merged.append(msg)
    return merged


def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        if isinstance(block, dict) and "text" in block:
            texts.append(block["text"])
    return "\n".join(texts).strip()


@dataclass
class LoopState:
    """循环状态，包含当前的消息、轮数和转换原因"""

    messages: list
    turn_count: int = 1
    transition_reason: str | None = None


def agent_loop(messages: list) -> None:
    """
    执行完整的对话循环。
    执行一轮时对话时，判断响应结果，如果调用工具，那么拿到工具响应结果追加到消息历史中，
    继续执行下一轮对话，直到不再调用工具则结束循环。
    """
    while True:
        """执行一轮对话，调用工具时继续循环，不调用工具时跳出循环"""
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=normalize_messages(messages),
            tools=TOOLS,
            max_tokens=8000,
        )

        print(
            "打印response:",
            json.dumps(response.model_dump(), indent=2, ensure_ascii=True),
        )
        # messages.append({"role": "assistant", "content": response.content})
        # 将TextBlock 和 ToolUseBlock 转换为字典格式，便于后面过滤
        messages.append(
            {
                "role": "assistant",
                "content": [block.model_dump() for block in response.content],
            }
        )

        # 不调用工具，直接跳出循环
        if response.stop_reason != "tool_use":
            return

        results = []
        used_todo = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = TOOL_HANDLERS.get(block.name)
            output = (
                handler(**block.input) if handler else f"Unknown tool: {block.name}"
            )

            print(f"> {block.name}:")
            print(output[:200])
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )
            if block.name == "todo":
                used_todo = True

        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            TODO.note_round_without_update()
            reminder = TODO.reminder()
            if reminder:
                results.insert(0, {"type": "text", "text": reminder})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "quit", "exit"):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 会话结束后拿到最后一次响应结果（LLM响应），截取最大长度度字符
        response_content = history[-1]["content"]

        final_text = extract_text(response_content)
        if final_text:
            print(final_text)
        print()
