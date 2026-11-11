import argparse
import json
import sys
from pathlib import Path

from .agent import KnowledgeAgent
from .knowledge import KnowledgeBase, normalize
from .models import LIVE_PROVIDERS, LiveModel, MockModel


DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "knowledge.json"
TRACE_ACTIONS = {"search_docs", "read_doc", "finish", "model_error", "step_limit", "tool_limit", "time_limit", None}


def load_input(args):
    if args.input:
        with args.input.open(encoding="utf-8-sig") as file:
            payload = json.load(file)
    else:
        payload = {"question": args.question}
    if not isinstance(payload, dict) or not isinstance(payload.get("question"), str) or not normalize(payload["question"]):
        raise ValueError("question 必须包含文字或数字")
    history = payload.get("history", [])
    if not isinstance(history, list) or any(
        not isinstance(entry, dict)
        or entry.get("role") not in ("user", "assistant")
        or not isinstance(entry.get("content"), str)
        or not entry["content"].strip()
        for entry in history
    ):
        raise ValueError("history 必须是由 user/assistant 与非空 content 组成的消息数组")
    return payload["question"], [{"role": entry["role"], "content": entry["content"]} for entry in history]


def redact_trace(trace):
    return [
        {
            "step": entry["step"],
            "action": entry["action"] if entry["action"] in TRACE_ACTIONS else "unknown_action",
            "next_action": entry["next_action"] if entry["next_action"] in TRACE_ACTIONS else "unknown_action",
            "tool_calls_remaining": entry["tool_calls_remaining"],
            "observation": {
                key: value for key, value in entry["observation"].items()
                if key in {"ok", "error", "status", "total", "cached", "next_offset"}
            },
        }
        for entry in trace
    ]


def main():
    parser = argparse.ArgumentParser(description="企业知识问答 Agent")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="直接输入问题")
    source.add_argument("--input", type=Path, help="包含 question 和可选 history 的 JSON 文件")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--provider", choices=LIVE_PROVIDERS, help="真实模型提供方，默认 dashscope")
    parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS, help="替换合成知识库 JSON")
    parser.add_argument("--max-steps", type=int, choices=range(1, 21), default=8)
    parser.add_argument("--max-tool-calls", type=int, choices=range(21), default=6, help="实际业务工具调用上限")
    parser.add_argument("--max-seconds", type=int, default=120, help="单次问答的运行时间预算")
    parser.add_argument("--allow-doc", action="append", help="仅使用指定文档 ID，可以重复传入")
    parser.add_argument("--redact-trace", action="store_true", help="Trace 仅输出动作、状态和用量")
    parser.add_argument("--fail-tool", choices=("search_docs", "read_doc"), help="注入持续业务工具故障")
    args = parser.parse_args()
    if args.provider and args.mode != "live":
        parser.error("--provider 需要配合 --mode live")
    if args.max_seconds <= 0:
        parser.error("--max-seconds 必须大于 0")
    try:
        question, history = load_input(args)
        allowed_docs = set(args.allow_doc) if args.allow_doc else None
        knowledge = KnowledgeBase(args.data, args.fail_tool, allowed_docs=allowed_docs)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    model = LiveModel(args.provider or "dashscope") if args.mode == "live" else MockModel()
    result = KnowledgeAgent(model, knowledge, args.max_steps, args.max_tool_calls, args.max_seconds).run(question, history)
    if args.redact_trace:
        result["trace"] = redact_trace(result["trace"])
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] in ("tool_error", "model_error", "step_limit", "tool_limit", "time_limit"))


if __name__ == "__main__":
    sys.exit(main())
