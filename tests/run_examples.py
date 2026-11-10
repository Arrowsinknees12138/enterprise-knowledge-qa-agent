"""Run and verify the actual CLI, then save its results and traces."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def run_cli(mode, provider, case, *options):
    command = [sys.executable, "-m", "src.knowledge_qa", "--mode", mode]
    if mode == "live":
        command.extend(["--provider", provider])
    command.extend(["--input", str(EXAMPLES / "inputs" / f"{case}.json"), *options])
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        timeout=620,
    )
    if not proc.stdout.strip():
        raise RuntimeError(f"{case} produced no JSON (exit {proc.returncode})")
    return json.loads(proc.stdout), proc.returncode


def actions_after_reads(result, doc_ids):
    pending = set(doc_ids)
    for index, step in enumerate(result["trace"]):
        if step["action"] == "read_doc" and step["observation"]["ok"]:
            pending.discard(step["observation"]["doc_id"])
            if not pending:
                return [later["action"] for later in result["trace"][index + 1:]]
    raise AssertionError(f"Missing successful reads: {sorted(pending)}")


def verify(mode, result, exit_code, status, docs, snippet=None):
    assert result["status"] == status, (status, result["status"], result["gap"])
    assert exit_code == int(status in {"step_limit", "tool_limit", "tool_error"})
    cited = {ref["doc_id"] for ref in result["citations"]}
    if docs is not None:
        assert cited == set(docs)
    if status == "answered":
        assert cited
    if snippet:
        assert snippet in result["answer"] and all(snippet in ref["quote"] for ref in result["citations"])
    trace = result["trace"]
    assert [step["step"] for step in trace] == list(range(1, len(trace) + 1))
    for step, next_step in zip(trace, trace[1:]):
        assert step["next_action"] == next_step["action"]
    observed = {
        (step["observation"]["doc_id"], passage["passage_id"], passage["quote"])
        for step in trace if step["action"] == "read_doc" and step["observation"]["ok"]
        for passage in step["observation"]["passages"]
    }
    for ref in result["citations"]:
        assert (ref["doc_id"], ref["passage_id"], ref["quote"]) in observed
    assert result["run"]["mode"] == mode
    assert result["run"]["api_requests"] == (len(trace) if mode == "live" else 0)
    remaining = result["run"]["max_tool_calls"]
    for step in trace:
        observation = step["observation"]
        if (step["action"] in {"search_docs", "read_doc"} and not observation.get("cached")
                and observation.get("error") not in {"invalid_action", "tool_limit"}):
            remaining -= 1
        assert step["tool_calls_remaining"] == remaining >= 0
    assert result["run"]["tool_calls_used"] == result["run"]["max_tool_calls"] - remaining

    if status == "conflict":
        assert "30 天" in result["answer"] and "15 天" in result["answer"]
        assert "search_docs" in actions_after_reads(result, cited), "冲突原文读取后应补查"
        searches = [step for step in trace if step["action"] == "search_docs"]
        assert len({step["arguments"]["query"] for step in searches}) >= 2
        assert result["gap"] and result["follow_up"]
    if status == "insufficient_evidence":
        assert result["gap"] and not result["answer"] and not result["citations"]
    if status == "tool_error":
        assert result["gap"] and not result["answer"]
        assert any(step["observation"].get("error") in {"read_unavailable", "search_unavailable"} for step in trace)
    if status == "step_limit":
        assert result["gap"] and not result["answer"] and len(trace) == 1
    if status == "tool_limit":
        assert result["gap"] and not result["answer"] and not result["citations"]
        assert remaining == 0 and trace[-1]["observation"]["error"] == "tool_limit"
    if status == "needs_clarification":
        assert result["gap"] and result["follow_up"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "mock"), required=True)
    parser.add_argument("--provider", choices=("dashscope", "deepseek"))
    parser.add_argument("--case", action="append", help="只重跑指定用例，如 12-multipassage，可重复")
    args = parser.parse_args()
    if args.provider and args.mode != "live":
        parser.error("--provider requires --mode live")
    provider = args.provider or "dashscope"
    if args.mode == "live":
        key_variable = "DEEPSEEK_API_KEY" if provider == "deepseek" else "DASHSCOPE_API_KEY"
        if not os.environ.get(key_variable, "").strip():
            parser.error(f"Set {key_variable} before running live examples.")
    output = EXAMPLES / ("deepseek" if provider == "deepseek" else args.mode)
    output.mkdir(exist_ok=True)
    runs = {}
    failed = []
    available = set()

    def check(name, input_name, status, docs, snippet=None, options=(), extra_check=None):
        available.add(name)
        if args.case and name not in args.case:
            return
        try:
            result, exit_code = run_cli(args.mode, provider, input_name, *options)
            if (args.mode == "live" and result["status"] == "model_error"
                    and result["gap"] == "模型连接失败、超时或返回格式无效"):
                print(f"RETRY {name}: model connection or timeout", flush=True)
                result, exit_code = run_cli(args.mode, provider, input_name, *options)
            (output / f"{name}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            verify(args.mode, result, exit_code, status, docs, snippet)
            if args.mode == "live":
                assert result["run"]["provider"] == provider
            if extra_check:
                extra_check(result)
        except (AssertionError, RuntimeError, subprocess.TimeoutExpired) as error:
            failed.append(name)
            print(f"FAIL {name}: {error or 'check failed'}", flush=True)
            return
        path = " -> ".join(step["action"] for step in result["trace"])
        print(f"PASS {name}: {status} | {path}", flush=True)
        runs[name] = result["run"]

    def verify_no_hits(result):
        searches = [step["observation"] for step in result["trace"] if step["action"] == "search_docs"]
        assert any(observation["ok"] and not observation["hits"] for observation in searches)

    def verify_resolved(result):
        assert {ref["doc_id"] for ref in result["citations"]} <= {"D01", "D02"}
        assert actions_after_reads(result, {"D01", "D02"}) == ["finish"], "一致的原文应直接结束"

    def verify_missing_detail(result):
        assert actions_after_reads(result, {"D03"})[-1] == "finish"

    def expect_passage(doc_id, passage_id, also_read=()):
        def check_passage(result):
            assert {(ref["doc_id"], ref["passage_id"]) for ref in result["citations"]} == {(doc_id, passage_id)}
            actions_after_reads(result, {doc_id, *also_read})
        return check_passage

    conflict_status = "conflict" if args.mode == "live" else "needs_clarification"
    check("01-answer", "01-answer", "answered", ["D03"], "2 个工作日")
    check("02-conflict", "02-conflict", conflict_status, ["D01", "D02"])
    check("03-outside", "03-outside", "insufficient_evidence", [], extra_check=verify_no_hits)
    check("04-paraphrase", "04-paraphrase", "answered", ["D03"], "2 个工作日")
    check("05-follow-up", "05-follow-up", "answered", ["D03"], "2 个工作日")
    check("06-tool-failure", "06-tool-failure", "tool_error", [], options=("--fail-tool", "read_doc"))

    corpus = json.loads((ROOT / "data" / "knowledge.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temp:
        updated = Path(temp) / "updated.json"

        def replace_passage(doc_id, replacement):
            variant = deepcopy(corpus)
            doc = next(doc for doc in variant if doc["doc_id"] == doc_id)
            doc["passages"]["P1"] = replacement
            updated.write_text(json.dumps(variant, ensure_ascii=False), encoding="utf-8")
            return ("--data", str(updated))

        check(
            "07-updated-source", "01-answer", "answered", ["D03"], "4 个工作日",
            options=replace_passage("D03", "员工提交完整的门禁权限申请后，行政会在 4 个工作日内开通门禁。"),
        )
        check("08-step-limit", "01-answer", "step_limit", [], options=("--max-steps", "1"))
        if args.mode == "live":
            check(
                "09-resolved-conflict", "02-conflict", "answered", None, "30 天",
                options=replace_passage("D02", "员工应在出差结束后 30 天内提交差旅费用报销申请。"),
                extra_check=verify_resolved,
            )
            check(
                "10-missing-detail", "01-answer", "insufficient_evidence", [],
                options=replace_passage("D03", "新员工须通过内部服务台提交门禁权限申请，并填写办公地点和员工编号。"),
                extra_check=verify_missing_detail,
            )
    if args.mode == "live":
        check("11-attachment-noise", "11-attachment-noise", "answered", ["D04"], "25 MB",
              extra_check=expect_passage("D04", "P1", ("D07",)))
        check("12-multipassage", "12-multipassage", "answered", ["D04"], "30 天",
              extra_check=expect_passage("D04", "P2"))
        check("13-explicit-supersession", "13-explicit-supersession", "answered", ["D09"], "5 个工作日",
              extra_check=expect_passage("D09", "P1", ("D08",)))
        check("14-audience-scope", "14-audience-scope", "answered", ["D11"], "1 个工作日",
              extra_check=expect_passage("D11", "P1"))
        check("15-ambiguous", "15-ambiguous", "needs_clarification", [])
    check("16-search-failure", "01-answer", "tool_error", [], options=("--fail-tool", "search_docs"))
    check("17-tool-limit", "01-answer", "tool_limit", [], options=("--max-tool-calls", "1"))
    check("18-finish-at-limit", "01-answer", "answered", ["D03"], "2 个工作日",
          options=("--max-tool-calls", "2"))
    check("19-password", "19-password", "answered", ["D12"], "15 分钟")
    unknown = set(args.case or []) - available
    if unknown:
        parser.error(f"Unknown {args.mode} cases: {', '.join(sorted(unknown))}")
    if args.mode == "live":
        totals = {
            key: sum(run["usage"][key] for run in runs.values())
            for key in ("prompt_tokens", "completion_tokens")
        }
        print(f"Live provider tokens: {json.dumps(totals)}", flush=True)
    print(f"{len(runs)}/{len(runs) + len(failed)} {args.mode} example checks passed", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
