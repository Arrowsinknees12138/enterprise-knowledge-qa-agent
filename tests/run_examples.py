"""Run and verify the actual CLI, then save its results and traces."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def run_cli(mode, case, *options):
    proc = subprocess.run(
        [sys.executable, "-m", "src.knowledge_qa", "--mode", mode,
         "--input", str(EXAMPLES / "inputs" / f"{case}.json"), *options],
        cwd=ROOT,
        capture_output=True,
        encoding="utf-8",
        timeout=620,
    )
    if not proc.stdout.strip():
        raise RuntimeError(f"{case} produced no JSON (exit {proc.returncode})")
    return json.loads(proc.stdout), proc.returncode


def verify(mode, result, exit_code, status, docs, snippet=None):
    assert result["status"] == status, (status, result["status"], result["gap"])
    assert exit_code == int(status in {"step_limit", "tool_error"})
    assert {ref["doc_id"] for ref in result["citations"]} == set(docs)
    if snippet:
        assert snippet in result["answer"]
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

    if status == "conflict":
        assert "30 天" in result["answer"] and "15 天" in result["answer"]
        last_read = max(step["step"] for step in trace if step["action"] == "read_doc")
        searches = [step for step in trace if step["action"] == "search_docs"]
        assert any(step["step"] > last_read for step in searches), "冲突原文读取后应补查"
        assert len({step["arguments"]["query"] for step in searches}) >= 2
        assert result["gap"] and result["follow_up"]
    if status == "insufficient_evidence":
        assert result["gap"] and not result["answer"]
        assert any(
            step["action"] == "search_docs"
            and step["observation"]["ok"] and not step["observation"]["hits"]
            for step in trace
        )
    if status == "tool_error":
        assert result["gap"] and not result["answer"]
        assert any(step["observation"].get("error") == "read_unavailable" for step in trace)
    if status == "step_limit":
        assert result["gap"] and not result["answer"] and len(trace) == 1


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("live", "mock"), required=True)
    args = parser.parse_args()
    output = EXAMPLES / args.mode
    output.mkdir(exist_ok=True)

    def check(name, input_name, status, docs, snippet=None, options=()):
        result, exit_code = run_cli(args.mode, input_name, *options)
        (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify(args.mode, result, exit_code, status, docs, snippet)
        path = " -> ".join(step["action"] for step in result["trace"])
        print(f"PASS {name}: {status} | {path}", flush=True)
        return result["run"].get("usage", {})

    conflict_status = "conflict" if args.mode == "live" else "needs_clarification"
    usages = [
        check("01-answer", "01-answer", "answered", ["D03"], "2 个工作日"),
        check("02-conflict", "02-conflict", conflict_status, ["D01", "D02"]),
        check("03-outside", "03-outside", "insufficient_evidence", []),
        check("04-paraphrase", "04-paraphrase", "answered", ["D03"], "2 个工作日"),
        check("05-follow-up", "05-follow-up", "answered", ["D03"], "2 个工作日"),
        check("06-tool-failure", "06-tool-failure", "tool_error", [], options=("--fail-tool", "read_doc")),
    ]
    corpus = json.loads((ROOT / "data" / "knowledge.json").read_text(encoding="utf-8"))
    for doc in corpus:
        if doc["doc_id"] == "D03":
            doc["passages"]["P1"] = doc["passages"]["P1"].replace("2 个工作日", "4 个工作日")
    with tempfile.TemporaryDirectory() as temp:
        updated = Path(temp) / "updated.json"
        updated.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
        usages.append(check(
            "07-updated-source", "01-answer", "answered", ["D03"], "4 个工作日",
            options=("--data", str(updated)),
        ))
    usages.append(check("08-step-limit", "01-answer", "step_limit", [], options=("--max-steps", "1")))
    if args.mode == "live":
        totals = {key: sum(usage[key] for usage in usages) for key in ("prompt_tokens", "completion_tokens")}
        print(f"Live provider tokens: {json.dumps(totals)}", flush=True)
    print(f"8/8 {args.mode} example checks passed", flush=True)


if __name__ == "__main__":
    main()
