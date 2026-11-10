"""Run the frozen QA set against the actual CLI and score its cited evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from run_examples import EXAMPLES, ROOT, run_cli


CASES = EXAMPLES / "holdout" / "cases.json"
EVIDENCE_STATUSES = {"answered", "conflict"}
FAILURE_STATUSES = {"model_error", "step_limit", "tool_error", "tool_limit"}


def next_after_read(trace, doc_ids):
    pending = set(doc_ids)
    for current, following in zip(trace, trace[1:]):
        if current["action"] == "read_doc" and current["observation"]["ok"]:
            pending.discard(current["observation"]["doc_id"])
            if not pending:
                return following["action"]
    return None


def score(case, result, exit_code):
    trace = result["trace"]
    first_search = next(
        (step["observation"] for step in trace if step["action"] == "search_docs"), {}
    )
    top3 = [hit["doc_id"] for hit in first_search.get("hits", [])[:3]]
    relevant = set(case["relevant_docs"])
    observed = {
        (step["observation"]["doc_id"], passage["passage_id"], passage["quote"])
        for step in trace if step["action"] == "read_doc" and step["observation"]["ok"]
        for passage in step["observation"]["passages"]
    }
    refs = {f'{ref["doc_id"]}/{ref["passage_id"]}' for ref in result["citations"]}
    grounded = all((ref["doc_id"], ref["passage_id"], ref["quote"]) in observed for ref in result["citations"])
    if "allowed_context_refs" in case:
        citation_match = grounded and refs <= set(case["allowed_context_refs"])
    else:
        accepted = case.get("acceptable_refs", [case["refs"]])
        citation_match = grounded and any(refs == set(candidate) for candidate in accepted)
    status_match = result["status"] == case["status"]
    follow_up = result.get("follow_up", "")
    follow_up_match = all(term in follow_up for term in case.get("follow_up_terms", []))
    if case["status"] in {"conflict", "needs_clarification"}:
        follow_up_match = follow_up_match and bool(follow_up.strip())
    expected_error = case.get("observation_error")
    observation_match = not expected_error or any(step["observation"].get("error") == expected_error for step in trace)
    run = result.get("run", {})
    budget_match = all(run.get(key) == value for key, value in case.get("expected_run", {}).items())
    transition = case.get("next_after_read")
    action_match = not transition or next_after_read(trace, transition["docs"]) == transition["action"]
    exit_match = exit_code == int(case["status"] in FAILURE_STATUSES)
    return {
        "passed": all((exit_match, status_match, citation_match, follow_up_match,
                       observation_match, budget_match, action_match)),
        "status_match": status_match,
        "actual_refs": sorted(refs),
        "citation_match": citation_match,
        "follow_up_match": follow_up_match,
        "observation_match": observation_match,
        "budget_match": budget_match,
        "action_match": action_match,
        "exit_match": exit_match,
        "unsupported_answer": result["status"] == "answered" and (not status_match or not citation_match),
        "first_search_top3": top3,
        "first_search_hit_at_3": bool(relevant.intersection(top3)) if relevant else None,
        "first_search_all_relevant_at_3": relevant.issubset(top3) if relevant else None,
    }


def case_options(case, base_corpus, variant_path):
    options = []
    for name in ("max_steps", "max_tool_calls", "fail_tool"):
        if name in case:
            options.extend(["--" + name.replace("_", "-"), str(case[name])])
    if edit := case.get("corpus_edit"):
        variant = deepcopy(base_corpus)
        document = next(doc for doc in variant if doc["doc_id"] == edit["doc_id"])
        document["passages"][edit["passage_id"]] = edit["text"]
        variant_path.write_text(json.dumps(variant, ensure_ascii=False), encoding="utf-8")
        options.extend(["--data", str(variant_path)])
    return options


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CASES, help="Frozen cases JSON file.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", help="Run specific frozen case IDs.")
    args = parser.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        parser.error("Set DEEPSEEK_API_KEY before running the holdout.")

    raw = args.dataset.read_bytes()
    corpus_bytes = (ROOT / "data" / "knowledge.json").read_bytes()
    base_corpus = json.loads(corpus_bytes)
    cases = json.loads(raw)
    unknown = set(args.case or []) - {case["id"] for case in cases}
    if unknown:
        parser.error(f"Unknown cases: {', '.join(sorted(unknown))}")
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
    results = []
    with tempfile.TemporaryDirectory() as temp:
        input_path = Path(temp) / "question.json"
        variant_path = Path(temp) / "corpus.json"
        for case in cases:
            payload = {key: case[key] for key in ("question", "history") if key in case}
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            options = case_options(case, base_corpus, variant_path)
            result, exit_code = run_cli("live", "deepseek", input_path, *options)
            retried = result["status"] == "model_error" and result["gap"] == "模型连接失败、超时或返回格式无效"
            if retried:
                print(f"RETRY {case['id']}: transport error", flush=True)
                result, exit_code = run_cli("live", "deepseek", input_path, *options)
            checks = score(case, result, exit_code)
            results.append({
                "id": case["id"], "group": case.get("group", "holdout"),
                "expected_status": case["status"], "expected_refs": case["refs"],
                "checks": checks, "retried": retried, "exit_code": exit_code, "result": result,
            })
            outcome = "PASS" if checks["passed"] else "FAIL"
            print(f"{outcome} {case['id']}: {result['status']} {checks['actual_refs']}", flush=True)

    with_evidence = [item for item in results if item["expected_status"] in EVIDENCE_STATUSES]
    searchable = [item for item in results if item["checks"]["first_search_hit_at_3"] is not None]
    metrics = {
        "cases": len(results),
        "passed": sum(item["checks"]["passed"] for item in results),
        "status_matches": sum(item["checks"]["status_match"] for item in results),
        "citation_cases": len(with_evidence),
        "citation_exact_matches": sum(item["checks"]["citation_match"] for item in with_evidence),
        "unsupported_answers": sum(item["checks"]["unsupported_answer"] for item in results),
        "first_search_cases": len(searchable),
        "first_search_hits_at_3": sum(item["checks"]["first_search_hit_at_3"] for item in searchable),
        "first_search_all_relevant_at_3": sum(item["checks"]["first_search_all_relevant_at_3"] for item in searchable),
    }
    groups = {}
    for item in results:
        counts = groups.setdefault(item["group"], {"cases": 0, "passed": 0})
        counts["cases"] += 1
        counts["passed"] += item["checks"]["passed"]
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "metrics": metrics,
        "groups": groups,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return int(metrics["passed"] != metrics["cases"])


if __name__ == "__main__":
    sys.exit(main())
