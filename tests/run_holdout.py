"""Run the frozen QA set against the actual CLI and score its cited evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from run_examples import EXAMPLES, ROOT, run_cli


CASES = EXAMPLES / "holdout" / "cases.json"
EVIDENCE_STATUSES = {"answered", "conflict"}


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
        citation_match = grounded and refs == set(case["refs"])
    status_match = result["status"] == case["status"]
    follow_up = result.get("follow_up", "")
    follow_up_match = all(term in follow_up for term in case.get("follow_up_terms", []))
    return {
        "passed": exit_code == 0 and status_match and citation_match and follow_up_match,
        "status_match": status_match,
        "actual_refs": sorted(refs),
        "citation_match": citation_match,
        "follow_up_match": follow_up_match,
        "unsupported_answer": result["status"] == "answered" and (not status_match or not citation_match),
        "first_search_top3": top3,
        "first_search_hit_at_3": bool(relevant.intersection(top3)) if relevant else None,
        "first_search_all_relevant_at_3": relevant.issubset(top3) if relevant else None,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", action="append", help="Run specific frozen case IDs.")
    args = parser.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        parser.error("Set DEEPSEEK_API_KEY before running the holdout.")

    raw = CASES.read_bytes()
    cases = json.loads(raw)
    unknown = set(args.case or []) - {case["id"] for case in cases}
    if unknown:
        parser.error(f"Unknown cases: {', '.join(sorted(unknown))}")
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
    results = []
    with tempfile.TemporaryDirectory() as temp:
        input_path = Path(temp) / "question.json"
        for case in cases:
            payload = {key: case[key] for key in ("question", "history") if key in case}
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result, exit_code = run_cli("live", "deepseek", input_path)
            retried = result["status"] == "model_error" and result["gap"] == "模型连接失败、超时或返回格式无效"
            if retried:
                print(f"RETRY {case['id']}: transport error", flush=True)
                result, exit_code = run_cli("live", "deepseek", input_path)
            checks = score(case, result, exit_code)
            results.append({
                "id": case["id"], "expected_status": case["status"], "expected_refs": case["refs"],
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
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cases_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_sha256": hashlib.sha256((ROOT / "data" / "knowledge.json").read_bytes()).hexdigest(),
        "metrics": metrics,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return int(metrics["passed"] != metrics["cases"])


if __name__ == "__main__":
    sys.exit(main())
