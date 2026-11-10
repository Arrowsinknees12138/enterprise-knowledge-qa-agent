import json
import tempfile
import unittest
from pathlib import Path

from run_holdout import case_options, score


class HoldoutScoringTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "status": "answered",
            "refs": ["D04/P2"],
            "relevant_docs": ["D04"],
        }
        quote = "企业邮箱回收站中的邮件自删除起保留 30 天。"
        self.result = {
            "status": "answered",
            "citations": [{"doc_id": "D04", "passage_id": "P2", "quote": quote}],
            "trace": [
                {"action": "search_docs", "observation": {"ok": True, "hits": [{"doc_id": "D07"}]}},
                {"action": "search_docs", "observation": {"ok": True, "hits": [{"doc_id": "D04"}]}},
                {"action": "read_doc", "observation": {
                    "ok": True, "doc_id": "D04", "passages": [{"passage_id": "P2", "quote": quote}],
                }},
            ],
        }

    def test_later_retrieval_does_not_count_as_a_first_search_hit(self):
        checks = score(self.case, self.result, 0)
        self.assertTrue(checks["passed"])
        self.assertFalse(checks["first_search_hit_at_3"])

    def test_matching_ids_with_changed_quote_do_not_support_answer(self):
        self.result["citations"][0]["quote"] = "企业邮箱回收站中的邮件自删除起保留 60 天。"
        checks = score(self.case, self.result, 0)
        self.assertFalse(checks["citation_match"])
        self.assertTrue(checks["unsupported_answer"])

    def test_answering_an_insufficient_evidence_case_is_unsupported(self):
        self.case.update(status="insufficient_evidence", refs=[])
        checks = score(self.case, self.result, 0)
        self.assertFalse(checks["status_match"])
        self.assertTrue(checks["unsupported_answer"])

    def test_expected_tool_limit_checks_exit_observation_and_budget(self):
        self.case.update(status="tool_limit", refs=[], observation_error="tool_limit",
                         expected_run={"tool_calls_used": 0})
        self.result.update(
            status="tool_limit", citations=[], run={"tool_calls_used": 0},
            trace=[{"action": "search_docs", "observation": {"ok": False, "error": "tool_limit"}}],
        )
        self.assertTrue(score(self.case, self.result, 1)["passed"])
        self.assertFalse(score(self.case, self.result, 0)["passed"])
        self.result["run"]["tool_calls_used"] = 1
        self.assertFalse(score(self.case, self.result, 1)["passed"])

    def test_transition_checked_after_required_read(self):
        self.case["next_after_read"] = {"docs": ["D04"], "action": "finish"}
        self.assertFalse(score(self.case, self.result, 0)["action_match"])
        self.result["trace"].append({"action": "finish", "observation": {"ok": True}})
        self.assertTrue(score(self.case, self.result, 0)["passed"])

    def test_variant_changes_one_passage_without_mutating_base(self):
        corpus = [{"doc_id": "D01", "passages": {"P1": "30 天", "P2": "其他条款"}}]
        case = {
            "max_tool_calls": 0,
            "corpus_edit": {"doc_id": "D01", "passage_id": "P1", "text": "15 天"},
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "variant.json"
            self.assertEqual(case_options(case, corpus, path), ["--max-tool-calls", "0", "--data", str(path)])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))[0]["passages"],
                             {"P1": "15 天", "P2": "其他条款"})
        self.assertEqual(corpus[0]["passages"]["P1"], "30 天")
