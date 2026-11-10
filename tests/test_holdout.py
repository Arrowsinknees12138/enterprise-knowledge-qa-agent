import unittest

from run_holdout import score


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
