import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.knowledge_qa.agent import Evidence, KnowledgeAgent
from src.knowledge_qa.__main__ import load_input, redact_trace
from src.knowledge_qa.knowledge import KnowledgeBase
from src.knowledge_qa.models import Action, LiveModel, MockModel, ModelError


CORPUS = Path(__file__).resolve().parents[1] / "data" / "knowledge.json"


class ScriptedModel:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.messages = []

    def choose(self, messages, *, deadline=None):
        self.messages.append(list(messages))
        name, arguments = next(self.decisions)
        return Action(name, {"reason": "测试动作", **arguments}, str(len(self.messages)))

    def metadata(self):
        return {"mode": "test-double"}


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.knowledge = KnowledgeBase(CORPUS)

    def test_search_returns_both_versions_without_exposing_text(self):
        result = self.knowledge.search_docs("出差报销期限")
        self.assertEqual([doc["doc_id"] for doc in result["hits"]], ["D01", "D02"])
        self.assertNotIn("passages", result["hits"][0])

    def test_search_supports_local_aliases_and_unicode_normalization(self):
        for query, doc_id in (("刷 卡进楼", "D03"), ("ＭＡＩＬ attachment", "D04")):
            with self.subTest(query=query):
                self.assertEqual(self.knowledge.search_docs(query)["hits"][0]["doc_id"], doc_id)

    def test_aliases_recall_paraphrase_missed_by_canonical_term_alone(self):
        query = "新入职的同事交完申请后，要等多久才能刷卡进楼？"
        self.assertEqual([hit["doc_id"] for hit in self.knowledge.search_docs(query)["hits"]], ["D03"])
        self.knowledge.documents["D03"]["terms"] = ["门禁"]
        self.assertEqual(self.knowledge.search_docs(query)["hits"], [])

    def test_retrieval_keeps_distractors_and_ranks_specific_mailbox_matches_first(self):
        hits = self.knowledge.search_docs("企业邮箱单封邮件附件大小上限")["hits"]
        self.assertEqual([hit["doc_id"] for hit in hits], ["D04", "D07"])
        self.assertGreater(hits[0]["score"], hits[1]["score"])
        self.assertNotIn("passages", hits[0])

    def test_unknown_document_and_search_failure_are_explicit(self):
        self.assertEqual(self.knowledge.read_doc("missing")["error"], "doc_not_found")
        broken = KnowledgeBase(CORPUS, "search_docs")
        self.assertEqual(broken.search_docs("门禁")["error"], "search_unavailable")
        self.assertEqual(self.knowledge.search_docs("宠物")["hits"], [])

    def test_runtime_rejects_unsearched_reads_and_invalid_arguments(self):
        evidence = Evidence(self.knowledge)
        for action, args in [
            ("read_doc", {"doc_id": "D03"}),
            ("read_doc", {"doc_id": []}),
            ("search_docs", {"query": ""}),
            ("search_docs", {"query": "门禁", "unknown": True}),
        ]:
            with self.subTest(action=action, args=args):
                self.assertEqual(evidence.execute(action, args)["error"], "invalid_action")

    def test_runtime_does_not_accept_a_single_unchecked_policy_version(self):
        evidence = Evidence(self.knowledge)
        evidence.execute("search_docs", {"query": "差旅报销"})
        evidence.execute("read_doc", {"doc_id": "D02"})
        result = evidence.execute("finish", {
            "status": "answered", "citations": [{"doc_id": "D02", "passage_id": "P1"}],
            "gap": "", "follow_up": "",
        })
        self.assertEqual(result["error"], "invalid_action")

    def test_runtime_rejects_wrong_field_in_the_correct_document(self):
        evidence = Evidence(self.knowledge, question="企业邮箱回收站保留多久？")
        evidence.execute("search_docs", {"query": "企业邮箱"})
        evidence.execute("read_doc", {"doc_id": "D04"})
        args = {"status": "answered", "gap": "", "follow_up": ""}
        wrong = evidence.finish({**args, "citations": [{"doc_id": "D04", "passage_id": "P3"}]})
        self.assertIn("所问字段不符", wrong["message"])
        correct = evidence.finish({**args, "citations": [{"doc_id": "D04", "passage_id": "P2"}]})
        self.assertTrue(correct["ok"])

    def test_runtime_rejects_a_quote_from_the_wrong_product(self):
        evidence = Evidence(self.knowledge, question="企业邮箱回收站保留多久？")
        evidence.search({"query": "企业邮箱 云盘"})
        for doc_id in ("D04", "D07"):
            evidence.execute("read_doc", {"doc_id": doc_id})
        args = {"status": "answered", "gap": "", "follow_up": ""}
        wrong = evidence.finish({**args, "citations": [{"doc_id": "D07", "passage_id": "P2"}]})
        self.assertFalse(wrong["ok"])
        self.assertIn("D04/P2", wrong["message"])
        correct = evidence.finish({**args, "citations": [{"doc_id": "D04", "passage_id": "P2"}]})
        self.assertTrue(correct["ok"])

    def test_runtime_rejects_an_unrelated_extra_citation(self):
        evidence = Evidence(self.knowledge, question="企业邮箱回收站保留多久？")
        evidence.search({"query": "企业邮箱 云盘"})
        for doc_id in ("D04", "D07"):
            evidence.execute("read_doc", {"doc_id": doc_id})
        refs = [{"doc_id": doc_id, "passage_id": "P2"} for doc_id in ("D04", "D07")]
        result = evidence.finish({"status": "answered", "citations": refs, "gap": "", "follow_up": ""})
        self.assertFalse(result["ok"])
        self.assertIn("D07/P2", result["message"])

    def test_runtime_checks_cited_audience_against_the_question(self):
        questions = (
            ("实习生能居家办公吗？", "D10", False),
            ("我是正式员工，听外包同事说能提前一天。按我的规则应提前多久？", "D11", False),
            ("我是正式员工，听外包同事说能提前一天。按我的规则应提前多久？", "D10", True),
        )
        for question, doc_id, expected in questions:
            with self.subTest(question=question, doc_id=doc_id):
                evidence = Evidence(self.knowledge, question=question)
                evidence.search({"query": "居家办公"})
                for candidate in ("D10", "D11"):
                    evidence.execute("read_doc", {"doc_id": candidate})
                result = evidence.finish({
                    "status": "answered", "citations": [{"doc_id": doc_id, "passage_id": "P1"}],
                    "gap": "", "follow_up": "",
                })
                self.assertEqual(result["ok"], expected)
        self.knowledge.documents["D10"]["passages"]["P1"] = "实习生须至少提前 3 个工作日提交远程办公申请。"
        evidence = Evidence(self.knowledge, question="实习生能居家办公吗？")
        evidence.search({"query": "居家办公"})
        for candidate in ("D10", "D11"):
            evidence.execute("read_doc", {"doc_id": candidate})
        result = evidence.finish({
            "status": "answered", "citations": [{"doc_id": "D10", "passage_id": "P1"}],
            "gap": "", "follow_up": "",
        })
        self.assertTrue(result["ok"])

    def test_explicit_multi_field_question_rejects_incomplete_citations(self):
        evidence = Evidence(
            self.knowledge, question="请同时告诉我云盘单个上传文件的大小上限，以及外部分享链接多久失效。",
        )
        evidence.execute("search_docs", {"query": "云盘"})
        evidence.execute("read_doc", {"doc_id": "D07"})
        args = {"status": "answered", "gap": "", "follow_up": ""}
        refs = [{"doc_id": "D07", "passage_id": "P2"}]
        self.assertIn("D07/P1", evidence.finish({**args, "citations": refs})["message"])
        refs.append({"doc_id": "D07", "passage_id": "P1"})
        self.assertTrue(evidence.finish({**args, "citations": refs})["ok"])

    def test_runtime_rejects_single_answer_after_both_conflicting_versions_are_read(self):
        evidence = Evidence(self.knowledge)
        evidence.execute("search_docs", {"query": "差旅报销"})
        for doc_id in ("D01", "D02"):
            evidence.execute("read_doc", {"doc_id": doc_id})
        result = evidence.finish({
            "status": "answered", "citations": [{"doc_id": "D02", "passage_id": "P1"}],
            "gap": "", "follow_up": "",
        })
        self.assertIn("同一规则存在不同数值", result["message"])

    def test_numeric_conflict_is_detected_with_an_extra_sentence(self):
        self.knowledge.documents["D02"]["passages"]["P1"] += "请保留报销凭证。"
        evidence = Evidence(self.knowledge)
        evidence.search({"query": "差旅报销"})
        for doc_id in ("D01", "D02"):
            evidence.execute("read_doc", {"doc_id": doc_id})
        result = evidence.finish({
            "status": "answered", "citations": [{"doc_id": "D02", "passage_id": "P1"}],
            "gap": "", "follow_up": "",
        })
        self.assertIn("同一规则存在不同数值", result["message"])

    def test_equal_numeric_values_accept_answer_and_reject_false_conflict(self):
        first = self.knowledge.documents["D01"]["passages"]
        first["P1"] = first["P1"].replace("30", "15.0")
        evidence = Evidence(self.knowledge)
        evidence.search({"query": "差旅报销"})
        for doc_id in ("D01", "D02"):
            evidence.execute("read_doc", {"doc_id": doc_id})
        evidence.search({"query": "差旅报销制度优先级"})
        refs = [{"doc_id": doc_id, "passage_id": "P1"} for doc_id in ("D01", "D02")]
        result = evidence.finish({
            "status": "conflict", "citations": refs, "gap": "数值不同", "follow_up": "哪份适用？",
        })
        self.assertIn("同一规则数值一致", result["message"])
        self.assertTrue(evidence.finish({
            "status": "answered", "citations": refs[:1], "gap": "", "follow_up": "",
        })["ok"])

    def test_conflict_requires_a_new_query_after_both_passages_are_read(self):
        evidence = Evidence(self.knowledge)
        evidence.execute("search_docs", {"query": "差旅报销"})
        for doc_id in ("D01", "D02"):
            evidence.execute("read_doc", {"doc_id": doc_id})
        end = {
            "status": "conflict",
            "citations": [{"doc_id": doc_id, "passage_id": "P1"} for doc_id in ("D01", "D02")],
            "gap": "未找到适用范围",
            "follow_up": "请问应采用哪份制度？",
        }
        self.assertFalse(evidence.execute("finish", end)["ok"])
        evidence.execute("search_docs", {"query": "差旅 报销"})
        self.assertFalse(evidence.execute("finish", end)["ok"])
        evidence.execute("search_docs", {"query": "木星卫星"})
        self.assertFalse(evidence.execute("finish", end)["ok"])
        check = evidence.execute("search_docs", {"query": "差旅报销 制度适用范围"})
        self.assertEqual(check["unread_doc_ids"], [])
        self.assertTrue(evidence.execute("finish", end)["ok"])

    def test_conflict_rejects_unrelated_clauses_after_authority_search(self):
        evidence = Evidence(self.knowledge, question="企业邮箱回收站保留多久？")
        evidence.search({"query": "企业邮箱 云盘"})
        for doc_id in ("D04", "D07"):
            evidence.execute("read_doc", {"doc_id": doc_id})
        evidence.search({"query": "企业邮箱 云盘 制度优先级"})
        result = evidence.finish({
            "status": "conflict",
            "citations": [{"doc_id": doc_id, "passage_id": "P2"} for doc_id in ("D04", "D07")],
            "gap": "期限不同", "follow_up": "哪份适用？",
        })
        self.assertFalse(result["ok"])
        self.assertIn("同一", result["message"])

    def test_conflict_rejects_distinct_audiences_but_allows_unmarked_versions_of_one_rule(self):
        for doc_ids, query, expected in [
            (("D10", "D11"), "远程办公", False),
            (("D01", "D02"), "差旅报销", True),
        ]:
            with self.subTest(query=query):
                for doc_id in doc_ids:
                    self.knowledge.documents[doc_id].pop("conflict_check", None)
                evidence = Evidence(self.knowledge, question=f"{query}的期限是多少？")
                evidence.search({"query": query})
                for doc_id in doc_ids:
                    evidence.execute("read_doc", {"doc_id": doc_id})
                evidence.search({"query": f"{query} 制度优先级"})
                result = evidence.finish({
                    "status": "conflict",
                    "citations": [{"doc_id": doc_id, "passage_id": "P1"} for doc_id in doc_ids],
                    "gap": "期限不同", "follow_up": "哪份适用？",
                })
                self.assertEqual(result["ok"], expected)

    def test_no_final_answer_before_remaining_candidate_pages_are_read(self):
        records = [
            {
                "doc_id": f"X{index:02}", "title": "差旅报销制度", "version_date": "2025-01-01",
                "terms": ["差旅报销"],
                "passages": {"P1": f"员工应在出差结束后 {15 if index < 6 else 30} 天内提交差旅费用报销申请。"},
                "conflict_check": {
                    "key": "reimbursement", "passage_id": "P1", "prefix": "员工应在出差结束后",
                    "suffix": "天内提交差旅费用报销申请",
                },
            }
            for index in range(1, 7)
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.json"
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            evidence = Evidence(KnowledgeBase(path), max_tool_calls=8)
        first = evidence.execute("search_docs", {"query": "差旅报销"})
        self.assertEqual(first["total"], 6)
        self.assertEqual(first["next_offset"], 5)
        for hit in first["hits"]:
            evidence.execute("read_doc", {"doc_id": hit["doc_id"]})
        answer = {
            "status": "answered", "citations": [{"doc_id": "X01", "passage_id": "P1"}],
            "gap": "", "follow_up": "",
        }
        self.assertEqual(evidence.finish(answer)["pending_searches"], [{"query": "差旅报销", "offset": 5}])
        self.assertFalse(evidence.execute("search_docs", {"query": "差旅报销", "offset": 10})["ok"])
        second = evidence.execute("search_docs", {"query": "差旅 报销", "offset": first["next_offset"]})
        self.assertEqual([hit["doc_id"] for hit in second["hits"]], ["X06"])
        self.assertIsNone(second["next_offset"])
        self.assertEqual(evidence.finish(answer)["unread_doc_ids"], ["X06"])
        evidence.execute("read_doc", {"doc_id": "X06"})
        self.assertIn("同一规则存在不同数值", evidence.finish(answer)["message"])

    def test_unread_candidates_prevent_insufficient_evidence(self):
        evidence = Evidence(self.knowledge)
        evidence.execute("search_docs", {"query": "门禁"})
        result = evidence.finish({
            "status": "insufficient_evidence", "citations": [],
            "gap": "资料未说明办理时限", "follow_up": "",
        })
        self.assertEqual(result["unread_doc_ids"], ["D03"])

    def test_narrow_search_replaces_broad_scope_and_discards_stale_candidates(self):
        for doc_id in ("D01", "D02", "D03", "D04", "D05", "D06"):
            self.knowledge.documents[doc_id]["terms"].append("制度")
        self.knowledge.term_frequency["制度"] = 6
        evidence = Evidence(self.knowledge, question="门禁开通需要几天？")
        self.assertEqual(evidence.search({"query": "制度"})["next_offset"], 5)
        evidence.execute("read_doc", {"doc_id": "D01"})
        refined = evidence.search({"query": "门禁开通"})
        self.assertEqual(refined["retired_queries"], ["制度"])
        self.assertFalse(evidence.search({"query": "制度", "offset": 5})["ok"])
        evidence.execute("read_doc", {"doc_id": "D03"})
        args = {"status": "answered", "gap": "", "follow_up": ""}
        self.assertFalse(evidence.finish({**args, "citations": [{"doc_id": "D01", "passage_id": "P1"}]})["ok"])
        self.assertTrue(evidence.finish({**args, "citations": [{"doc_id": "D03", "passage_id": "P1"}]})["ok"])

    def test_narrow_search_cannot_drop_other_question_matches(self):
        self.knowledge.documents["D01"]["terms"].append("票据")
        self.knowledge.term_frequency["票据"] = 1
        evidence = Evidence(self.knowledge, question="差旅报销期限多久？")
        evidence.search({"query": "差旅报销 云盘"})
        refined = evidence.search({"query": "票据"})
        self.assertEqual([hit["doc_id"] for hit in refined["hits"]], ["D01"])
        self.assertNotIn("retired_queries", refined)
        self.assertTrue({"D01", "D02"} <= evidence.candidates)

    def test_narrow_search_keeps_policy_versions_with_different_aliases(self):
        self.knowledge.documents["D01"]["terms"] = ["差旅报销", "差旅制度"]
        self.knowledge.documents["D02"]["terms"] = ["出差费用", "差旅制度"]
        self.knowledge.term_frequency.update({"差旅报销": 1, "出差费用": 1, "差旅制度": 2})
        evidence = Evidence(self.knowledge, question="差旅报销期限多久？")
        evidence.search({"query": "差旅制度"})
        refined = evidence.search({"query": "差旅报销"})
        self.assertEqual([hit["doc_id"] for hit in refined["hits"]], ["D01"])
        evidence.execute("read_doc", {"doc_id": "D01"})
        answer = {
            "status": "answered", "citations": [{"doc_id": "D01", "passage_id": "P1"}],
            "gap": "", "follow_up": "",
        }
        self.assertEqual(evidence.finish(answer)["unread_doc_ids"], ["D02"])
        evidence.execute("read_doc", {"doc_id": "D02"})
        self.assertIn("同一规则存在不同数值", evidence.finish(answer)["message"])

    def test_search_failure_survives_unrelated_empty_search_until_same_query_recovers(self):
        evidence = Evidence(self.knowledge, question="员工门禁多久开通？")
        self.knowledge.fail_tool = "search_docs"
        self.assertFalse(evidence.search({"query": "员工门禁"})["ok"])
        self.knowledge.fail_tool = None
        self.assertEqual(evidence.search({"query": "木星卫星"})["hits"], [])
        args = {"status": "insufficient_evidence", "citations": [], "gap": "没有对应条款", "follow_up": ""}
        blocked = evidence.finish(args)
        self.assertEqual(blocked["failed_searches"], [{"query": "员工门禁", "offset": 0}])
        self.assertTrue(evidence.finish({**args, "status": "tool_error"})["ok"])
        self.assertTrue(evidence.search({"query": "员工 门禁"})["ok"])
        self.assertEqual(evidence.finish(args)["unread_doc_ids"], ["D03"])

    def test_punctuation_only_is_rejected_as_input(self):
        with self.assertRaisesRegex(ValueError, "必须包含文字或数字"):
            load_input(SimpleNamespace(input=None, question="？"))

    def test_document_scope_applies_to_search_and_read(self):
        knowledge = KnowledgeBase(CORPUS, allowed_docs={"D04"})
        self.assertEqual([hit["doc_id"] for hit in knowledge.search_docs("邮箱 附件")["hits"]], ["D04"])
        self.assertEqual(knowledge.read_doc("D07")["error"], "doc_not_found")
        with self.assertRaisesRegex(ValueError, "未知文档"):
            KnowledgeBase(CORPUS, allowed_docs={"unknown"})

    def test_redacted_trace_omits_queries_and_read_text(self):
        result = KnowledgeAgent(MockModel(), self.knowledge).run("门禁办理多久？", [])
        trace = redact_trace(result["trace"])
        self.assertEqual([step["action"] for step in trace], ["search_docs", "read_doc", "finish"])
        self.assertEqual(trace[-1]["observation"]["status"], "answered")
        encoded = json.dumps(trace, ensure_ascii=False)
        self.assertNotIn("门禁", encoded)
        self.assertNotIn("arguments", encoded)
        self.assertNotIn("quote", encoded)
        result["trace"][0]["action"] = "门禁专用动作"
        result["trace"][0]["next_action"] = "门禁专用动作"
        self.assertNotIn("门禁", json.dumps(redact_trace(result["trace"]), ensure_ascii=False))

    def test_time_budget_stops_before_executing_a_late_model_action(self):
        model = ScriptedModel(("search_docs", {"query": "门禁"}))
        with patch("src.knowledge_qa.agent.monotonic", side_effect=[0, 0.1, 1.1]):
            result = KnowledgeAgent(model, self.knowledge, max_seconds=1).run("门禁", [])
        self.assertEqual(result["status"], "time_limit")
        self.assertEqual(result["run"]["tool_calls_used"], 0)

    def test_time_budget_returns_while_model_is_still_blocked(self):
        started, release, completed = Event(), Event(), Event()

        class BlockingModel(ScriptedModel):
            def choose(self, messages, *, deadline=None):
                started.set()
                release.wait(timeout=2)
                completed.set()
                return super().choose(messages, deadline=deadline)

        model = BlockingModel(("search_docs", {"query": "门禁"}))
        try:
            result = KnowledgeAgent(model, self.knowledge, max_seconds=0.1).run("门禁", [])
            self.assertTrue(started.is_set())
            self.assertFalse(completed.is_set())
            self.assertEqual(result["status"], "time_limit")
            self.assertEqual(result["run"]["tool_calls_used"], 0)
        finally:
            release.set()
            completed.wait(timeout=2)

    def test_transient_model_connection_is_retried_once(self):
        response = {
            "choices": [{"message": {"tool_calls": [
                {"id": "a", "function": {"name": "search_docs", "arguments": '{"query":"门禁"}'}},
                {"id": "b", "function": {"name": "read_doc", "arguments": '{"doc_id":"D03"}'}},
            ]}}],
        }
        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}),
            patch("src.knowledge_qa.models.sleep"),
            patch("src.knowledge_qa.models.urlopen", side_effect=[
                URLError("unavailable"), BytesIO(json.dumps(response).encode()),
            ]) as http,
        ):
            model = LiveModel("deepseek")
            action = model.choose([])
        self.assertEqual(http.call_count, 2)
        self.assertEqual(model.metadata()["api_retries"], 1)
        self.assertEqual(action.name, "search_docs")
        self.assertEqual(action.skipped_tool_calls, ("read_doc",))

    def test_model_authentication_errors_are_not_retried(self):
        unauthorized = HTTPError("https://api.deepseek.com", 401, "unauthorized", None, BytesIO())
        with (
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}),
            patch("src.knowledge_qa.models.urlopen", side_effect=unauthorized) as http,
        ):
            model = LiveModel("deepseek")
            with self.assertRaisesRegex(ModelError, "HTTP 401"):
                model.choose([])
        http.assert_called_once()

    def test_invalid_citation_is_observed_and_corrected(self):
        end = {"status": "answered", "gap": "", "follow_up": ""}
        model = ScriptedModel(
            ("search_docs", {"query": "门禁"}),
            ("read_doc", {"doc_id": "D03"}),
            ("finish", {**end, "citations": [{"doc_id": "D03", "passage_id": "invented"}]}),
            ("finish", {**end, "citations": [{"doc_id": "D03", "passage_id": "P1"}]}),
        )
        result = KnowledgeAgent(model, self.knowledge).run("多久开通门禁？", [])
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["quote"], self.knowledge.read_doc("D03")["passages"][0]["quote"])
        last_observation = json.loads(model.messages[-1][-1]["content"])["result"]
        self.assertEqual(last_observation["error"], "invalid_action")
        self.assertEqual(result["trace"][2]["next_action"], "finish")

    def test_search_error_remains_distinct_from_empty_results(self):
        result = KnowledgeAgent(MockModel(), KnowledgeBase(CORPUS, "search_docs")).run("门禁", [])
        self.assertEqual(result["status"], "tool_error")
        self.assertEqual(result["trace"][0]["observation"]["error"], "search_unavailable")
        self.assertEqual(result["answer"], "")

    def test_insufficient_evidence_keeps_background_out_of_answer_citations(self):
        model = ScriptedModel(
            ("search_docs", {"query": "门禁"}),
            ("read_doc", {"doc_id": "D03"}),
            ("finish", {
                "status": "insufficient_evidence",
                "citations": [{"doc_id": "D03", "passage_id": "P1"}],
                "gap": "资料未说明办理地点",
                "follow_up": "",
            }),
        )
        result = KnowledgeAgent(model, self.knowledge).run("门禁在哪里办理？", [])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["trace"][1]["observation"]["passages"][0]["passage_id"], "P1")

    def run_false_deadline_refusal(self, question, history=None):
        model = ScriptedModel(
            ("search_docs", {"query": "门禁"}),
            ("read_doc", {"doc_id": "D03"}),
            ("finish", {
                "status": "insufficient_evidence", "citations": [],
                "gap": "找不到单独的起算规定", "follow_up": "请确认起算点。",
            }),
        )
        return KnowledgeAgent(model, self.knowledge, max_tool_calls=2).run(question, history or [])

    def test_explicit_applicable_deadline_corrects_false_refusal(self):
        for question in (
            "新员工的门禁办理时限是从什么时候开始算，行政最迟几个工作日开通？",
            "试用期员工的门禁办理期限何时起算，行政最迟多少个工作日内开通门禁？",
            "新员工办门禁，从哪一步算起，行政多久能开通？",
        ):
            with self.subTest(question=question):
                result = self.run_false_deadline_refusal(question)
                self.assertEqual(result["status"], "answered")
                self.assertEqual(result["gap"], "")
                self.assertEqual(result["follow_up"], "")
                self.assertIn("提交完整的门禁权限申请后", result["answer"])
                self.assertIn("2 个工作日", result["answer"])
                self.assertIn("correction", result["trace"][-1]["observation"])
                self.assertEqual(result["run"]["tool_calls_used"], 2)

    def test_deadline_correction_does_not_infer_scope_or_counting_rules(self):
        for question in (
            "外包人员的门禁办理时限是从什么时候开始算，行政最迟几个工作日开通？",
            "新员工的门禁办理时限是从什么时候开始算，行政最迟几个工作日开通，提交当天计入吗？",
        ):
            with self.subTest(question=question):
                result = self.run_false_deadline_refusal(question)
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertEqual(result["citations"], [])

    def test_deadline_correction_uses_the_observed_source(self):
        question = "新员工的门禁办理时限是从什么时候开始算，行政最迟几个工作日开通？"
        passages = self.knowledge.documents["D03"]["passages"]
        passages["P1"] = passages["P1"].replace("2 个工作日", "4 个工作日")
        self.assertIn("4 个工作日", self.run_false_deadline_refusal(question)["answer"])
        passages["P1"] = "新员工的门禁权限申请由行政处理。"
        self.assertEqual(self.run_false_deadline_refusal(question)["status"], "insufficient_evidence")

    def test_limit_marks_unfinished_without_citations(self):
        result = KnowledgeAgent(MockModel(), self.knowledge, max_steps=1).run("门禁", [])
        self.assertEqual(result["status"], "step_limit")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["trace"][0]["next_action"], "step_limit")

    def test_tool_limit_blocks_actual_read_before_it_runs(self):
        with patch.object(self.knowledge, "read_doc", wraps=self.knowledge.read_doc) as read:
            result = KnowledgeAgent(MockModel(), self.knowledge, max_tool_calls=1).run("门禁", [])
        read.assert_not_called()
        self.assertEqual(result["status"], "tool_limit")
        self.assertEqual(result["run"]["tool_calls_used"], 1)
        self.assertEqual(result["trace"][-1]["observation"]["error"], "tool_limit")

    def test_cached_search_and_read_leave_budget_for_finishing(self):
        model = ScriptedModel(
            ("search_docs", {"query": "门禁权限"}),
            ("search_docs", {"query": "门禁 权限"}),
            ("read_doc", {"doc_id": "D03"}),
            ("read_doc", {"doc_id": "D03"}),
            ("finish", {
                "status": "answered", "citations": [{"doc_id": "D03", "passage_id": "P1"}],
                "gap": "", "follow_up": "",
            }),
        )
        with (
            patch.object(self.knowledge, "search_docs", wraps=self.knowledge.search_docs) as search,
            patch.object(self.knowledge, "read_doc", wraps=self.knowledge.read_doc) as read,
        ):
            result = KnowledgeAgent(model, self.knowledge, max_tool_calls=2).run("门禁", [])
        search.assert_called_once()
        read.assert_called_once()
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["run"]["tool_calls_used"], 2)
        self.assertEqual([step["tool_calls_remaining"] for step in result["trace"]], [1, 1, 0, 0, 0])
        self.assertTrue(result["trace"][1]["observation"]["cached"])
        self.assertTrue(result["trace"][3]["observation"]["cached"])

    def test_tool_failures_consume_budget_and_are_not_cached(self):
        evidence = Evidence(KnowledgeBase(CORPUS, "search_docs"), max_tool_calls=2)
        for remaining in (1, 0):
            result = evidence.execute("search_docs", {"query": "门禁"})
            self.assertEqual(result["error"], "search_unavailable")
            self.assertEqual(evidence.remaining, remaining)
        self.assertEqual(evidence.execute("search_docs", {"query": "门禁"})["error"], "tool_limit")

    def test_missing_live_key_reports_model_error_without_mocking(self):
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": ""}):
            result = KnowledgeAgent(LiveModel(), self.knowledge).run("门禁", [])
        self.assertEqual(result["status"], "model_error")
        self.assertEqual(result["run"]["mode"], "live")
        self.assertEqual(result["run"]["api_requests"], 0)
        self.assertEqual(result["trace"][0]["action"], "model_error")

    def test_deepseek_requires_its_own_key(self):
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "unused-test-key", "DEEPSEEK_API_KEY": ""}):
            result = KnowledgeAgent(LiveModel("deepseek"), self.knowledge).run("门禁", [])
        self.assertEqual(result["status"], "model_error")
        self.assertIn("DEEPSEEK_API_KEY", result["gap"])
        self.assertEqual(result["run"]["provider"], "deepseek")
        self.assertEqual(result["run"]["api_requests"], 0)

    def test_deepseek_replays_tool_observations_over_chat_completions(self):
        def response(step, name, arguments):
            body = {
                "model": "deepseek-flash",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"message": {"content": None, "tool_calls": [{
                    "type": "function",
                    "id": f"call-{step}",
                    "function": {"name": name, "arguments": json.dumps({"reason": "测试动作", **arguments})},
                }]}}],
            }
            return BytesIO(json.dumps(body).encode())

        env = {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_MODEL": "",
            "QA_BASE_URL": "https://example.invalid/qwen-only",
            "QA_MODEL": "qwen-only",
        }
        with patch.dict("os.environ", env), patch("src.knowledge_qa.models.urlopen") as http:
            http.side_effect = [
                response(1, "search_docs", {"query": "门禁"}),
                response(2, "read_doc", {"doc_id": "D03"}),
                response(3, "finish", {
                    "status": "answered", "citations": [{"doc_id": "D03", "passage_id": "P1"}],
                    "gap": "", "follow_up": "",
                }),
            ]
            result = KnowledgeAgent(LiveModel("deepseek"), self.knowledge).run("门禁", [])

        self.assertEqual(result["status"], "answered")
        self.assertIn("2 个工作日", result["answer"])
        self.assertEqual(result["run"]["provider"], "deepseek")
        self.assertEqual(result["run"]["api_requests"], 3)
        self.assertEqual(result["run"]["usage"]["prompt_tokens"], 30)
        requests = [call.args[0] for call in http.call_args_list]
        for request in requests:
            self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "deepseek-flash")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(payload["tool_choice"], "required")
        messages = json.loads(requests[-1].data)["messages"]
        for step, assistant, observation in ((1, messages[2], messages[3]), (2, messages[4], messages[5])):
            self.assertEqual(assistant["tool_calls"][0]["id"], f"call-{step}")
            self.assertEqual(observation["role"], "tool")
            self.assertEqual(observation["tool_call_id"], f"call-{step}")
            self.assertTrue(json.loads(observation["content"])["result"]["ok"])
        self.assertEqual(json.loads(messages[-1]["content"])["result"]["doc_id"], "D03")


if __name__ == "__main__":
    unittest.main()
