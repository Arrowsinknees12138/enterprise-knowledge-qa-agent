import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from src.knowledge_qa.agent import Evidence, KnowledgeAgent
from src.knowledge_qa.knowledge import KnowledgeBase
from src.knowledge_qa.models import Action, LiveModel, MockModel


CORPUS = Path(__file__).resolve().parents[1] / "data" / "knowledge.json"


class ScriptedModel:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.messages = []

    def choose(self, messages):
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
        check = evidence.execute("search_docs", {"query": "差旅报销 制度适用范围"})
        self.assertEqual(check["unread_doc_ids"], [])
        self.assertTrue(evidence.execute("finish", end)["ok"])

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

    def test_limit_marks_unfinished_without_citations(self):
        result = KnowledgeAgent(MockModel(), self.knowledge, max_steps=1).run("门禁", [])
        self.assertEqual(result["status"], "step_limit")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["trace"][0]["next_action"], "step_limit")

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
