import json

from .knowledge import normalize
from .models import STATUSES, SYSTEM_PROMPT, ModelError


def rejection(message):
    return {"ok": False, "error": "invalid_action", "message": message}


def short_text(value, max_length):
    return isinstance(value, str) and len(value) <= max_length and bool(value.strip())


class KnowledgeAgent:
    def __init__(self, model, knowledge, max_steps=8, max_tool_calls=6):
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls 不能小于 0")
        self.model = model
        self.knowledge = knowledge
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls

    def run(self, question, history):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"question": question, "history": history}, ensure_ascii=False)},
        ]
        evidence = Evidence(self.knowledge, self.max_tool_calls, question=question if not history else "")
        trace = []
        for step in range(1, self.max_steps + 1):
            try:
                action = self.model.choose(messages)
            except ModelError as error:
                trace.append({
                    "step": step,
                    "action": "model_error",
                    "arguments": {},
                    "reason": "未取得可执行的模型动作",
                    "observation": {"ok": False, "error": "model_error", "message": str(error)},
                    "tool_calls_remaining": evidence.remaining,
                })
                return self.result("model_error", trace, evidence, gap=str(error))

            arguments = action.arguments.copy()
            reason = arguments.pop("reason", None)
            if short_text(reason, 160):
                observation = evidence.execute(action.name, arguments)
            else:
                reason = ""
                observation = rejection("reason 必须是一句不超过 160 字的可公开说明")
            trace.append({
                "step": step,
                "action": action.name,
                "arguments": arguments,
                "reason": reason,
                "observation": observation,
                "tool_calls_remaining": evidence.remaining,
            })
            if observation.get("error") == "tool_limit":
                return self.result("tool_limit", trace, evidence, gap=observation["message"])
            if action.name == "finish" and observation["ok"]:
                return self.result(
                    observation["status"], trace, evidence, observation["citations"],
                    observation["gap"], observation.get("follow_up", arguments["follow_up"]),
                )

            messages.extend([
                action.assistant_message(),
                {
                    "role": "tool",
                    "tool_call_id": action.call_id,
                    "content": json.dumps(
                        {"action": action.name, "result": observation, "tool_calls_remaining": evidence.remaining},
                        ensure_ascii=False,
                    ),
                },
            ])
        return self.result("step_limit", trace, evidence, gap=f"已用完 {self.max_steps} 步，问答未完成。")

    def result(self, status, trace, evidence, citations=None, gap="", follow_up=""):
        citations = citations or []
        for current, following in zip(trace, trace[1:]):
            current["next_action"] = following["action"]
        if trace:
            trace[-1]["next_action"] = status if status in ("step_limit", "tool_limit") else None
        answer = ""
        if status in ("answered", "conflict", "needs_clarification") and citations:
            excerpts = [f'【{ref["doc_id"]}/{ref["passage_id"]}】{ref["quote"]}' for ref in citations]
            if status == "conflict":
                excerpts.insert(0, "找到相互冲突的原文，适用口径待确认。")
            answer = "\n".join(excerpts)
        return {
            "status": status,
            "answer": answer,
            "citations": citations,
            "gap": gap,
            "follow_up": follow_up,
            "run": {
                **self.model.metadata(), "max_steps": self.max_steps, "steps_used": len(trace),
                "max_tool_calls": evidence.max_tool_calls, "tool_calls_used": evidence.tool_calls_used,
            },
            "trace": trace,
        }


class Evidence:
    def __init__(self, knowledge, max_tool_calls=6, question=""):
        self.knowledge = knowledge
        self.max_tool_calls = max_tool_calls
        self.question = question
        self.tool_calls_used = 0
        self.search_cache = {}
        self.candidates = set()
        self.reads = {}
        self.failed_reads = set()
        self.search_queries = set()
        self.reads_before_new_query = set()
        self.search_failed = False

    @property
    def remaining(self):
        return self.max_tool_calls - self.tool_calls_used

    def tool_limit(self):
        return {
            "ok": False,
            "error": "tool_limit",
            "message": f"已用完 {self.max_tool_calls} 次实际工具调用，当前证据不足以完成问答。",
        }

    def execute(self, name, arguments):
        if name == "search_docs":
            query = arguments.get("query")
            if arguments.keys() != {"query"} or not short_text(query, 200) or not normalize(query):
                return rejection("query 必须是非空且不超过 200 字的字符串")
            normalized_query = normalize(query)
            cached = normalized_query in self.search_cache
            if cached:
                result = self.search_cache[normalized_query]
            else:
                if not self.remaining:
                    return self.tool_limit()
                self.tool_calls_used += 1
                result = self.knowledge.search_docs(query)
            self.search_failed = not result["ok"]
            if not result["ok"]:
                return result
            if not cached:
                self.search_cache[normalized_query] = result
                self.reads_before_new_query.update(self.reads)
                self.search_queries.add(normalized_query)
                self.candidates.update(hit["doc_id"] for hit in result["hits"])
            return {
                **result, "cached": cached,
                "unread_doc_ids": [hit["doc_id"] for hit in result["hits"] if hit["doc_id"] not in self.reads],
            }
        if name == "read_doc":
            doc_id = arguments.get("doc_id")
            if arguments.keys() != {"doc_id"} or not isinstance(doc_id, str) or doc_id not in self.candidates:
                return rejection("doc_id 必须来自本轮的检索结果")
            if doc_id in self.reads:
                return {**self.reads[doc_id], "cached": True}
            if not self.remaining:
                return self.tool_limit()
            self.tool_calls_used += 1
            result = self.knowledge.read_doc(doc_id)
            if result["ok"]:
                self.failed_reads.discard(doc_id)
                self.reads[doc_id] = result
            else:
                self.failed_reads.add(doc_id)
            return {**result, "cached": False}
        if name == "finish":
            return self.finish(arguments)
        return rejection("未知动作")

    def finish(self, arguments):
        status = arguments.get("status")
        citations = arguments.get("citations")
        gap, follow_up = arguments.get("gap"), arguments.get("follow_up")
        if (
            arguments.keys() != {"status", "citations", "gap", "follow_up"}
            or status not in STATUSES
            or not isinstance(citations, list)
            or any(not isinstance(value, str) or len(value) > 300 for value in (gap, follow_up))
        ):
            return rejection("结束动作的字段或状态无效")
        if status != "answered" and not gap.strip():
            return rejection("未解决的任务必须说明具体缺口")
        if status in ("conflict", "needs_clarification") and not follow_up.strip():
            return rejection("有待确认项时必须提出一个澄清问题")
        if status == "tool_error" and not (self.search_failed or self.failed_reads):
            return rejection("没有未恢复的业务工具错误")
        if status in ("insufficient_evidence", "needs_clarification") and (self.search_failed or self.failed_reads):
            return rejection("存在未恢复的业务工具错误，应先尝试恢复或明确返回 tool_error")
        if status == "insufficient_evidence":
            if not self.search_queries:
                return rejection("资料不足需要成功完成至少一次检索")
            if not self.remaining and self.candidates - self.reads.keys():
                return self.tool_limit()
            # Keep background passages in the trace, separate from answer citations.
            citations = []
            if self.question and len(self.reads) == 1 and self.candidates == self.reads.keys():
                observed_doc = next(iter(self.reads.values()))
                citation = self.knowledge.explicit_deadline_citation(self.question, observed_doc)
                if citation:
                    status = "answered"
                    citations = [citation]
                    gap = ""
        if status == "answered" and (self.search_failed or self.candidates - self.reads.keys()):
            return rejection("存在未核对的候选文档或未恢复的检索失败")

        refs = {}
        for citation in citations:
            if not isinstance(citation, dict) or citation.keys() != {"doc_id", "passage_id"}:
                return rejection("引用必须包含 doc_id 和 passage_id")
            doc_id, passage_id = citation["doc_id"], citation["passage_id"]
            if not isinstance(doc_id, str) or not isinstance(passage_id, str):
                return rejection("引用 ID 必须为字符串")
            doc = self.reads.get(doc_id)
            if doc is None:
                return rejection(f"尚未读取引用文档 {doc_id}")
            passage = next((item for item in doc["passages"] if item["passage_id"] == passage_id), None)
            if passage is None:
                return rejection(f"未读取引用片段 {doc_id}/{passage_id}")
            refs[doc_id, passage_id] = {
                "doc_id": doc_id, "title": doc["title"], "version_date": doc["version_date"], **passage
            }
        if status == "answered" and not refs:
            return rejection("回答至少需要一个有效引用")
        if status == "conflict":
            documents = {doc_id for doc_id, _ in refs}
            if len(documents) < 2:
                return rejection("冲突状态至少需要引用两份不同文档")
            if not documents <= self.reads_before_new_query:
                return rejection("请在读完冲突原文后，使用一条新查询补查适用范围、优先级或废止依据")
            gap = "资料中缺少足以判定适用口径的依据，需要确认适用范围、优先级或废止关系。"
        result = {"ok": True, "status": status, "citations": list(refs.values()), "gap": gap}
        if status != arguments["status"]:
            result["follow_up"] = ""
            result["correction"] = "已读且适用的原文明示起算事件和期限"
        return result
