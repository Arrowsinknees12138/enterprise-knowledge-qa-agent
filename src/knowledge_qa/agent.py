import json
from time import monotonic

from .knowledge import normalize
from .models import STATUSES, SYSTEM_PROMPT, ModelError, TimeLimitError


def rejection(message):
    return {"ok": False, "error": "invalid_action", "message": message}


def short_text(value, max_length):
    return isinstance(value, str) and len(value) <= max_length and bool(value.strip())


class KnowledgeAgent:
    def __init__(self, model, knowledge, max_steps=8, max_tool_calls=6, max_seconds=120):
        if max_steps < 1:
            raise ValueError("max_steps 必须大于 0")
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls 不能小于 0")
        if max_seconds <= 0:
            raise ValueError("max_seconds 必须大于 0")
        self.model = model
        self.knowledge = knowledge
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_seconds = max_seconds

    def run(self, question, history):
        deadline = monotonic() + self.max_seconds
        time_gap = f"已用完 {self.max_seconds} 秒运行预算，问答未完成。"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"question": question, "history": history}, ensure_ascii=False)},
        ]
        evidence = Evidence(self.knowledge, self.max_tool_calls, question=question, has_history=bool(history))
        trace = []
        for step in range(1, self.max_steps + 1):
            try:
                if monotonic() >= deadline:
                    raise TimeLimitError
                action = self.model.choose(messages, deadline=deadline)
                if monotonic() >= deadline:
                    raise TimeLimitError
            except TimeLimitError:
                return self.result("time_limit", trace, evidence, gap=time_gap)
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
            if action.skipped_tool_calls:
                observation = {**observation, "skipped_tool_calls": list(action.skipped_tool_calls)}
            trace.append({
                "step": step,
                "action": action.name,
                "arguments": arguments,
                "reason": reason,
                "observation": observation,
                "tool_calls_remaining": evidence.remaining,
            })
            if monotonic() >= deadline:
                return self.result("time_limit", trace, evidence, gap=time_gap)
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
            trace[-1]["next_action"] = status if status in ("step_limit", "tool_limit", "time_limit") else None
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
                "max_seconds": self.max_seconds,
            },
            "trace": trace,
        }


class Evidence:
    def __init__(self, knowledge, max_tool_calls=6, question="", has_history=False):
        self.knowledge = knowledge
        self.max_tool_calls = max_tool_calls
        self.question = question
        self.has_history = has_history
        self.tool_calls_used = 0
        self.search_cache = {}
        self.pending_pages = {}
        self.candidates = set()
        self.reads = {}
        self.failed_reads = set()
        self.search_history = []
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
            return self.search(arguments)
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

    def search(self, arguments):
        query = arguments.get("query")
        offset = arguments.get("offset", 0)
        if not arguments.keys() <= {"query", "offset"} or not short_text(query, 200) or not normalize(query):
            return rejection("query 必须是非空且不超过 200 字的字符串")
        if type(offset) is not int or offset < 0:
            return rejection("offset 必须是非负整数")
        normalized_query = normalize(query)
        key = normalized_query, offset
        cached = key in self.search_cache
        if cached:
            result = self.search_cache[key]
        else:
            if offset and self.pending_pages.get(normalized_query, (None, None))[1] != offset:
                return rejection("请使用同一查询和最近一次返回的 next_offset 继续检索")
            if not self.remaining:
                return self.tool_limit()
            self.tool_calls_used += 1
            result = self.knowledge.search_docs(query, offset)
            self.search_failed = not result["ok"]
            if not result["ok"]:
                return result
            self.search_cache[key] = result
            self.candidates.update(hit["doc_id"] for hit in result["hits"])
            if not offset:
                self.search_history.append((normalized_query, set(self.reads)))
            if result["next_offset"] is not None:
                self.pending_pages[normalized_query] = (query, result["next_offset"])
            else:
                self.pending_pages.pop(normalized_query, None)
        next_page = self.pending_pages.get(normalized_query)
        return {
            **result, "query": query, "cached": cached,
            "next_offset": next_page[1] if next_page else None,
            "unread_doc_ids": [hit["doc_id"] for hit in result["hits"] if hit["doc_id"] not in self.reads],
        }

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
        if status in ("answered", "insufficient_evidence", "conflict"):
            if self.search_failed or self.failed_reads:
                return rejection("存在未恢复的业务工具错误")
            unread = sorted(self.candidates - self.reads.keys())
            if unread or self.pending_pages:
                if not self.remaining:
                    return self.tool_limit()
                return {
                    **rejection("先读取未核对的候选并继续检索剩余页面，再提交结论"),
                    "unread_doc_ids": unread,
                    "pending_searches": [
                        {"query": query, "offset": offset} for query, offset in self.pending_pages.values()
                    ],
                }
        if status == "insufficient_evidence":
            if not self.search_cache:
                return rejection("资料不足需要成功完成至少一次检索")
            # Keep background passages in the trace, separate from answer citations.
            citations = []
            if self.question and not self.has_history and len(self.reads) == 1:
                observed_doc = next(iter(self.reads.values()))
                citation = self.knowledge.explicit_deadline_citation(self.question, observed_doc)
                if citation:
                    status = "answered"
                    citations = [citation]
                    gap = ""

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
        if status == "answered":
            if not refs:
                return rejection("回答至少需要一个有效引用")
            wrong_topics = self.knowledge.wrong_passage_topics(self.question, refs)
            if wrong_topics:
                return rejection(f"以下片段与问题所问字段不符：{', '.join(wrong_topics)}")
            for conflict in self.knowledge.conflicting_refs(self.reads):
                if conflict.intersection(refs):
                    pairs = ", ".join(f"{doc_id}/{passage_id}" for doc_id, passage_id in sorted(conflict))
                    return rejection(f"已核对的同一规则存在不同数值：{pairs}。请查证适用依据后报告冲突。")
        if status == "conflict":
            documents = {doc_id for doc_id, _ in refs}
            if len(documents) < 2:
                return rejection("冲突状态至少需要引用两份不同文档")
            if not any(
                documents <= reads and self.knowledge.is_authority_query(query, documents)
                for query, reads in self.search_history
            ):
                return rejection("请在读完冲突原文后，针对该制度补查适用范围、优先级、废止依据或通用制度效力")
            gap = "资料中缺少足以判定适用口径的依据，需要确认适用范围、优先级或废止关系。"
        result = {"ok": True, "status": status, "citations": list(refs.values()), "gap": gap}
        if status != arguments["status"]:
            result["follow_up"] = ""
            result["correction"] = "已读且适用的原文明示起算事件和期限"
        return result
