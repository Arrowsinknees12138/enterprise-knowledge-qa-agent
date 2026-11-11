import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


SEARCH_PAGE_SIZE = 5
AUTHORITY_TERMS = ("适用", "优先", "废止", "替代", "生效", "效力", "层级", "口径", "现行", "适用对象")
GENERAL_AUTHORITY_QUERIES = ("制度效力", "制度优先", "制度层级", "政策优先", "规章效力", "政策冲突处理")
MULTI_FIELD_REQUESTS = ("分别", "同时告诉", "请同时", "两项都", "两个期限都", "两个规则都", "两种链接")


# Match a self-contained question that asks only for a start event and a deadline.
DEADLINE_QUESTION = re.compile(
    r"(?P<audience>[\u4e00-\u9fff]{1,16}?)(?:的|办理|办|申请)"
    r"(?P<topic>[\u4e00-\u9fff]{2,24}?)(?:(?:办理|处理|开通)?(?:时限|期限)(?:是)?|[,，]\s*)"
    r"(?:从(?:什么时候|何时|哪一步|哪个环节)(?:开始)?(?:计算|起算|算起|算)|何时起算)[,，]\s*"
    r"(?P<responsible>[\u4e00-\u9fff]{1,12}?)"
    r"(?:(?:最迟|最多|至多|应在|需要|要|需)(?:几个|多少个|多少)\s*工作日(?:内)?|(?:要|需)?多久(?:能|可以)?)"
    r"(?P<action>[\u4e00-\u9fff]{1,12})[。？?]?"
)


def normalize(text):
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in folded if char.isalnum())


def matching_passage_ids(topics, normalized_question):
    return {
        passage_id for passage_id, terms in topics.items()
        if any(normalize(term) in normalized_question for term in terms)
    }


class KnowledgeBase:
    def __init__(self, path: Path, fail_tool: str | None = None, allowed_docs: set[str] | None = None):
        with path.open(encoding="utf-8-sig") as file:
            records = json.load(file)
        if not isinstance(records, list) or not records:
            raise ValueError("知识库必须是非空的文档数组")
        self.documents = {}
        self.term_frequency = Counter()
        self.fail_tool = fail_tool
        for doc in records:
            if not isinstance(doc, dict):
                raise ValueError("文档必须是对象")
            for key in ("doc_id", "title", "version_date"):
                if not isinstance(doc.get(key), str) or not doc[key].strip():
                    raise ValueError(f"文档字段 {key} 必须是非空字符串")
            date.fromisoformat(doc["version_date"])
            doc_id = doc["doc_id"]
            if allowed_docs is not None and doc_id not in allowed_docs:
                continue
            if doc_id in self.documents:
                raise ValueError(f"重复文档 ID {doc_id}")
            terms, passages = doc.get("terms"), doc.get("passages")
            if not isinstance(terms, list) or not terms or any(
                not isinstance(term, str) or not normalize(term) for term in terms
            ):
                raise ValueError(f"{doc_id} 的 terms 必须是非空的检索词数组")
            if not isinstance(passages, dict) or not passages or any(
                not key.strip() or not isinstance(value, str) or not value.strip()
                for key, value in passages.items()
            ):
                raise ValueError(f"{doc_id} 的 passages 必须包含有效的片段 ID 和原文")
            topics = doc.get("passage_topics", {})
            if not isinstance(topics, dict) or any(
                key not in passages or not isinstance(aliases, list) or not aliases
                or any(not isinstance(alias, str) or not normalize(alias) for alias in aliases)
                for key, aliases in topics.items()
            ):
                raise ValueError(f"{doc_id} 的 passage_topics 必须把现有片段映射到非空别名列表")
            if "conflict_check" in doc:
                rule = doc["conflict_check"]
                fields = {"key", "passage_id", "prefix", "suffix"}
                if (
                    not isinstance(rule, dict) or rule.keys() != fields
                    or any(not isinstance(rule[key], str) or not normalize(rule[key]) for key in fields)
                    or rule["passage_id"] not in passages
                ):
                    raise ValueError(f"{doc_id} 的 conflict_check 必须指向现有片段和数值前后的原文")
            self.documents[doc_id] = doc
            self.term_frequency.update({normalize(term) for term in terms})
        if allowed_docs is not None and (unknown := allowed_docs - self.documents.keys()):
            raise ValueError(f"允许列表包含未知文档：{', '.join(sorted(unknown))}")

    @staticmethod
    def metadata(doc):
        return {key: doc[key] for key in ("doc_id", "title", "version_date")}

    def search_docs(self, query, offset=0):
        if self.fail_tool == "search_docs":
            return {"ok": False, "error": "search_unavailable", "message": "模拟检索服务不可用"}
        query = normalize(query)
        matches = []
        for doc in self.documents.values():
            matched = {normalize(term): term for term in doc["terms"] if normalize(term) in query}
            if matched:
                score = sum(1 / self.term_frequency[term] for term in matched)
                matches.append({**self.metadata(doc), "matched_terms": list(matched.values()), "score": round(score, 6)})
        matches.sort(key=lambda doc: (-doc["score"], doc["doc_id"]))
        end = offset + SEARCH_PAGE_SIZE
        return {
            "ok": True, "total": len(matches), "hits": matches[offset:end],
            "next_offset": end if end < len(matches) else None,
        }

    def read_doc(self, doc_id):
        if self.fail_tool == "read_doc":
            return {"ok": False, "error": "read_unavailable", "message": "模拟文档读取服务不可用"}
        doc = self.documents.get(doc_id)
        if doc is None:
            return {"ok": False, "error": "doc_not_found", "message": "文档不存在"}
        return {
            "ok": True,
            **self.metadata(doc),
            "passage_topics": doc.get("passage_topics", {}),
            "passages": [
                {"passage_id": passage_id, "quote": quote}
                for passage_id, quote in doc["passages"].items()
            ],
        }

    def citation_topic_error(self, question, refs, reads):
        query = normalize(question)
        matching = {
            doc_id: matching_passage_ids(self.documents[doc_id].get("passage_topics", {}), query)
            for doc_id in reads
        }
        wrong = [
            f"{doc_id}/{passage_id}" for doc_id, passage_id in refs
            if matching[doc_id] and passage_id not in matching[doc_id]
        ]
        if wrong:
            return f"以下片段与问题所问字段不符：{', '.join(wrong)}"
        if any(marker in query for marker in MULTI_FIELD_REQUESTS):
            expected = {(doc_id, passage_id) for doc_id, ids in matching.items() for passage_id in ids}
            missing = [f"{doc_id}/{passage_id}" for doc_id, passage_id in sorted(expected - refs.keys())]
            if missing:
                return f"问题明确询问多项内容，请补充缺少的原文片段：{', '.join(missing)}"
        return ""

    def conflicting_refs(self, reads):
        groups = defaultdict(dict)
        for doc_id, observed in reads.items():
            rule = self.documents[doc_id].get("conflict_check")
            if not rule:
                continue
            for passage in observed["passages"]:
                if passage["passage_id"] != rule["passage_id"]:
                    continue
                pattern = rf"{re.escape(normalize(rule['prefix']))}([0-9]+){re.escape(normalize(rule['suffix']))}"
                match = re.fullmatch(pattern, normalize(passage["quote"]))
                if match:
                    groups[rule["key"]][doc_id, passage["passage_id"]] = int(match[1])
        return [
            set(values) for values in groups.values()
            if len({doc_id for doc_id, _ in values}) > 1 and len(set(values.values())) > 1
        ]

    def is_authority_query(self, query, doc_ids):
        if any(term in query for term in GENERAL_AUTHORITY_QUERIES):
            return True
        return any(term in query for term in AUTHORITY_TERMS) and any(
            normalize(term) in query
            for doc_id in doc_ids
            for term in self.documents[doc_id]["terms"]
        )

    def explicit_deadline_citation(self, question, observed_doc):
        request = DEADLINE_QUESTION.fullmatch(unicodedata.normalize("NFKC", question.strip()))
        passages = observed_doc["passages"]
        if not request or len(passages) != 1:
            return None

        doc_id = observed_doc["doc_id"]
        doc = self.documents[doc_id]
        passage = passages[0]
        rule = doc.get("deadline")
        if not isinstance(rule, dict) or rule.get("passage_id") != passage["passage_id"]:
            return None
        audiences = rule.get("audiences")
        trigger, responsible, outcome = (rule.get(field) for field in ("trigger", "responsible", "outcome"))
        if not isinstance(audiences, list) or not all(
            isinstance(field, str) and field for field in (trigger, responsible, outcome)
        ):
            return None

        topic = request["topic"]
        if (
            request["audience"] not in audiences
            or normalize(topic) not in {normalize(term) for term in doc["terms"]}
            or request["responsible"] != responsible
            or outcome not in {request["action"], request["action"] + topic}
            or not trigger.endswith("后")
        ):
            return None

        # Revalidate the rule against the actual read. Edited source text may invalidate its annotation.
        clause = (
            rf"{re.escape(trigger)}[,，]\s*{re.escape(responsible)}(?:会|将|应)?在\s*"
            rf"[0-9]+\s*个?工作日内{re.escape(outcome)}[。.]?"
        )
        quote = unicodedata.normalize("NFKC", passage["quote"].strip())
        if not re.fullmatch(clause, quote):
            return None
        return {"doc_id": doc_id, "passage_id": passage["passage_id"]}
