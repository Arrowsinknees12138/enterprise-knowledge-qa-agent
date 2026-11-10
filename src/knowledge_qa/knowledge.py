import json
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path


def normalize(text):
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in folded if char.isalnum())


class KnowledgeBase:
    def __init__(self, path: Path, fail_tool: str | None = None):
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
            self.documents[doc_id] = doc
            self.term_frequency.update({normalize(term) for term in terms})

    @staticmethod
    def metadata(doc):
        return {key: doc[key] for key in ("doc_id", "title", "version_date")}

    def search_docs(self, query):
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
        return {"ok": True, "total": len(matches), "hits": matches[:5]}

    def read_doc(self, doc_id):
        if self.fail_tool == "read_doc":
            return {"ok": False, "error": "read_unavailable", "message": "模拟文档读取服务不可用"}
        doc = self.documents.get(doc_id)
        if doc is None:
            return {"ok": False, "error": "doc_not_found", "message": "文档不存在"}
        return {
            "ok": True,
            **self.metadata(doc),
            "passages": [
                {"passage_id": passage_id, "quote": quote}
                for passage_id, quote in doc["passages"].items()
            ],
        }
