from __future__ import annotations

from difflib import SequenceMatcher
import json
import re
from typing import Any

from .chunking import TextChunk
from .graph_service import RunControl, _chat_json


EVALUATION_CONTEXT_LIMIT = 30_000


def _normalized_question(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def questions_are_similar(first: str, second: str) -> bool:
    left = _normalized_question(first)
    right = _normalized_question(second)
    if not left or not right:
        return False
    if left == right:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.96


def _evaluation_context(chunks: list[TextChunk]) -> tuple[str, set[int]]:
    if not chunks:
        return "", set()
    max_samples = max(1, EVALUATION_CONTEXT_LIMIT // 500)
    if len(chunks) <= max_samples:
        sampled = chunks
    else:
        sampled = [
            chunks[round(index * (len(chunks) - 1) / (max_samples - 1))]
            for index in range(max_samples)
        ]
    per_chunk = max(200, EVALUATION_CONTEXT_LIMIT // len(sampled) - 80)
    context = "\n\n".join(
        f"[CHUNK {chunk.number}; PAGES {','.join(map(str, chunk.pages))}]\n{chunk.text[:per_chunk]}"
        for chunk in sampled
    )
    return context, {chunk.number for chunk in sampled}


def generate_document_summary(
    base_url: str,
    api_key: str,
    model: str,
    chunks: list[TextChunk],
    control: RunControl | None = None,
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("無法為空文件建立摘要")
    context, _ = _evaluation_context(chunks)
    document = chunks[0].document or "未命名文件"

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise ValueError("文件摘要不得為空")
        primary_identifier = str(payload.get("primary_identifier", "")).strip()
        normalized: dict[str, Any] = {
            "document": document, "summary": summary,
            "primary_identifier": primary_identifier,
        }
        for field in ("identifiers", "topics", "keywords"):
            values = payload.get(field, [])
            if not isinstance(values, list):
                raise ValueError(f"{field} 必須是陣列")
            normalized[field] = list(dict.fromkeys(
                str(value).strip() for value in values if str(value).strip()
            ))
        if primary_identifier:
            normalized["identifiers"] = list(dict.fromkeys(
                [primary_identifier, *normalized["identifiers"]]
            ))[:5]
        else:
            normalized["identifiers"] = normalized["identifiers"][:5]
        return normalized

    return _chat_json(
        base_url,
        api_key,
        model,
        "你是文件分類與路由摘要助手。只能根據文件內容整理摘要，並只輸出 JSON。",
        "請建立供後續問題路由使用的繁體中文短摘要。文件識別資訊只可包含文件主要描述對象的正式名稱、型號、系統名稱、規範名稱、研究對象或其他唯一名稱。"
        "primary_identifier 請優先使用能涵蓋整份文件的共同名稱或系列名稱。identifiers 最多列 5 項。"
        "排除出版商、作者、版權所有者、文件編號、作業系統、支援平台、一般欄位名稱、版本欄位、通用術語，以及僅在內文附帶提及的名稱。"
        "判斷標準是：把名稱放入問題後，是否能讓讀者辨認該問題適用的主要文件對象；若不能就不得收錄。找不到時 primary_identifier 使用空字串且 identifiers 回傳空陣列。"
        "摘要應能區分內容相似但來源不同的文件。輸出格式："
        '{"summary":"...","primary_identifier":"...","identifiers":["..."],"topics":["..."],"keywords":["..."]}。\n\n'
        f"文件名稱：{document}\n文件內容：\n{context}",
        temperature=0,
        validator=validate,
        control=control,
    )


def select_relevant_documents(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    document_summaries: list[dict[str, Any]],
    max_documents: int = 3,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("文件路由問題不可為空")
    available = {
        str(item.get("document", "")).strip()
        for item in document_summaries
        if str(item.get("document", "")).strip()
    }
    if not available:
        raise ValueError("請先建立 PDF 路由摘要")
    limit = min(max(int(max_documents), 1), len(available))

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        raw_documents = payload.get("documents")
        reason = str(payload.get("reason", "")).strip()
        if not isinstance(raw_documents, list) or not raw_documents:
            raise ValueError("documents 必須是非空陣列")
        documents = list(dict.fromkeys(
            str(document).strip() for document in raw_documents
            if str(document).strip()
        ))
        unknown = [document for document in documents if document not in available]
        if unknown:
            raise ValueError(f"文件路由包含不存在的文件：{unknown}")
        if not 1 <= len(documents) <= limit:
            raise ValueError(f"文件路由必須選擇 1 到 {limit} 份文件")
        if not reason:
            raise ValueError("文件路由必須包含 reason")
        try:
            confidence = float(payload.get("confidence", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence 必須是數字") from exc
        return {
            "documents": documents,
            "reason": reason,
            "confidence": min(max(confidence, 0.0), 1.0),
        }

    summaries = [
        {
            "document": item.get("document", ""),
            "summary": item.get("summary", ""),
            "primary_identifier": item.get("primary_identifier", ""),
            "identifiers": item.get("identifiers", item.get("product_names", [])),
            "topics": item.get("topics", []),
            "keywords": item.get("keywords", []),
        }
        for item in document_summaries
    ]
    return _chat_json(
        base_url,
        api_key,
        model,
        "你是多文件檢索路由器。只能根據問題與文件摘要選擇應搜尋的文件，並只輸出 JSON。",
        f"問題：{question.strip()}\n最多選擇 {limit} 份文件。"
        "問題指向單一文件識別資訊時只選該文件；需要跨文件比較時才能選多份。"
        "不得回傳清單以外的文件。輸出格式："
        '{"documents":["..."],"reason":"...","confidence":0.0}。\n\n'
        f"文件摘要：\n{json.dumps(summaries, ensure_ascii=False)}",
        temperature=0,
        validator=validate,
    )




def generate_evaluation_questions(
    base_url: str,
    api_key: str,
    model: str,
    chunks: list[TextChunk],
    question_count: int,
    excluded_questions: list[str] | None = None,
    document_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    count = int(question_count)
    if not chunks:
        raise ValueError("請先解析 PDF 並產生 chunks")
    if not 1 <= count <= 100:
        raise ValueError("題目數量必須介於 1 到 100")

    excluded = [
        str(question).strip() for question in (excluded_questions or [])
        if str(question).strip()
    ]
    summary = document_summary or {}
    primary_identifier = str(summary.get("primary_identifier", "")).strip()
    listed_identifiers = list(dict.fromkeys(
        str(value).strip()
        for value in summary.get("identifiers", summary.get("product_names", []))
        if str(value).strip()
    ))
    identifiers = [primary_identifier] if primary_identifier else listed_identifiers[:1]
    if not identifiers:
        identifiers = [chunks[0].document] if chunks[0].document else []

    context, available_chunk_numbers = _evaluation_context(chunks)
    chunk_lookup = {chunk.number: chunk for chunk in chunks}

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        questions = payload.get("questions")
        if not isinstance(questions, list) or len(questions) != count:
            raise ValueError(f"questions 必須剛好包含 {count} 題")
        normalized = []
        accepted_questions = list(excluded)
        for index, item in enumerate(questions, start=1):
            if not isinstance(item, dict):
                raise ValueError("每一題必須是 JSON 物件")
            question = str(item.get("question", "")).strip()
            answer = str(item.get("expected_answer", "")).strip()
            if not question or not answer:
                raise ValueError("每一題都必須包含 question 與 expected_answer")
            if identifiers and not any(identifier.casefold() in question.casefold() for identifier in identifiers):
                raise ValueError("每一題都必須包含至少一項文件識別資訊")
            if any(questions_are_similar(question, existing) for existing in accepted_questions):
                raise ValueError(f"題目與既有題目重複或過度相似：{question}")
            accepted_questions.append(question)
            pages = item.get("source_pages", [])
            if not isinstance(pages, list):
                raise ValueError("source_pages 必須是陣列")
            raw_numbers = item.get("source_chunk_numbers", [])
            if not isinstance(raw_numbers, list):
                raise ValueError("source_chunk_numbers 必須是陣列")
            chunk_numbers: list[int] = []
            for value in raw_numbers:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number in available_chunk_numbers and number not in chunk_numbers:
                    chunk_numbers.append(number)
            if not chunk_numbers:
                raise ValueError("每一題都必須包含至少一個有效的 source_chunk_numbers")
            document = "、".join(
                dict.fromkeys(
                    chunk_lookup[number].document
                    for number in chunk_numbers
                    if chunk_lookup[number].document
                )
            )
            normalized.append({
                "number": index,
                "question": question,
                "expected_answer": answer,
                "source_pages": [int(page) for page in pages],
                "source_chunk_numbers": chunk_numbers,
                "document": document,
            })
        return {"questions": normalized}

    exclusion_instruction = ""
    if excluded:
        exclusion_instruction = (
            "\n不得重複或改寫以下已建立題目：\n- " + "\n- ".join(excluded) + "\n"
        )

    result = _chat_json(
        base_url,
        api_key,
        model,
        "你是文件問答評測資料設計師。只能根據提供的文件內容出題，並只輸出 JSON。",
        f"請建立剛好 {count} 道可由文件明確回答、彼此不重複且涵蓋不同內容的繁體中文問題。"
        f"每一題都必須自然包含至少一項文件識別資訊：{json.dumps(identifiers, ensure_ascii=False)}。"
        "識別資訊用來讓讀者不看答案也能判斷問題所屬文件，不可只寫成無法區分來源的通用問題。"
        f"{exclusion_instruction}"
        "每題提供精確標準答案、來源頁碼，以及該題所依據的 CHUNK 編號"
        "（source_chunk_numbers，必須引用下方文件中標示的 CHUNK 編號）。輸出格式："
        '{"questions":[{"question":"...","expected_answer":"...","source_pages":[1],'
        '"source_chunk_numbers":[1]}]}。\n\n'
        f"文件：\n{context}",
        temperature=0.2,
        validator=validate,
    )
    return result["questions"]


_ABSTENTION_PHRASES = (
    "無法回答", "無法判斷", "沒有任何資訊", "找不到相關資訊",
    "文件未提及", "證據未提及", "未提供相關", "證據不足",
    "cannot answer", "not enough information", "insufficient evidence",
)


def _is_abstention(answer: str) -> bool:
    normalized = " ".join(answer.casefold().split())
    return any(phrase in normalized for phrase in _ABSTENTION_PHRASES)


def judge_evaluation_answer(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> dict[str, Any]:
    if not actual_answer.strip():
        return {"passed": False, "reason": "實際答案為空，未回答標準答案中的關鍵事實。"}
    if _is_abstention(actual_answer) and not _is_abstention(expected_answer):
        return {
            "passed": False,
            "reason": "實際答案表示無法回答或資料不足，但標準答案包含明確事實。",
        }

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        passed = payload.get("passed")
        reason = str(payload.get("reason", "")).strip()
        if not isinstance(passed, bool) or not reason:
            raise ValueError("評判結果必須包含 passed boolean 與 reason")
        return {"passed": passed, "reason": reason}

    return _chat_json(
        base_url,
        api_key,
        model,
        "你是嚴謹的問答評測員。只有實際答案包含標準答案的核心事實才能通過。"
        "誠實表示不知道、文件未提及、找不到資訊或證據不足，不等於回答正確；"
        "當標準答案有明確事實而實際答案拒答時，passed 必須為 false。"
        "不要求逐字相同，只輸出 JSON。",
        f"問題：{question}\n標準答案：{expected_answer}\n實際答案：{actual_answer}\n"
        "請逐項檢查標準答案中的數值、單位、名稱、條件與結論是否出現在實際答案。"
        '輸出格式：{"passed":true,"reason":"簡短理由"}。',
        validator=validate,
    )
