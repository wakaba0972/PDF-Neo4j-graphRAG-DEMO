from __future__ import annotations

from difflib import SequenceMatcher
import random
import re
from typing import Any

from .chunking import TextChunk
from .graph_service import _chat_json


EVALUATION_CONTEXT_LIMIT = 30_000


def _normalized_question(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


_QUESTION_FILLERS = (
    "請問", "请问", "請說明", "请说明", "請問一下", "请问一下",
    "如何", "怎麼", "怎么", "怎樣", "怎样", "為什麼", "为什么",
    "有什麼", "有什么", "哪些", "哪一些", "哪一種", "哪种", "哪個", "哪个",
    "是否", "能否", "可否", "可以", "能夠", "能够", "需要", "請", "请",
    "請介紹", "请介绍", "說明", "说明", "介紹", "介绍", "告訴我", "告诉我",
    "是什麼", "是什么", "為何", "为何", "嗎", "吗", "呢", "如何處理", "如何处理",
)


def _question_features(value: str) -> tuple[str, set[str]]:
    normalized = _normalized_question(value)
    content = normalized
    for filler in sorted(_QUESTION_FILLERS, key=len, reverse=True):
        content = content.replace(_normalized_question(filler), "")
    grams = {
        content[index:index + 2]
        for index in range(max(0, len(content) - 1))
    }
    words = set(re.findall(r"[a-z0-9]+", content))
    return content, grams | words


def _similarity_score(first: str, second: str) -> float:
    left, left_features = _question_features(first)
    right, right_features = _question_features(second)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if not left_features or not right_features:
        return SequenceMatcher(None, left, right).ratio()
    overlap = len(left_features & right_features)
    dice = 2 * overlap / (len(left_features) + len(right_features))
    containment = overlap / min(len(left_features), len(right_features))
    sequence = SequenceMatcher(None, left, right).ratio()
    return max(dice, containment * 0.82, sequence)


def questions_are_similar(first: str, second: str) -> bool:
    return _similarity_score(first, second) >= 0.64


def _question_value(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("question", "")).strip()
    return str(item).strip()


def evaluation_questions_are_similar(first: dict[str, Any], second: Any) -> bool:
    second_question = _question_value(second)
    score = _similarity_score(str(first.get("question", "")), second_question)
    if score >= 0.64:
        return True
    if not isinstance(second, dict):
        return False
    first_chunks = {str(value) for value in first.get("source_chunk_numbers", [])}
    second_chunks = {str(value) for value in second.get("source_chunk_numbers", [])}
    same_source = bool(first_chunks & second_chunks)
    first_answer = _normalized_question(str(first.get("expected_answer", "")))
    second_answer = _normalized_question(str(second.get("expected_answer", "")))
    answer_similarity = (
        SequenceMatcher(None, first_answer, second_answer).ratio()
        if first_answer and second_answer else 0.0
    )
    same_detailed_answer = (
        min(len(first_answer), len(second_answer)) >= 8 and answer_similarity >= 0.96
    )
    return same_detailed_answer or (
        score >= 0.40 and answer_similarity >= 0.90
    )


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


def sample_random_page_context(
    chunks: list[TextChunk],
) -> tuple[int, list[TextChunk]]:
    """Choose a focus page and return only chunks that contain that page."""
    ordered = sorted(chunks, key=lambda chunk: chunk.number)
    pages = sorted({page for chunk in ordered for page in chunk.pages})
    if not ordered or not pages:
        raise ValueError("PDF chunks 沒有可供隨機抽樣的來源頁碼")
    focus_page = random.choice(pages)
    return focus_page, [chunk for chunk in ordered if focus_page in chunk.pages]


def expand_page_context_until_complete(
    base_url: str,
    api_key: str,
    model: str,
    chunks: list[TextChunk],
    focus_page: int,
    reasoning_effort: str | None = None,
) -> list[TextChunk]:
    """Expand a focus page by adjacent pages until an LLM judges it self-contained."""
    ordered = sorted(chunks, key=lambda chunk: chunk.number)
    pages = sorted({page for chunk in ordered for page in chunk.pages})
    if focus_page not in pages:
        raise ValueError(f"找不到焦點頁 {focus_page} 的文件內容")

    lower = upper = pages.index(focus_page)
    while True:
        selected_pages = set(pages[lower:upper + 1])
        context_chunks = [chunk for chunk in ordered if selected_pages.intersection(chunk.pages)]
        context, _ = _evaluation_context(context_chunks)

        def validate(payload: dict[str, Any]) -> dict[str, Any]:
            sufficient = payload.get("sufficient")
            reason = str(payload.get("reason", "")).strip()
            if not isinstance(sufficient, bool):
                raise ValueError("內容完整性檢查必須回傳 sufficient boolean")
            return {"sufficient": sufficient, "reason": reason}

        assessment = _chat_json(
            base_url,
            api_key,
            model,
            "You are a document-context completeness reviewer. Determine whether the supplied page content "
            "is self-contained enough to understand a clear fact and formulate an unambiguous question-answer pair. "
            "Return JSON only.",
            f"Focus page: {focus_page}. Check whether the supplied content includes enough subjects, conditions, "
            "steps, and conclusions for a question that can be answered without missing preceding or following context. "
            "Set sufficient=true if the content is complete. Set it to false if a sentence is cut off, a reference is "
            "unclear, a step/condition/conclusion continues on an unavailable page, or the information is insufficient "
            'for an unambiguous question-answer pair. Output format: {"sufficient":true,"reason":"brief explanation"}.\n\n'
            f"Currently supplied pages {pages[lower]} through {pages[upper]}:\n{context}",
            temperature=0,
            validator=validate,
            reasoning_effort=reasoning_effort,
        )
        if assessment["sufficient"] or (lower == 0 and upper == len(pages) - 1):
            return context_chunks
        if lower > 0:
            lower -= 1
        if upper < len(pages) - 1:
            upper += 1


def generate_evaluation_questions(
    base_url: str,
    api_key: str,
    model: str,
    chunks: list[TextChunk],
    question_count: int,
    excluded_questions: list[Any] | None = None,
    focus_page: int | None = None,
    reasoning_effort: str | None = None,
) -> list[dict[str, Any]]:
    count = int(question_count)
    if not chunks:
        raise ValueError("請先解析 PDF 並產生 chunks")
    if not 1 <= count <= 100:
        raise ValueError("題目數量必須介於 1 到 100")

    excluded = [
        question for question in (excluded_questions or [])
        if _question_value(question)
    ]
    context, available_chunk_numbers = _evaluation_context(chunks)
    chunk_lookup = {chunk.number: chunk for chunk in chunks}
    available_pages = {page for chunk in chunks for page in chunk.pages}

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
            raw_question_pages = item.get(
                "question_source_pages", item.get("source_pages", [])
            )
            raw_answer_pages = item.get(
                "answer_source_pages", item.get("source_pages", [])
            )
            if not isinstance(raw_question_pages, list):
                raise ValueError("question_source_pages 必須是陣列")
            if not isinstance(raw_answer_pages, list):
                raise ValueError("answer_source_pages 必須是陣列")
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
            def valid_pages(values: list[Any]) -> list[int]:
                result: list[int] = []
                for value in values:
                    try:
                        page = int(value)
                    except (TypeError, ValueError):
                        continue
                    if page in available_pages and page not in result:
                        result.append(page)
                return result

            cited_chunk_pages = list(dict.fromkeys(
                    page for number in chunk_numbers for page in chunk_lookup[number].pages
                ))
            answer_pages = valid_pages(raw_answer_pages) or cited_chunk_pages
            question_pages = valid_pages(raw_question_pages) or answer_pages
            def source_references(pages: list[int]) -> list[dict[str, Any]]:
                references: list[dict[str, Any]] = []
                for number in chunk_numbers:
                    chunk = chunk_lookup[number]
                    selected_pages = [page for page in chunk.pages if page in pages]
                    if not selected_pages:
                        continue
                    key = (chunk.document_id, chunk.document)
                    reference = next((
                        value for value in references
                        if (value["document_id"], value["document_name"]) == key
                    ), None)
                    if reference is None:
                        reference = {
                            "document_id": chunk.document_id,
                            "document_name": chunk.document,
                            "pages": [],
                        }
                        references.append(reference)
                    reference["pages"] = list(dict.fromkeys([*reference["pages"], *selected_pages]))
                return references

            question_sources = source_references(question_pages)
            answer_sources = source_references(answer_pages)
            candidate = {
                "question": question,
                "expected_answer": answer,
                "source_chunk_numbers": chunk_numbers,
            }
            if any(
                evaluation_questions_are_similar(candidate, existing)
                for existing in accepted_questions
            ):
                raise ValueError(f"題目與既有題目重複或過度相似：{question}")
            accepted_questions.append(candidate)
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
                "question_source_pages": question_pages,
                "answer_source_pages": answer_pages,
                "source_pages": answer_pages,
                "question_sources": question_sources,
                "answer_sources": answer_sources,
                "source_chunk_numbers": chunk_numbers,
                "document": document,
            })
        return {"questions": normalized}

    exclusion_instruction = ""
    if excluded:
        excluded_text = [_question_value(item) for item in excluded]
        exclusion_instruction = (
            "\nDo not duplicate or paraphrase any of these existing questions:\n- "
            + "\n- ".join(excluded_text) + "\n"
        )
    focus_instruction = (
        f"The randomly selected focus page for this generation attempt is page {focus_page}. "
        "Prioritize a question grounded in that page; use adjacent chunks only when needed to provide sufficient context. "
        if focus_page is not None else ""
    )

    result = _chat_json(
        base_url,
        api_key,
        model,
        "You are a document-grounded QA dataset designer. Create questions only from the supplied document content. "
        "Return JSON only.",
        f"Generate exactly {count} questions that are clearly answerable from the document, are not duplicates, "
        "and cover distinct facts. Write each question and expected answer in the same language as the source content. "
        "Questions that ask about the same fact, procedure, or expected answer are duplicates even if phrased differently. "
        "Each question must test distinct information; do not create variants by changing only synonyms, word order, or a question template. "
        f"{focus_instruction}"
        f"{exclusion_instruction}"
        "For each item, provide a precise expected answer, the question source pages (question_source_pages), "
        "the answer source pages (answer_source_pages), and the supporting chunk numbers (source_chunk_numbers). "
        "Use only chunk numbers shown in the supplied document text. Output format: "
        '{"questions":[{"question":"...","expected_answer":"...",'
        '"question_source_pages":[1],"answer_source_pages":[1],'
        '"source_chunk_numbers":[1]}]}\n\n'
        f"Document text:\n{context}",
        temperature=0.2,
        validator=validate,
        reasoning_effort=reasoning_effort,
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
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    if not actual_answer.strip():
        return {"score": 0, "reason": "實際答案為空，未回答標準答案中的關鍵事實。"}
    if _is_abstention(actual_answer) and not _is_abstention(expected_answer):
        return {
            "score": 0,
            "reason": "實際答案表示無法回答或資料不足，但標準答案包含明確事實。",
        }

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        score = payload.get("score")
        reason = str(payload.get("reason", "")).strip()
        if not isinstance(score, int) or isinstance(score, bool) or score not in {0, 1, 2} or not reason:
            raise ValueError("評判結果必須包含 0、1、2 的 score 與 reason")
        return {"score": score, "reason": reason}

    return _chat_json(
        base_url,
        api_key,
        model,
        "You are a rigorous question-answer evaluator. Assign a score from 0 to 2 based on how completely "
        "the response covers the essential content of the expected answer. Score 0 for an incorrect response "
        "or one that omits the key fact; score 1 for a partially correct response that misses important details "
        "or conditions; score 2 for a fully correct response that covers the required facts. An honest statement "
        "of uncertainty, missing information, or insufficient evidence must not receive credit. Exact wording is "
        "not required. Return JSON only.",
        f"Question: {question}\nExpected answer: {expected_answer}\nActual answer: {actual_answer}\n"
        "Check whether the response covers each relevant value, unit, name, condition, and conclusion in the expected answer. "
        'Output format: {"score":2,"reason":"brief rationale"}.',
        validator=validate,
        reasoning_effort=reasoning_effort,
    )


def verify_evaluation_judgment(
    base_url: str,
    api_key: str,
    model: str,
    question: str,
    expected_answer: str,
    actual_answer: str,
    first_score: int,
    first_reason: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Independently review a first-pass answer judgment and its rationale."""
    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        score = payload.get("score")
        reason = str(payload.get("reason", "")).strip()
        if not isinstance(score, int) or isinstance(score, bool) or score not in {0, 1, 2} or not reason:
            raise ValueError("複核結果必須包含 0、1、2 的 score 與 reason")
        return {"score": score, "reason": reason}

    first_result = first_score
    return _chat_json(
        base_url,
        api_key,
        model,
        "You are an independent second-pass reviewer for a question-answer evaluation. Recheck the question, "
        "expected answer, and actual answer, then verify whether the first-pass score (0–2) and rationale are justified. "
        "Do not accept the first-pass conclusion without review. Score 0 means incorrect, 1 means partially correct, "
        "and 2 means fully correct. An abstention is not correct. Return JSON only.",
        f"Question: {question}\nExpected answer: {expected_answer}\nActual answer: {actual_answer}\n"
        f"First-pass score: {first_result}\nFirst-pass rationale: {first_reason}\n"
        "Review the answer independently and return the final score from 0 to 2 with a brief rationale. "
        'Output format: {"score":2,"reason":"rationale after review"}.',
        validator=validate,
        reasoning_effort=reasoning_effort,
    )
