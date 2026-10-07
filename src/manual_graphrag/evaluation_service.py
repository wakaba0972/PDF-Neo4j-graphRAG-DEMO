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
            "你是文件脈絡完整性檢查員。判斷提供的頁面內容是否足以獨立理解一個明確事實，"
            "並能據此提出答案不含糊的問答題。只輸出 JSON。",
            f"焦點頁：第 {focus_page} 頁。檢查目前提供的內容是否已包含足夠主詞、條件、步驟與結論，"
            "使一個可由內容明確回答的問題不依賴缺失的前文或後文。若內容已完整，sufficient=true；"
            "若句子被截斷、指代不明、步驟/條件/結論延續到未提供頁面，或資訊不足以形成明確問答，則為 false。"
            '輸出格式：{"sufficient":true,"reason":"簡短說明"}。\n\n'
            f"目前已提供頁面 {pages[lower]} 至 {pages[upper]}：\n{context}",
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
            "\n不得重複或改寫以下已建立題目：\n- " + "\n- ".join(excluded_text) + "\n"
        )
    focus_instruction = (
        f"本次隨機抽樣的焦點頁是第 {focus_page} 頁；請優先從該頁出題，必要時使用前後相鄰 chunk 補足脈絡。"
        if focus_page is not None else ""
    )

    result = _chat_json(
        base_url,
        api_key,
        model,
        "你是文件問答評測資料設計師。只能根據提供的文件內容出題，並只輸出 JSON。",
        f"請建立剛好 {count} 道可由文件明確回答、彼此不重複且涵蓋不同知識點的繁體中文問題。"
        "不同措辭若詢問相同事實、操作步驟或預期答案，仍視為重複；每題必須測試不同資訊，不可只替換同義詞、語序或問句模板。"
        f"{focus_instruction}"
        f"{exclusion_instruction}"
        "每題提供精確標準答案、題目來源頁碼（question_source_pages）、答案來源頁碼（answer_source_pages），"
        "並提供該題所依據的 CHUNK 編號"
        "（source_chunk_numbers，必須引用下方文件中標示的 CHUNK 編號）。輸出格式："
        '{"questions":[{"question":"...","expected_answer":"...",'
        '"question_source_pages":[1],"answer_source_pages":[1],'
        '"source_chunk_numbers":[1]}]}。\n\n'
        f"文件：\n{context}",
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
        "你是嚴謹的問答評測員，請依答案涵蓋標準答案核心內容的程度給 0 到 2 分。"
        "0 代表錯誤或未回答核心事實；1 代表部分正確但遺漏重要內容或條件；"
        "2 代表完整正確，涵蓋標準答案的必要事實。"
        "誠實表示不知道、文件未提及、找不到資訊或證據不足，不能得分；"
        "不要求逐字相同，只輸出 JSON。",
        f"問題：{question}\n標準答案：{expected_answer}\n實際答案：{actual_answer}\n"
        "請逐項檢查標準答案中的數值、單位、名稱、條件與結論是否出現在實際答案。"
        '輸出格式：{"score":2,"reason":"簡短理由"}。',
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
        "你是獨立的第二輪問答評測複核員。重新核對題目、標準答案和實際答案，"
        "並檢查第一輪 0 到 2 分的判定及理由是否成立。不要因為第一輪已給結論就照單全收；"
        "0=錯誤，1=部分正確，2=完整正確。拒答不能視為正確。只輸出 JSON。",
        f"問題：{question}\n標準答案：{expected_answer}\n實際答案：{actual_answer}\n"
        f"第一輪判定分數：{first_result}\n第一輪理由：{first_reason}\n"
        "請重新獨立檢查；輸出最終 0 到 2 分及簡短理由，格式為 "
        '{"score":2,"reason":"複核後理由"}。',
        validator=validate,
        reasoning_effort=reasoning_effort,
    )
