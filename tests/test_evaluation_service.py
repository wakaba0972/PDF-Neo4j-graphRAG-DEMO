import pytest

from manual_graphrag.chunking import TextChunk
from manual_graphrag import evaluation_service


def test_generate_evaluation_questions_validates_and_numbers(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "CX17 系列如何查詢網路 IP？", "expected_answer": "從網路設定頁查看 IP", "source_pages": [2],
                 "source_chunk_numbers": [1]},
                {"question": "CX17 系列出現 E01 時如何復歸？", "expected_answer": "長按重設鍵", "source_pages": [3],
                 "source_chunk_numbers": [1]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    questions = evaluation_service.generate_evaluation_questions(
        "http://models", "key", "model", [TextChunk(1, "文件", (2, 3), "manual.pdf")], 2, [],
    )

    assert [item["number"] for item in questions] == [1, 2]
    assert questions[0]["source_pages"] == [2]
    assert questions[0]["question_source_pages"] == [2]
    assert questions[0]["answer_source_pages"] == [2]
    assert questions[0]["document"] == "manual.pdf"


def test_generate_evaluation_questions_keeps_question_and_answer_pages_separate(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({"questions": [{
            "question": "第二頁提出的問題？",
            "expected_answer": "答案在第三、四頁說明。",
            "question_source_pages": [2],
            "answer_source_pages": [3, 4],
            "source_chunk_numbers": [7],
        }]})

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    questions = evaluation_service.generate_evaluation_questions(
        "url", "", "model", [TextChunk(7, "跨頁內容", (2, 3, 4), "manual.pdf")], 1,
    )

    assert questions[0]["question_source_pages"] == [2]
    assert questions[0]["answer_source_pages"] == [3, 4]
    assert questions[0]["source_pages"] == [3, 4]


def test_generate_evaluation_questions_rejects_similar_questions(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [{
                "question": "系統的最大容量是多少？",
                "expected_answer": "100",
                "source_pages": [1],
                "source_chunk_numbers": [1],
            }]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)

    with pytest.raises(ValueError, match="重複或過度相似"):
        evaluation_service.generate_evaluation_questions(
            "url",
            "",
            "model",
            [TextChunk(1, "容量為 100", (1,), "manual.pdf")],
            1,
            ["系統的最大容量是多少"],
        )


def test_question_similarity_catches_chinese_rewording_without_flagging_other_topics() -> None:
    assert evaluation_service.questions_are_similar(
        "馬達異音時如何檢查？", "馬達出現異常聲音要怎麼檢查？"
    )
    assert not evaluation_service.questions_are_similar(
        "如何查詢目前 IP 位址？", "如何重設網路設定？"
    )


def test_question_dedup_rejects_same_answer_across_chunks(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({"questions": [
            {"question": "列印品質不佳時如何清潔噴頭？", "expected_answer": "依照步驟清潔噴頭",
             "source_pages": [1], "source_chunk_numbers": [8]},
            {"question": "如何改善印出來的字跡模糊？", "expected_answer": "依照步驟清潔噴頭",
             "source_pages": [1], "source_chunk_numbers": [1]},
        ]})

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    with pytest.raises(ValueError, match="重複或過度相似"):
        evaluation_service.generate_evaluation_questions(
            "url", "", "model", [
                TextChunk(1, "清潔噴頭", (1,), "manual.pdf"),
                TextChunk(8, "列印品質", (1,), "manual.pdf"),
            ], 2
        )


def test_random_page_context_selects_only_focus_page_chunks(monkeypatch) -> None:
    chunks = [
        TextChunk(number, f"chunk-{number}", (number,), "manual.pdf")
        for number in range(1, 6)
    ]
    monkeypatch.setattr(evaluation_service.random, "choice", lambda pages: 3)

    focus_page, context = evaluation_service.sample_random_page_context(chunks)

    assert focus_page == 3
    assert [chunk.number for chunk in context] == [3]


def test_random_page_context_selects_focus_page_at_document_boundaries(monkeypatch) -> None:
    chunks = [
        TextChunk(number, f"chunk-{number}", (number,), "manual.pdf")
        for number in range(1, 4)
    ]
    monkeypatch.setattr(evaluation_service.random, "choice", lambda pages: 1)

    focus_page, context = evaluation_service.sample_random_page_context(chunks)

    assert focus_page == 1
    assert [chunk.number for chunk in context] == [1]


def test_page_context_expands_until_llm_reports_complete(monkeypatch) -> None:
    assessments = iter([False, False, True])
    checked_contexts = []

    def fake_chat(*args, **kwargs):
        checked_contexts.append(args[4])
        return kwargs["validator"]({"sufficient": next(assessments), "reason": "需更多上下文"})

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    chunks = [
        TextChunk(number, f"page-{number}", (number,), "manual.pdf")
        for number in range(1, 6)
    ]

    context = evaluation_service.expand_page_context_until_complete(
        "url", "key", "model", chunks, 3,
    )

    assert [chunk.number for chunk in context] == [1, 2, 3, 4, 5]
    assert len(checked_contexts) == 3
    assert "PAGES 3" in checked_contexts[0]
    assert "PAGES 2" in checked_contexts[1] and "PAGES 4" in checked_contexts[1]


def test_generate_evaluation_questions_tells_model_which_questions_to_avoid(
    monkeypatch,
) -> None:
    captured = {}

    def fake_chat(*args, **kwargs):
        captured["prompt"] = args[4]
        return kwargs["validator"]({
            "questions": [{
                "question": "另一個問題？",
                "expected_answer": "答案",
                "source_pages": [1],
                "source_chunk_numbers": [1],
            }]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    evaluation_service.generate_evaluation_questions(
        "url", "", "model", [TextChunk(1, "文件", (1,))], 1, ["既有問題？"]
    )

    assert "既有問題？" in captured["prompt"]

    assert "不得重複或改寫以下已建立題目" in captured["prompt"]
def test_generate_evaluation_questions_rejects_missing_chunk_numbers(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "manual.pdf 的問題一？", "expected_answer": "答案一", "source_pages": [2]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    with pytest.raises(ValueError, match="source_chunk_numbers"):
        evaluation_service.generate_evaluation_questions(
            "http://models", "key", "model", [TextChunk(1, "文件", (2,), "manual.pdf")], 1
        )


def test_generate_evaluation_questions_joins_multiple_source_documents(monkeypatch) -> None:
    def fake_chat(*args, **kwargs):
        return kwargs["validator"]({
            "questions": [
                {"question": "a.pdf 的問題一？", "expected_answer": "答案一", "source_pages": [1],
                 "source_chunk_numbers": [1, 2]},
            ]
        })

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    questions = evaluation_service.generate_evaluation_questions(
        "http://models", "key", "model",
        [TextChunk(1, "文件一", (1,), "a.pdf", "id-a"), TextChunk(2, "文件二", (1,), "b.pdf", "id-b")], 1
    )

    assert questions[0]["document"] == "a.pdf、b.pdf"
    assert questions[0]["answer_sources"] == [
        {"document_id": "id-a", "document_name": "a.pdf", "pages": [1]},
        {"document_id": "id-b", "document_name": "b.pdf", "pages": [1]},
    ]


def test_generate_evaluation_questions_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="先解析"):
        evaluation_service.generate_evaluation_questions("url", "", "model", [], 2)
    with pytest.raises(ValueError, match="1 到 100"):
        evaluation_service.generate_evaluation_questions(
            "url", "", "model", [TextChunk(1, "text", (1,))], 0
        )


def test_judge_evaluation_answer_returns_boolean(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "_chat_json",
        lambda *args, **kwargs: kwargs["validator"]({"passed": True, "reason": "語意相符"}),
    )

    result = evaluation_service.judge_evaluation_answer(
        "url", "key", "model", "問題", "標準", "實際"
    )

    assert result == {"passed": True, "reason": "語意相符"}


def test_verify_evaluation_judgment_reviews_first_pass_and_reason(monkeypatch) -> None:
    captured = {}

    def fake_chat(base_url, api_key, model, system_prompt, user_prompt, **kwargs):
        captured.update(
            base_url=base_url, api_key=api_key, model=model,
            system_prompt=system_prompt, user_prompt=user_prompt,
        )
        return kwargs["validator"]({"passed": False, "reason": "標準答案中的數值遺漏"})

    monkeypatch.setattr(evaluation_service, "_chat_json", fake_chat)
    result = evaluation_service.verify_evaluation_judgment(
        "url", "key", "judge", "問題", "標準答案", "實際答案",
        True, "第一輪認為正確", reasoning_effort="low",
    )

    assert result == {"passed": False, "reason": "標準答案中的數值遺漏"}
    assert "第一輪判定：正確" in captured["user_prompt"]
    assert "第一輪理由：第一輪認為正確" in captured["user_prompt"]
    assert "實際答案：實際答案" in captured["user_prompt"]


def test_judge_rejects_abstention_when_expected_answer_has_fact(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "_chat_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("明確拒答應由程式直接判錯，不應交給模型")
        ),
    )

    result = evaluation_service.judge_evaluation_answer(
        "url",
        "key",
        "model",
        "Epson AcuLaser CX17 的中央輸出承接盤可容納多少張 A4 紙？",
        "約 100 張 (A4)",
        "根據提供的證據內容，沒有任何資訊提及容量。因此，無法回答此問題。",
    )

    assert result["passed"] is False
    assert "標準答案包含明確事實" in result["reason"]


def test_judge_rejects_empty_answer_without_model_call(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation_service,
        "_chat_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不應呼叫模型")),
    )

    result = evaluation_service.judge_evaluation_answer(
        "url", "key", "model", "問題", "明確答案", "  "
    )

    assert result["passed"] is False
    assert "實際答案為空" in result["reason"]
