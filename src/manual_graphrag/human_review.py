from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime
from typing import Any
from uuid import uuid4


def create_human_review(
    results: list[dict[str, Any]], sample_per_stratum: int,
    reviewers: list[str], *, created_by: str | None = None, seed: int | None = None,
) -> dict[str, Any]:
    """Sample results evenly across experiment group/project/question-set strata."""
    if sample_per_stratum < 1:
        raise ValueError("每個實驗組／成員專案／題目集的抽樣數必須至少為 1")
    reviewer_list = list(dict.fromkeys(str(item).strip() for item in reviewers if str(item).strip()))
    if not reviewer_list:
        raise ValueError("沒有可分派的評測使用者")
    if not results:
        raise ValueError("沒有已保存的 AI 評測結果可供抽樣")

    strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, result in enumerate(results):
        key = (
            str(result.get("group_name") or result.get("group_index") or "未命名實驗組"),
            str(result.get("source_project_id") or result.get("source_project_name") or "未知專案"),
            str(result.get("question_set_id") or result.get("question_set_name") or "未分類題目集"),
        )
        strata[key].append(index)

    if seed is None:
        seed = random.SystemRandom().randrange(1 << 63)
    rng = random.Random(seed)
    selected: list[int] = []
    for indices in strata.values():
        selected.extend(rng.sample(indices, min(sample_per_stratum, len(indices))))
    rng.shuffle(selected)

    tasks = [
        {
            "task_id": uuid4().hex,
            "result_index": result_index,
            "assigned_to": reviewer_list[index % len(reviewer_list)],
            "score": None,
            "note": "",
            "submitted_at": None,
        }
        for index, result_index in enumerate(selected)
    ]
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "created_by": created_by or "未選擇使用者",
        "sample_per_stratum": sample_per_stratum,
        "stratum_count": len(strata),
        "seed": seed,
        "tasks": tasks,
    }


def human_review_complete(review: dict[str, Any] | None) -> bool:
    tasks = (review or {}).get("tasks") or []
    return bool(tasks) and all(task.get("score") in (0, 1, 2) for task in tasks)


def human_review_progress(review: dict[str, Any] | None) -> tuple[int, int]:
    tasks = (review or {}).get("tasks") or []
    completed = sum(task.get("score") in (0, 1, 2) for task in tasks)
    return completed, len(tasks)
