from __future__ import annotations

import random
from collections import defaultdict
from datetime import datetime
from typing import Any
from uuid import uuid4


def create_human_review(
    results: list[dict[str, Any]], sample_per_stratum: int,
    reviewers: list[str], *, sampling_mode: str = "fixed_per_stratum",
    created_by: str | None = None, seed: int | None = None,
) -> dict[str, Any]:
    """Sample results by a fixed count per stratum or a proportional total."""
    if sample_per_stratum < 1:
        raise ValueError("抽樣數必須至少為 1")
    reviewer_list = list(dict.fromkeys(str(item).strip() for item in reviewers if str(item).strip()))
    if not reviewer_list:
        raise ValueError("沒有可分派的評測使用者")
    if not results:
        raise ValueError("沒有已保存的 AI 評測結果可供抽樣")
    if sampling_mode not in {"fixed_per_stratum", "proportional_total"}:
        raise ValueError("抽樣方式無效")

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
    allocations: dict[tuple[str, str, str], int] = {}
    if sampling_mode == "fixed_per_stratum":
        allocations = {
            key: min(sample_per_stratum, len(indices))
            for key, indices in strata.items()
        }
    else:
        total_sample_count = min(sample_per_stratum, len(results))
        population = len(results)
        remainders: list[tuple[float, int, tuple[str, str, str]]] = []
        allocated = 0
        for order, (key, indices) in enumerate(strata.items()):
            exact = total_sample_count * len(indices) / population
            base = int(exact)
            allocations[key] = base
            allocated += base
            remainders.append((exact - base, order, key))
        for _remainder, _order, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:total_sample_count - allocated]:
            allocations[key] += 1

    for key, indices in strata.items():
        selected.extend(rng.sample(indices, allocations[key]))
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
        "sample_count": sample_per_stratum,
        "sampling_mode": sampling_mode,
        "requested_sample_count": sample_per_stratum,
        "sampled_count": len(selected),
        "stratum_sample_counts": [
            {
                "group": key[0], "project": key[1], "question_set": key[2],
                "available": len(indices), "sampled": allocations[key],
            }
            for key, indices in strata.items()
        ],
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
