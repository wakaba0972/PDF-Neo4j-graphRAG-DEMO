import pytest

from manual_graphrag.human_review import (
    create_human_review,
    human_review_complete,
    human_review_progress,
)


def test_sample_is_randomized_per_group_project_and_question_set_and_evenly_assigned():
    results = [
        {
            "group_name": group,
            "source_project_id": project,
            "question_set_id": question_set,
            "number": number,
        }
        for group, project, question_set, size in [
            ("A", "p1", "q1", 8),
            ("A", "p1", "q2", 4),
            ("B", "p2", "q1", 5),
        ]
        for number in range(size)
    ]

    review = create_human_review(
        results, 2, ["Jay", "Christine", "Tai"], created_by="Zhao", seed=42,
    )

    indices = [task["result_index"] for task in review["tasks"]]
    assert len(indices) == 6
    assert len(set(indices)) == 6
    assert review["stratum_count"] == 3
    assert {task["assigned_to"] for task in review["tasks"]} == {"Jay", "Christine", "Tai"}
    assignments = [task["assigned_to"] for task in review["tasks"]]
    assert max(assignments.count(name) for name in set(assignments)) - min(
        assignments.count(name) for name in set(assignments)
    ) <= 1
    assert review["created_by"] == "Zhao"
    assert human_review_progress(review) == (0, 6)
    assert not human_review_complete(review)


def test_sample_caps_each_stratum_and_rejects_invalid_inputs():
    results = [{"group_name": "A"}, {"group_name": "A"}, {"group_name": "B"}]
    review = create_human_review(results, 10, ["Jay", "Christine"], seed=1)
    assert len(review["tasks"]) == 3
    assert human_review_complete({"tasks": [{"score": 0}, {"score": 1}, {"score": 2}]})
    assert human_review_progress({"tasks": [{"score": 0}, {"score": None}]}) == (1, 2)

    with pytest.raises(ValueError, match="至少為 1"):
        create_human_review(results, 0, ["Jay"])
    with pytest.raises(ValueError, match="沒有可分派"):
        create_human_review(results, 1, [])
    with pytest.raises(ValueError, match="沒有已保存"):
        create_human_review([], 1, ["Jay"])
