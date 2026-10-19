"""Split the existing MuSiQue accuracy samples into separate hop-count sets."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SAMPLES = (
    (Path("sample"), "questions_accuracy_100.json", "100-question sample"),
    (
        Path("sample_40docs"),
        "questions_accuracy_40docs.json",
        "40-document sample",
    ),
)


def split_sample(directory: Path, source_name: str, description: str) -> None:
    source_path = ROOT / directory / source_name
    source = json.loads(source_path.read_text(encoding="utf-8"))
    questions = source["questions"]

    for hop_count in (2, 3, 4):
        selected = [
            {**question, "number": number}
            for number, question in enumerate(
                (item for item in questions if int(item["hops"]) == hop_count),
                start=1,
            )
        ]
        output = {
            "schema_version": source["schema_version"],
            "dataset": (
                f"MuSiQue-Ans dev — {hop_count}-hop accuracy sample "
                f"({description})"
            ),
            "questions": selected,
        }
        output_path = ROOT / directory / f"questions_accuracy_{hop_count}hop.json"
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{output_path.relative_to(ROOT)}: {len(selected)} questions")


if __name__ == "__main__":
    for sample_directory, source_file, label in SAMPLES:
        split_sample(sample_directory, source_file, label)
