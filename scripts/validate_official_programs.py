"""Validate the safe executor against every public FinQA gold program."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from finqa_rag.data.finqa import load_split
from finqa_rag.evaluation.numerical_metrics import answers_match
from finqa_rag.reasoning.program import ProgramError, execute_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/finqa"))
    parser.add_argument("--max-failures", type=int, default=20)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    for split in ("train", "dev", "test"):
        for example in load_split(args.data_dir / f"{split}.json", split):
            if not example.program or not example.exe_answer:
                continue
            counts["eligible"] += 1
            try:
                predicted = execute_program(example.program, table=example.table)
            except ProgramError as error:
                counts["execution_errors"] += 1
                if len(failures) < args.max_failures:
                    failures.append(
                        {
                            "example_id": example.example_id,
                            "program": example.program,
                            "gold": example.exe_answer,
                            "error": str(error),
                        }
                    )
                continue
            if answers_match(predicted, example.exe_answer):
                counts["matches"] += 1
            else:
                counts["mismatches"] += 1
                if len(failures) < args.max_failures:
                    failures.append(
                        {
                            "example_id": example.example_id,
                            "program": example.program,
                            "gold": example.exe_answer,
                            "predicted": str(predicted),
                        }
                    )

    print(json.dumps({"counts": counts, "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
