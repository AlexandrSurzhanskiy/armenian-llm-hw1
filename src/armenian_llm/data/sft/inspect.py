"""Первый разбор качества существующего SFT train, без изменения данных.

%run -m armenian_llm.data.sft.inspect
Показываем распределение длин и по две случайные пары из каждой группы.
Это выборка для разбора типов ошибок, а не оценка их частоты во всём train.
"""

from pathlib import Path
import argparse
import hashlib
import json
import random
from statistics import mean, median

from armenian_llm.paths import DATA_DIR


BUCKETS = (
    ("1–8", 1, 8),
    ("9–24", 9, 24),
    ("25–64", 25, 64),
    ("65+", 65, None),
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main(input_dir=DATA_DIR / "sft" / "aya_dolly_qa",
         output_dir=DATA_DIR / "analysis" / "sft_aya_dolly_qa",
         examples_per_group=2, seed=42):
    if examples_per_group < 1:
        raise ValueError("examples_per_group должен быть положительным")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    train_path = input_dir / "train.jsonl"
    with train_path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError("Train пуст")
    # При подготовке каждая последовательность ответа завершалась одним EOS.
    lengths = [row["answer_tokens_including_eos"] - 1 for row in rows]
    if min(lengths) < 1:
        raise ValueError("Найдены ответы без текстовых токенов")
    total_targets = sum(row["answer_tokens_including_eos"] for row in rows)
    rng = random.Random(seed)
    distributions, samples = [], []
    print(f"Train: {len(rows):,} пар")
    print(f"Токенов в ответе без EOS: среднее {mean(lengths):.1f}; "
          f"медиана {median(lengths):.1f}; минимум {min(lengths)}; максимум {max(lengths)}")
    print("\nДлина ответа без EOS | Пар | Доля пар | Доля целевых токенов с EOS")
    for label, lower, upper in BUCKETS:
        bucket = [row for row, length in zip(rows, lengths)
                  if length >= lower and (upper is None or length <= upper)]
        targets = sum(row["answer_tokens_including_eos"] for row in bucket)
        info = {"answer_tokens_without_eos": label, "examples": len(bucket),
                "example_fraction": len(bucket) / len(rows),
                "target_tokens_including_eos": targets, "target_token_fraction": targets / total_targets}
        distributions.append(info)
        print(f"{label:>19} | {len(bucket):>4} | {len(bucket) / len(rows):>8.2%} | {targets / total_targets:>25.2%}")
        for row in rng.sample(bucket, min(examples_per_group, len(bucket))):
            samples.append({
                "number": len(samples) + 1, "id": row["id"], "source_ids": row["source_ids"],
                "length_group": label, "answer_tokens_without_eos": row["answer_tokens_including_eos"] - 1,
                "question": row["question"], "answer": row["answer"],
                "review": {"decision": "pending", "issues": [], "comment_ru": ""},
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample.jsonl"
    temporary = sample_path.with_suffix(".jsonl.part")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    temporary.replace(sample_path)
    metadata_path = input_dir / "metadata.json"
    report = {
        "source_split": "train", "source_file": str(train_path.resolve()),
        "source_sha256": file_sha256(train_path),
        "source_metadata_sha256": file_sha256(metadata_path) if metadata_path.exists() else None,
        "examples": len(rows), "target_tokens_including_eos": total_targets,
        "answer_tokens_without_eos": {"mean": mean(lengths), "median": median(lengths),
                                       "min": min(lengths), "max": max(lengths)},
        "distributions": distributions, "sample_examples": len(samples),
        "sampling": "uniform without replacement within each answer-length group",
        "seed": seed, "examples_per_group": examples_per_group,
        "sample_role": "train data development; manually review before choosing filtering rules",
        "quality_labels": "pending; no automatic factual or translation correctness judgments",
        "script_sha256": file_sha256(Path(__file__)),
    }
    stats_path = output_dir / "stats.json"
    temporary = stats_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(stats_path)
    print(f"\nПары для первого ручного разбора: {len(samples)}")
    for row in samples:
        print(f"\nПример {row['number']}; группа {row['length_group']}; "
              f"токенов ответа без EOS: {row['answer_tokens_without_eos']}; source_ids={row['source_ids']}")
        print("Вопрос:", row["question"])
        print("Ответ:", row["answer"])
    print(f"\nВыборка: {sample_path}\nСтатистика: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "sft" / "aya_dolly_qa")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "analysis" / "sft_aya_dolly_qa")
    parser.add_argument("--examples-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.examples_per_group, args.seed)
