"""Дополнительный ручной разбор коротких QA из существующего SFT train.

%run /workdir/inspect_short_sft.py
Ответы 1–24 токена без EOS; уже показанные пары исключены из выборки.
Исходные train/validation/test не изменяются.
"""

from pathlib import Path
import argparse
import json
import random

from inspect_sft_data import file_sha256
from paths import DATA_DIR


def main(sample_size=12, seed=43):
    if sample_size < 1:
        raise ValueError("sample_size должен быть положительным")
    train_path = DATA_DIR / "sft" / "aya_dolly_qa" / "train.jsonl"
    analysis_dir = DATA_DIR / "analysis" / "sft_aya_dolly_qa"
    with train_path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    previous_path = analysis_dir / "sample.jsonl"
    seen = set()
    if previous_path.exists():
        with previous_path.open(encoding="utf-8") as source:
            seen = {json.loads(line)["id"] for line in source if line.strip()}
    short = [row for row in rows if 1 <= row["answer_tokens_including_eos"] - 1 <= 24]
    candidates = [row for row in short if row["id"] not in seen]
    if not candidates:
        raise ValueError("Нет новых коротких пар для просмотра")
    selected = random.Random(seed).sample(candidates, min(sample_size, len(candidates)))
    samples = [{
        "number": number, "id": row["id"], "source_ids": row["source_ids"],
        "answer_tokens_without_eos": row["answer_tokens_including_eos"] - 1,
        "question": row["question"], "answer": row["answer"],
        "review": {"decision": "pending", "issues": [], "comment_ru": ""},
    } for number, row in enumerate(selected, start=1)]
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sample_path = analysis_dir / "short_sample_v2.jsonl"
    temporary = sample_path.with_suffix(".jsonl.part")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    temporary.replace(sample_path)
    report = {
        "source_file": str(train_path.resolve()), "source_sha256": file_sha256(train_path),
        "source_split": "train", "short_answer_tokens_without_eos": [1, 24],
        "short_examples": len(short), "excluded_previously_shown": len(short) - len(candidates),
        "sampling_population": len(candidates), "sample_size": len(samples), "seed": seed,
        "sampling": "uniform without replacement from short train answers excluding the first audit sample",
        "previous_sample_sha256": file_sha256(previous_path) if previous_path.exists() else None,
        "script_sha256": file_sha256(Path(__file__)), "quality_labels": "pending manual review",
    }
    stats_path = analysis_dir / "short_stats_v2.json"
    temporary = stats_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(stats_path)
    print(f"Коротких ответов в train: {len(short):,}")
    print(f"Исключено уже показанных: {len(short) - len(candidates)}; новых примеров: {len(samples)}; seed={seed}")
    print("Это кандидаты для ручного разбора, а не готовый очищенный SFT-набор.")
    for row in samples:
        print(f"\nПример {row['number']}; токенов ответа: {row['answer_tokens_without_eos']}; source_ids={row['source_ids']}")
        print("Вопрос:", row["question"])
        print("Ответ:", row["answer"])
    print(f"\nВыборка: {sample_path}\nСтатистика: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    main(args.sample_size, args.seed)
