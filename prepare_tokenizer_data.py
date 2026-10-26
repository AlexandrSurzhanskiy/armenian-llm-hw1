"""Равномерные выборки из очищенного ArmWeb для экспериментов с токенизатором."""

from bisect import bisect_left
from collections import Counter
import json
import random

import pyarrow.parquet as pq

from paths import PROCESSED_DIR, TOKENIZER_DATA_DIR


def sample_split(split, input_dir, output_dir, requested, seed, expected_rows):
    files = sorted((input_dir / split).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет Parquet-файлов: {input_dir / split}")

    total = sum(pq.read_metadata(path).num_rows for path in files)
    if total != expected_rows:
        raise ValueError(f"Число документов {split} не совпало со stats.json")
    if total == 0:
        raise ValueError(f"Пустая часть корпуса: {split}")

    count = min(requested, total)
    indices = sorted(random.Random(seed).sample(range(total), count))
    target = output_dir / f"{split}.jsonl"
    temporary = target.with_name(target.name + ".part")
    offset = 0
    selected = 0
    text_bytes = 0
    sources = Counter()

    with temporary.open("w", encoding="utf-8") as output:
        for path in files:
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(
                    batch_size=4096,
                    columns=["id", "text", "source", "url"],
                ):
                    left = bisect_left(indices, offset)
                    right = bisect_left(indices, offset + batch.num_rows)
                    local_indices = [i - offset for i in indices[left:right]]

                    if local_indices:
                        rows = batch.take(local_indices).to_pylist()
                        for row in rows:
                            output.write(json.dumps(row, ensure_ascii=False) + "\n")
                            text_bytes += len(row["text"].encode("utf-8"))
                            sources[row["source"] or "unknown"] += 1
                        selected += len(rows)

                    offset += batch.num_rows

            print(
                f"{split}/{path.name}: выбрано {selected:,}/{count:,}",
                flush=True,
            )

    if offset != total or selected != count:
        raise RuntimeError(f"Не сошлись счётчики выборки: {split}")
    temporary.replace(target)

    return {
        "input_documents": total,
        "sample_documents": selected,
        "sample_text_bytes": text_bytes,
        "sample_file_bytes": target.stat().st_size,
        "sources": dict(sources),
    }


def main(
    input_dir=PROCESSED_DIR,
    output_dir=TOKENIZER_DATA_DIR,
    train_size=100_000,
    validation_size=5_000,
    seed=42,
):
    if train_size < 1 or validation_size < 1:
        raise ValueError("Размеры выборок должны быть положительными")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Выборки нужно сохранять отдельно от исходных данных")
    with (input_dir / "stats.json").open(encoding="utf-8") as f:
        processing_stats = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = output_dir / "sample_info.json"
    info_path.unlink(missing_ok=True)
    info = {
        "seed": seed,
        "selection": "uniform_without_replacement",
        "input_dir": str(input_dir.resolve()),
        "cleaning_config": processing_stats["config"],
        "splits": {},
    }

    for split, size in (("train", train_size), ("validation", validation_size)):
        info["splits"][split] = sample_split(
            split, input_dir, output_dir, size, seed,
            processing_stats["splits"][split]["kept_documents"],
        )

    temporary_info = output_dir / "sample_info.json.part"
    with temporary_info.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    temporary_info.replace(info_path)

    print("\nВыборки для токенизатора:")
    for split, stats in info["splits"].items():
        print(
            f"{split}: {stats['sample_documents']:,} документов,"
            f" {stats['sample_text_bytes'] / 10**6:.1f} МБ текста UTF-8"
        )
    print(f"Каталог: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
