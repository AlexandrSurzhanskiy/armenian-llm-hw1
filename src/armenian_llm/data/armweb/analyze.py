from armenian_llm.paths import DATA_DIR, RAW_DIR
from pathlib import Path
from collections import Counter
import json
import random
import re
import statistics

import pyarrow.parquet as pq


def main():
    files = sorted((RAW_DIR / "data" / "train").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("Не найдены файлы train")

    total = sum(pq.read_metadata(path).num_rows for path in files)

    rng = random.Random(42)
    selected = set(rng.sample(range(total), 5000))

    sample = []
    position = 0

    for path in files:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(
                batch_size=2048,
                columns=["id", "text", "source", "url"],
            ):
                for row in batch.to_pylist():
                    if position in selected:
                        sample.append(row)
                    position += 1

        print(f"Прочитан {path.name}", flush=True)

    assert len(sample) == 5000

    sample_path = (DATA_DIR / "train_sample.jsonl")
    with sample_path.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = []
    issues = Counter()
    sources = Counter()

    entity_pattern = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z]+);")

    for row in sample:
        text = row["text"] or ""
        lengths.append(len(text))
        sources[row["source"]] += 1

        issues["Пустой текст"] += not text.strip()
        issues["HTML-сущности"] += bool(entity_pattern.search(text))
        issues["Цепочки из 10 и более точек"] += bool(re.search(r"\.{10,}", text))

        words = re.findall(r"[^\W\d_]+", text.lower())
        grams = [tuple(words[i:i + 10]) for i in range(len(words) - 9)]

        if grams:
            repeat_fraction = 1 - len(set(grams)) / len(grams)
            issues["Доля повторных 10-грамм > 20%"] += repeat_fraction > 0.2

    quantiles = statistics.quantiles(lengths, n=100, method="inclusive")

    print(f"\nДокументов в выборке: {len(sample)}")
    print("\nДлины текстов в символах:")
    print(f"Минимум: {min(lengths)}")
    print(f"Медиана: {statistics.median(lengths):.0f}")
    print(f"90-й процентиль: {quantiles[89]:.0f}")
    print(f"99-й процентиль: {quantiles[98]:.0f}")
    print(f"Максимум: {max(lengths)}")

    print("\nПризнаки возможных проблем:")
    for name, count in issues.items():
        print(f"{name}: {count} ({count / len(sample):.2%})")

    print("\nИсточники в выборке:")
    for source, count in sources.most_common():
        print(f"{source}: {count} ({count / len(sample):.2%})")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
