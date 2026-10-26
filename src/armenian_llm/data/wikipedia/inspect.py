"""Статистика случайных 5000 статей для выбора правил очистки Википедии.

Запуск: %run -m armenian_llm.data.wikipedia.inspect
Исходный корпус не меняется. Все признаки являются диагностическими.
"""

from bisect import bisect_left
import json
import random
import re
from statistics import median, quantiles

import pyarrow.parquet as pq

from armenian_llm.paths import DATA_DIR
from armenian_llm.data.armweb.prepare import armenian_share, normalize_text, repetition_fraction


SAMPLE_SIZE = 5000
SEED = 42
# Пока отмечаем только два служебных заголовка, уже замеченных в примерах.
SERVICE_HEADINGS = ("Ծանոթագրություններ", "Արտաքին հղումներ")
HEADING_PATTERN = re.compile(
    r"^[ \t]*(?:" + "|".join(map(re.escape, SERVICE_HEADINGS)) + r")[ \t]*$", re.MULTILINE
)
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


def sample_articles(files):
    total = sum(pq.read_metadata(path).num_rows for path in files)
    if total == 0:
        raise ValueError("В корпусе нет статей")
    indices = sorted(random.Random(SEED).sample(range(total), min(SAMPLE_SIZE, total)))
    sample, offset = [], 0
    for path in files:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=4096, columns=["id", "url", "title", "text"]):
                left = bisect_left(indices, offset)
                right = bisect_left(indices, offset + batch.num_rows)
                if left < right:
                    local = [index - offset for index in indices[left:right]]
                    sample.extend(batch.take(local).to_pylist())
                offset += batch.num_rows
        print(f"Прочитан {path.name}; выбрано {len(sample):,}/{len(indices):,}", flush=True)
    if len(sample) != len(indices):
        raise RuntimeError("Не совпало число выбранных статей")
    return sample, total


def inspect_article(row):
    text = normalize_text(row.get("text"))
    heading = HEADING_PATTERN.search(text)
    return {
        "characters": len(text), "armenian_share": armenian_share(text),
        "repetition_fraction": repetition_fraction(text, 10),
        "has_url": bool(URL_PATTERN.search(text)),
        "has_empty_parentheses": bool(re.search(r"\(\s*\)", text)),
        "has_wiki_markup": bool(re.search(r"\{\{|\}\}|\[\[|\]\]", text)),
        "service_heading": heading.group().strip() if heading else None,
        "service_start": heading.start() if heading else None,
        "characters_after_heading": len(text) - heading.start() if heading else 0,
    }


def main():
    source_dir = DATA_DIR / "raw" / "wikipedia_hy_20231101"
    source_info = json.loads((source_dir / "download_info.json").read_text(encoding="utf-8"))
    files = sorted((source_dir / source_info["subset"]).glob("*.parquet"))
    if not files:
        raise FileNotFoundError("Сначала запусти download_wikipedia.py")
    rows, total = sample_articles(files)
    metrics = [inspect_article(row) for row in rows]
    lengths = [item["characters"] for item in metrics]
    shares = [item["armenian_share"] for item in metrics]
    percentiles = quantiles(lengths, n=100, method="inclusive") if len(lengths) > 1 else [lengths[0]] * 99

    counts = {
        "empty": sum(length == 0 for length in lengths),
        "shorter_than_200": sum(length < 200 for length in lengths),
        "shorter_than_500": sum(length < 500 for length in lengths),
        "shorter_than_1000": sum(length < 1000 for length in lengths),
        "armenian_share_below_50_percent": sum(share < 0.5 for share in shares),
        "armenian_share_below_80_percent": sum(share < 0.8 for share in shares),
        "has_service_heading": sum(item["service_heading"] is not None for item in metrics),
        "has_url": sum(item["has_url"] for item in metrics),
        "has_empty_parentheses": sum(item["has_empty_parentheses"] for item in metrics),
        "has_wiki_markup": sum(item["has_wiki_markup"] for item in metrics),
        "repetition_above_20_percent": sum(item["repetition_fraction"] > 0.2 for item in metrics),
    }
    tail_share = sum(item["characters_after_heading"] for item in metrics) / max(1, sum(lengths))
    report = {
        "source": source_info, "population_documents": total, "sample_documents": len(rows),
        "seed": SEED, "sampling": "uniform_without_replacement_across_all_articles",
        "sample_role": "cleaning development sample; keep these article IDs in train when creating splits",
        "statistics_text": "NFC, HTML unescape and whitespace normalization; original sample texts are preserved",
        "length_characters": {"min": min(lengths), "median": median(lengths),
                              "p90": percentiles[89], "p99": percentiles[98], "max": max(lengths)},
        "median_armenian_share": median(shares), "counts": counts,
        "service_headings": list(SERVICE_HEADINGS),
        "characters_after_first_service_heading_share": tail_share,
        "note": "Overlapping diagnostic flags, not rejection reasons. Text after a heading is not automatically considered noise.",
    }
    output_dir = DATA_DIR / "analysis" / "wikipedia_hy_20231101"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample.jsonl"
    temporary = sample_path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as output:
        for row, item in zip(rows, metrics):
            output.write(json.dumps({**row, "inspection": item}, ensure_ascii=False) + "\n")
    temporary.replace(sample_path)
    report_path = output_dir / "stats.json"
    temporary = report_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)

    print(f"\nСтатей в выборке: {len(rows):,} из {total:,}")
    print("Длина нормализованного текста в символах:", report["length_characters"])
    print(f"Медианная доля армянских букв среди всех букв: {median(shares):.2%}")
    print("\nПризнаки в выборке — могут пересекаться:")
    for name, count in counts.items():
        print(f"{name}: {count:,} ({count / len(rows):.2%})")
    print(f"\nПосле первого отмеченного служебного заголовка: {tail_share:.2%} символов выборки")

    print("\nДве статьи с наименьшей долей армянских букв:")
    for index in sorted(range(len(rows)), key=lambda i: shares[i])[:2]:
        row = rows[index]
        print(f"\n{row['title']} | армянских букв {shares[index]:.2%} | {lengths[index]} символов")
        print(row["url"])
        print(normalize_text(row.get("text"))[:900])

    candidates = [i for i, item in enumerate(metrics) if item["service_start"] is not None]
    chosen = random.Random(SEED + 1).sample(candidates, min(2, len(candidates)))
    print("\nДва случайных примера со служебными разделами:")
    for index in chosen:
        row, item = rows[index], metrics[index]
        start = item["service_start"]
        print(f"\n{row['title']} | {item['service_heading']}")
        print(row["url"])
        print(normalize_text(row.get("text"))[max(0, start - 150):start + 650])
    print(f"\nВыборка: {sample_path}\nСтатистика: {report_path}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
