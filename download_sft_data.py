"""Скачивание армянской части переведённого Dolly из Aya Collection.

Сохраняем все армянские строки и показываем короткие QA-кандидаты. Фильтр
по вопросительному знаку является эвристикой, а не готовой разметкой QA.
Разбиение, очистка, удаление контрольных фактов и SFT выполняются отдельно.
"""

from collections import Counter
import json
import random

from huggingface_hub import snapshot_download
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from paths import DATA_DIR


REPO_ID = "CohereLabs/aya_collection"
REVISION = "09069079fad96ad9c7f8781be8c1d70471dfa768"
SUBSET = "translated_dolly"


def is_qa_candidate(row):
    question, answer = row.get("inputs"), row.get("targets")
    return (
        isinstance(question, str) and isinstance(answer, str)
        and ("՞" in question or "?" in question)
        and 5 <= len(question.strip()) <= 800
        and 1 <= len(answer.strip()) <= 600
        and "context:" not in question.casefold()
    )


def main():
    output_dir = DATA_DIR / "raw" / "aya_dolly"
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"{SUBSET}/*.parquet", "README.md"],
        local_dir=output_dir,
        token=False,
    )

    info = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "subset": SUBSET,
        "source_url": f"https://huggingface.co/datasets/{REPO_ID}/tree/{REVISION}/{SUBSET}",
        "original_dolly_url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "language_filter": "hye",
        "source_files": [],
        "qa_preview_filter": {
            "question_mark": ["՞", "?"], "question_characters": [5, 800],
            "answer_characters": [1, 600], "exclude_literal_context_prefix": True,
        },
    }
    print(f"\nКаталог: {output_dir}")
    source_files = sorted((output_dir / SUBSET).glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError("Исходные Parquet-файлы не найдены")
    selected_path = output_dir / "armenian.parquet"
    temporary_path = output_dir / "armenian.parquet.part"
    writer = None
    scanned = selected_count = 0
    try:
        for path in source_files:
            parquet = pq.ParquetFile(path)
            info["source_files"].append({
                "file": str(path.relative_to(output_dir)),
                "rows": parquet.metadata.num_rows, "bytes": path.stat().st_size,
            })
            for batch in parquet.iter_batches(batch_size=4096):
                table = pa.Table.from_batches([batch])
                scanned += table.num_rows
                armenian = table.filter(pc.equal(table["language"], "hye"))
                if armenian.num_rows:
                    if writer is None:
                        writer = pq.ParquetWriter(temporary_path, armenian.schema, compression="zstd")
                    writer.write_table(armenian)
                    selected_count += armenian.num_rows
            print(f"Прочитан {path.name}: всего {scanned:,} строк, армянских {selected_count:,}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if selected_count == 0:
        raise ValueError("В исходных файлах не найдены строки language=hye")
    temporary_path.replace(selected_path)

    table = pq.read_table(selected_path)
    records = table.to_pylist()
    print(f"\nАрмянских записей: {len(records):,}")
    print("Поля:", table.column_names)
    for field in ("language", "task_type"):
        counts = Counter(str(row.get(field)) for row in records)
        print(f"{field}:", dict(counts))
        info[f"{field}_counts"] = dict(counts)

    candidates = [row for row in records if is_qa_candidate(row)]
    info.update(scanned_rows=scanned, armenian_rows=len(records),
                armenian_file=selected_path.name, qa_candidates=len(candidates), preview_seed=42)
    print(f"Коротких QA-кандидатов по эвристике: {len(candidates):,}")
    indices = random.Random(42).sample(range(len(candidates)), min(3, len(candidates)))
    print("\nТри случайных QA-кандидата:")
    for number, index in enumerate(indices, start=1):
        row = candidates[index]
        print(f"\nПример {number}; id {row.get('id')}")
        print("Вопрос:", row["inputs"])
        print("Ответ:", row["targets"])

    info_path = output_dir / "download_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nМетаданные: {info_path}")


if __name__ == "__main__":
    main()
