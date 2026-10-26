"""Скачивание армянской Википедии и просмотр трёх случайных статей.

Запуск из ноутбука: %run /workdir/download_wikipedia.py
Сохраняем исходные Parquet. Очистка и разбиение на выборки выполняются отдельно.
"""

import json
import random

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

from paths import DATA_DIR


REPO_ID = "wikimedia/wikipedia"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SUBSET = "20231101.hy"
PREVIEW_SEED = 42
PREVIEW_CHARACTERS = 1500


def read_article(files, index):
    """Читаем только row group, содержащую выбранную статью."""
    if index < 0:
        raise IndexError(index)
    for path in files:
        with pq.ParquetFile(path) as parquet:
            if index >= parquet.metadata.num_rows:
                index -= parquet.metadata.num_rows
                continue
            for group in range(parquet.metadata.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                if index >= size:
                    index -= size
                    continue
                table = parquet.read_row_group(group, columns=["id", "url", "title", "text"])
                return table.slice(index, 1).to_pylist()[0]
    raise IndexError("Индекс статьи вышел за границы корпуса")


def main():
    output_dir = DATA_DIR / "raw" / "wikipedia_hy_20231101"
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=[f"{SUBSET}/*.parquet", "README.md"],
        local_dir=output_dir,
        token=False,
    )
    files = sorted((output_dir / SUBSET).glob("*.parquet"))
    if not files:
        raise FileNotFoundError("Файлы армянской Википедии не найдены")

    info = {
        "repo_id": REPO_ID, "revision": REVISION, "subset": SUBSET,
        "language": "hy", "snapshot_date": "2023-11-01", "source_split": "train",
        "source_url": f"https://huggingface.co/datasets/{REPO_ID}/tree/{REVISION}/{SUBSET}",
        "source_files": [], "preview_seed": PREVIEW_SEED,
        "preview_characters": PREVIEW_CHARACTERS,
    }
    total_rows = total_bytes = 0
    print(f"\nКаталог: {output_dir}", flush=True)
    print(f"Снимок: {SUBSET}; файлов Parquet: {len(files)}", flush=True)
    for path in files:
        with pq.ParquetFile(path) as parquet:
            rows = parquet.metadata.num_rows
            columns = parquet.schema_arrow.names
        size = path.stat().st_size
        total_rows += rows
        total_bytes += size
        info["source_files"].append({
            "file": str(path.relative_to(output_dir)), "rows": rows, "bytes": size,
            "columns": columns,
        })
        print(f"{path.name}: {rows:,} статей, {size / 10**6:.1f} МБ", flush=True)
    if total_rows == 0:
        raise ValueError("В скачанных файлах нет статей")
    print(f"\nВсего статей: {total_rows:,}", flush=True)
    print(f"Размер Parquet: {total_bytes / 10**6:.1f} МБ", flush=True)
    print("Поля:", info["source_files"][0]["columns"], flush=True)

    indices = random.Random(PREVIEW_SEED).sample(range(total_rows), min(3, total_rows))
    info.update(documents=total_rows, parquet_bytes=total_bytes, preview_row_indices=indices)
    print("\nТри случайные статьи:", flush=True)
    for number, index in enumerate(indices, start=1):
        row = read_article(files, index)
        print(f"\nПример {number}; индекс {index}; id {row['id']}", flush=True)
        print("Заголовок:", row["title"], flush=True)
        print("Ссылка:", row["url"], flush=True)
        print("Длина полного текста:", len(row["text"]), "символов", flush=True)
        print(f"Текст — первые {PREVIEW_CHARACTERS} символов:", flush=True)
        print(row["text"][:PREVIEW_CHARACTERS], flush=True)

    metadata_path = output_dir / "download_info.json"
    temporary = metadata_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    print(f"\nМетаданные: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
