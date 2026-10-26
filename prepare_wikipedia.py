"""Очистка всей армянской Википедии, точная дедупликация и разбиение по статьям.

Запуск: %run /workdir/prepare_wikipedia.py
Применяем правила, проверенные на выборке. Токенизация выполняется отдельно.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import time

import pyarrow as pa
import pyarrow.parquet as pq

from clean_wikipedia import CONFIG as BASIC_CONFIG, clean_text
from filter_wikipedia import CONFIG as FILTER_CONFIG, filter_text
from paths import DATA_DIR


SPLIT_CONFIG = {"seed": 42, "validation_fraction": 0.01, "test_fraction": 0.01}
SCHEMA = pa.schema([
    ("id", pa.string()), ("url", pa.string()), ("title", pa.string()),
    ("text", pa.string()), ("source", pa.string()), ("split", pa.string()),
    ("text_sha256", pa.string()), ("armenian_share", pa.float64()),
    ("repetition_fraction", pa.float64()), ("language_salvaged", pa.bool_()),
])


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def choose_split(digest, calibration_hashes, config=None):
    config = config or SPLIT_CONFIG
    if digest in calibration_hashes:
        return "train"
    value = int(hashlib.sha256(f"{config['seed']}:{digest}".encode()).hexdigest()[:16], 16) / 2**64
    if value < config["validation_fraction"]:
        return "validation"
    if value < config["validation_fraction"] + config["test_fraction"]:
        return "test"
    return "train"


def main(raw_dir=None, analysis_dir=None, output_dir=None):
    name = "wikipedia_hy_20231101"
    raw_dir = Path(raw_dir) if raw_dir is not None else DATA_DIR / "raw" / name
    analysis_dir = Path(analysis_dir) if analysis_dir is not None else DATA_DIR / "analysis" / name
    output_dir = Path(output_dir) if output_dir is not None else DATA_DIR / "processed" / name
    if output_dir.resolve() in (raw_dir.resolve(), analysis_dir.resolve()):
        raise ValueError("Для готового корпуса нужен отдельный каталог")
    if (output_dir / "stats.json").exists():
        raise FileExistsError(f"Корпус уже подготовлен: {output_dir / 'stats.json'}")

    source_info = json.loads((raw_dir / "download_info.json").read_text(encoding="utf-8"))
    review_path = analysis_dir / "filtering_stats.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["config"] != FILTER_CONFIG or review["basic_cleaning_config"] != BASIC_CONFIG:
        raise ValueError("Правила отличаются от проверенных на выборке")
    for key in ("repo_id", "revision", "subset"):
        if review["source"][key] != source_info[key]:
            raise ValueError("Просмотренная выборка получена из другой версии Википедии")
    files = [raw_dir / item["file"] for item in source_info["source_files"]]
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError("Исходные Parquet не найдены; сначала запусти download_wikipedia.py")

    # Решение о train принимаем для всей группы дубликатов, даже если её первая
    # статья не входила в выборку, по которой настраивали очистку.
    calibration_path = analysis_dir / "sample_filtered.jsonl"
    calibration_hashes = set()
    calibration_count = 0
    with calibration_path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                calibration_hashes.add(hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
                calibration_count += 1
    if calibration_count != review["kept_documents"]:
        raise ValueError("Размер просмотренной выборки не совпал с filtering_stats.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ("train", "validation", "test")
    writers, buffers, temporary_paths, final_paths = {}, {}, {}, {}
    lengths = {split: [] for split in splits}
    split_stats = {split: {"kept_documents": 0, "kept_characters": 0,
                           "kept_text_bytes": 0, "calibration_groups": 0} for split in splits}
    counts, rejections, seen = Counter(), Counter(), {}
    rejected_path = output_dir / "rejected.jsonl"
    temporary_rejected = rejected_path.with_suffix(".jsonl.part")
    started = time.monotonic()
    next_progress = 50_000

    def flush(split):
        if buffers[split]:
            writers[split].write_table(pa.Table.from_pylist(buffers[split], schema=SCHEMA))
            buffers[split].clear()

    try:
        for split in splits:
            split_dir = output_dir / split
            split_dir.mkdir(exist_ok=True)
            final_paths[split] = split_dir / "part-0000.parquet"
            temporary_paths[split] = split_dir / "part-0000.parquet.part"
            writers[split] = pq.ParquetWriter(temporary_paths[split], SCHEMA, compression="zstd")
            buffers[split] = []
        with temporary_rejected.open("w", encoding="utf-8") as audit:
            for path in files:
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(batch_size=2048, columns=["id", "url", "title", "text"]):
                        for row in batch.to_pylist():
                            original = row["text"] or ""
                            counts["input_documents"] += 1
                            counts["input_characters"] += len(original)
                            counts["input_text_bytes"] += len(original.encode("utf-8"))
                            basic, changes = clean_text(original)
                            counts["normalized_documents"] += changes["normalized"]
                            counts["articles_with_tail_removed"] += changes["service_heading"] is not None
                            counts["empty_parentheses_removed"] += changes["empty_parentheses_removed"]
                            text, reason, details = filter_text(basic, FILTER_CONFIG)
                            counts["language_salvage_attempts"] += details["language_salvage_attempted"]
                            digest = None
                            if reason is None:
                                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                                if digest in seen:
                                    reason = "exact_duplicate"
                                    details["duplicate_of_id"] = seen[digest]
                                else:
                                    seen[digest] = row["id"]
                            if reason is not None:
                                rejections[reason] += 1
                                audit.write(json.dumps({
                                    "id": row["id"], "title": row["title"], "url": row["url"],
                                    "reason": reason, "filtering": details,
                                    "candidate_preview": text[:500], "text_sha256": digest,
                                }, ensure_ascii=False) + "\n")
                                continue

                            split = choose_split(digest, calibration_hashes)
                            buffers[split].append({
                                "id": row["id"], "url": row["url"], "title": row["title"], "text": text,
                                "source": "wikipedia_hy_20231101", "split": split, "text_sha256": digest,
                                "armenian_share": details["armenian_share_after"],
                                "repetition_fraction": details["repetition_fraction"],
                                "language_salvaged": details["language_salvage_attempted"],
                            })
                            info = split_stats[split]
                            info["kept_documents"] += 1
                            info["kept_characters"] += len(text)
                            info["kept_text_bytes"] += len(text.encode("utf-8"))
                            info["calibration_groups"] += digest in calibration_hashes
                            lengths[split].append(len(text))
                            counts["kept_documents"] += 1
                            counts["kept_after_language_salvage"] += details["language_salvage_attempted"]
                            if len(buffers[split]) >= 2048:
                                flush(split)
                        if counts["input_documents"] >= next_progress:
                            print(f"Обработано {counts['input_documents']:,}/{source_info['documents']:,}; "
                                  f"сохранено {counts['kept_documents']:,}; {time.monotonic() - started:.0f} с", flush=True)
                            next_progress = counts["input_documents"] + 50_000
                print(f"Готово: {path.name}", flush=True)
        for split in splits:
            flush(split)
    finally:
        for writer in writers.values():
            writer.close()

    if counts["input_documents"] != source_info["documents"]:
        raise RuntimeError("Число исходных статей не совпало с download_info.json")
    if counts["input_documents"] != counts["kept_documents"] + sum(rejections.values()):
        raise RuntimeError("Не сошлись счётчики статей")
    for split in splits:
        info = split_stats[split]
        if pq.read_metadata(temporary_paths[split]).num_rows != info["kept_documents"]:
            raise RuntimeError(f"Не совпало число сохранённых строк: {split}")
        temporary_paths[split].replace(final_paths[split])
        info.update(
            files=[str(final_paths[split].relative_to(output_dir))],
            parquet_bytes=final_paths[split].stat().st_size,
            median_characters=median(lengths[split]) if lengths[split] else None,
            max_characters=max(lengths[split]) if lengths[split] else None,
            parquet_sha256=file_sha256(final_paths[split]),
        )
    temporary_rejected.replace(rejected_path)

    source_dir = Path(__file__).resolve().parent
    code_dir = output_dir / "code"
    code_dir.mkdir(exist_ok=True)
    code_hashes = {}
    for filename in ("prepare_wikipedia.py", "clean_wikipedia.py", "filter_wikipedia.py",
                     "inspect_wikipedia.py", "prepare_data.py", "paths.py"):
        shutil.copy2(source_dir / filename, code_dir / filename)
        code_hashes[filename] = file_sha256(code_dir / filename)
    full_filter_config = {**FILTER_CONFIG, "deduplication": "exact SHA256 of final cleaned UTF-8 text across all Wikipedia articles before splitting"}
    report = {
        "status": "complete", "source": source_info,
        "config": {"basic_cleaning": BASIC_CONFIG, "filtering": full_filter_config, "splitting": SPLIT_CONFIG},
        "counts": dict(counts), "rejections": dict(rejections), "splits": split_stats,
        "split_method": "seeded hash of final cleaned-text SHA256; reviewed calibration groups forced to train",
        "calibration": {
            "documents": calibration_count, "text_groups": len(calibration_hashes),
            "sample_sha256": file_sha256(calibration_path), "review_sha256": file_sha256(review_path),
            "groups_found_in_train": split_stats["train"]["calibration_groups"],
        },
        "code_sha256": code_hashes, "pyarrow_version": pa.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "limitations_ru": [
            "Удаляются точные дубли после очистки. Близкие по смыслу и шаблонные статьи могут оставаться в разных выборках.",
            "Пересечения с ArmWeb не проверялись.",
            "Порог длины может исключать корректные короткие определения; очистка не проверяет истинность фактов.",
        ],
    }
    stats_path = output_dir / "stats.json"
    temporary = stats_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(stats_path)

    print(f"\nИсходных статей: {counts['input_documents']:,}; сохранено уникальных: {counts['kept_documents']:,}")
    print("Исключено — первая подходящая причина:")
    for reason, count in rejections.items():
        print(f"  {reason}: {count:,}")
    print(f"Сохранено статей после извлечения армянских блоков: {counts['kept_after_language_salvage']:,}")
    print("\nВыборки Википедии:")
    for split, info in split_stats.items():
        print(f"{split}: {info['kept_documents']:,} статей; {info['kept_text_bytes'] / 10**6:.1f} МБ UTF-8; "
              f"медиана {info['median_characters']} символов")
    print(f"Групп из просмотренной выборки в train: {split_stats['train']['calibration_groups']:,}")
    print(f"\nКаталог: {output_dir}\nСтатистика: {stats_path}\nЖурнал исключений: {rejected_path}")


if __name__ == "__main__":
    main()
