"""Очистка ArmWeb по правилам, выбранным на обучающей выборке."""

from collections import Counter
import html
import json
import platform
import re
import time
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

from paths import PROCESSED_DIR, PROJECT_DIR, RAW_DIR


WORD_PATTERN = re.compile(r"[^\W\d_]+")
SPLITS = ("train", "validation", "test_iid", "test_tail")


def normalize_text(text):
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repetition_fraction(text, ngram_size):
    """Доля повторных n-грамм; регистр, числа и пунктуация игнорируются."""
    words = WORD_PATTERN.findall(text.lower())
    count = len(words) - ngram_size + 1
    if count <= 0:
        return 0.0
    unique = {
        tuple(words[i:i + ngram_size])
        for i in range(count)
    }
    return 1 - len(unique) / count


def armenian_share(text):
    """Доля армянских букв среди всех букв документа."""
    # Название Unicode проверяется один раз для каждого различного символа.
    counts = Counter(text)
    letters = sum(n for char, n in counts.items() if char.isalpha())
    if letters == 0:
        return 0.0
    armenian = sum(
        n for char, n in counts.items()
        if char.isalpha()
        and unicodedata.name(char, "").startswith("ARMENIAN")
    )
    return armenian / letters


def clean_document(text, config):
    text = normalize_text(text)
    if not text:
        return text, "empty", None, None

    repetition = repetition_fraction(text, config["ngram_size"])
    if repetition > config["max_repetition_fraction"]:
        return text, "repetition", repetition, None

    share = armenian_share(text)
    if share < config["min_armenian_share"]:
        return text, "language", repetition, share

    return text, None, repetition, share


def process_split(split, files, output_dir, config):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    stats = dict.fromkeys([
        "input_documents", "kept_documents", "dropped_empty",
        "dropped_repetition", "dropped_language", "normalized_kept_documents",
        "input_characters", "kept_characters",
        "input_text_bytes", "kept_text_bytes",
    ], 0)
    sources_before = Counter()
    sources_after = Counter()
    expected_documents = sum(pq.read_metadata(p).num_rows for p in files)
    started = time.monotonic()
    next_progress = 50_000

    audit_path = output_dir / f"{split}_rejected.jsonl"
    with audit_path.open("w", encoding="utf-8") as rejected_file:
        for path in files:
            target = split_dir / path.name
            temporary = target.with_name(target.name + ".part")
            kept_in_file = 0

            with pq.ParquetFile(path) as parquet:
                with pq.ParquetWriter(
                    temporary, parquet.schema_arrow, compression="zstd"
                ) as writer:
                    for batch in parquet.iter_batches(batch_size=4096):
                        kept = []
                        for row in batch.to_pylist():
                            original = row["text"] or ""
                            source = row["source"] or "unknown"
                            stats["input_documents"] += 1
                            stats["input_characters"] += len(original)
                            stats["input_text_bytes"] += len(original.encode("utf-8"))
                            sources_before[source] += 1

                            text, reason, repetition, share = clean_document(
                                original, config
                            )
                            if reason is not None:
                                stats[f"dropped_{reason}"] += 1
                                record = {
                                    "id": row["id"],
                                    "source": row["source"],
                                    "url": row["url"],
                                    "reason": reason,
                                    "repetition_fraction": repetition,
                                    "armenian_share": share,
                                }
                                rejected_file.write(
                                    json.dumps(record, ensure_ascii=False) + "\n"
                                )
                                continue

                            stats["kept_documents"] += 1
                            stats["normalized_kept_documents"] += text != original
                            stats["kept_characters"] += len(text)
                            stats["kept_text_bytes"] += len(text.encode("utf-8"))
                            sources_after[source] += 1
                            kept.append({**row, "text": text})

                        if kept:
                            table = pa.Table.from_pylist(
                                kept, schema=parquet.schema_arrow
                            )
                            writer.write_table(table)
                            kept_in_file += len(kept)

                        if stats["input_documents"] >= next_progress:
                            elapsed = time.monotonic() - started
                            print(
                                f"{split}: обработано {stats['input_documents']:,}"
                                f"/{expected_documents:,}, сохранено"
                                f" {stats['kept_documents']:,}, {elapsed:.0f} с",
                                flush=True,
                            )
                            next_progress = stats["input_documents"] + 50_000

            if pq.read_metadata(temporary).num_rows != kept_in_file:
                raise RuntimeError(f"Не совпало число сохранённых строк: {path}")
            temporary.replace(target)
            print(f"Готово: {split}/{path.name}", flush=True)

    dropped = sum(stats[f"dropped_{reason}"] for reason in (
        "empty", "repetition", "language"
    ))
    if stats["input_documents"] != expected_documents:
        raise RuntimeError(f"Не совпало число исходных документов: {split}")
    if stats["input_documents"] != stats["kept_documents"] + dropped:
        raise RuntimeError(f"Не сошлись счётчики документов: {split}")

    stats["sources_before"] = dict(sources_before)
    stats["sources_after"] = dict(sources_after)
    return stats


def main(
    raw_dir=RAW_DIR / "data",
    output_dir=PROCESSED_DIR,
    config_path=PROJECT_DIR / "cleaning_config.json",
):
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config["ngram_size"], int) or config["ngram_size"] < 1:
        raise ValueError("ngram_size должен быть положительным целым числом")
    for key in ("max_repetition_fraction", "min_armenian_share"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} должен находиться в диапазоне [0, 1]")
    if raw_dir.resolve() == output_dir.resolve():
        raise ValueError("Каталоги исходных и обработанных данных должны различаться")

    split_files = {
        split: sorted((raw_dir / split).glob("*.parquet"))
        for split in SPLITS
    }
    for split, files in split_files.items():
        if not files:
            raise FileNotFoundError(f"Нет исходных Parquet-файлов: {raw_dir / split}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "stats.json"
    stats_path.unlink(missing_ok=True)
    report = {
        "config": config,
        "filter_order": ["normalize", "empty", "repetition", "language"],
        "versions": {
            "python": platform.python_version(),
            "pyarrow": pa.__version__,
            "unicode": unicodedata.unidata_version,
        },
        "splits": {},
    }

    for split, files in split_files.items():
        report["splits"][split] = process_split(
            split, files, output_dir, config
        )

    temporary_stats = output_dir / "stats.json.part"
    with temporary_stats.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    temporary_stats.replace(stats_path)

    print("\nИтог очистки:", flush=True)
    for split, stats in report["splits"].items():
        before = stats["input_documents"]
        after = stats["kept_documents"]
        print(f"{split}: {before:,} -> {after:,}, сохранено {after / before:.2%}")
        print(
            f"  Исключено: пустые {stats['dropped_empty']:,},"
            f" повторы {stats['dropped_repetition']:,},"
            f" письменность {stats['dropped_language']:,}"
        )
        print(
            f"  Текст UTF-8: {stats['input_text_bytes'] / 10**9:.3f} ->"
            f" {stats['kept_text_bytes'] / 10**9:.3f} ГБ"
        )
    print(f"\nСтатистика: {stats_path.resolve()}")


if __name__ == "__main__":
    main()
