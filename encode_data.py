"""Токенизация очищенного корпуса без обрезания документов.

По умолчанию обрабатываем ArmWeb. Для другого корпуса задаём его каталог,
отдельный каталог результата и имена выборок через --splits.
"""

from array import array
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import sys
import time

import pyarrow
import pyarrow.parquet as pq
import tokenizers
from tokenizers import Tokenizer

from paths import DEFAULT_TOKENIZER_DIR, PROCESSED_DIR, TOKENIZED_ROOT


SPLITS = ("train", "validation", "test_iid", "test_tail")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_array(values, output, digest):
    """На диске всегда little-endian, независимо от системы."""
    if sys.byteorder != "little":
        values.byteswap()
    data = memoryview(values).cast("B")
    output.write(data)
    digest.update(data)


def quantile(histogram, q):
    """Квантиль длин документов с линейной интерполяцией."""
    position = (sum(histogram.values()) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    seen = 0
    lower_value = None
    for value, count in sorted(histogram.items()):
        if lower_value is None and seen + count > lower:
            lower_value = value
        if seen + count > upper:
            return lower_value + (value - lower_value) * (position - lower)
        seen += count
    raise ValueError("Не удалось вычислить квантиль")


def encode_split(split, files, output_dir, tokenizer, expected, batch_size):
    token_path = output_dir / f"{split}.bin"
    offset_path = output_dir / f"{split}_offsets.bin"
    temporary_tokens = token_path.with_name(token_path.name + ".part")
    temporary_offsets = offset_path.with_name(offset_path.name + ".part")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    token_digest, offset_digest = hashlib.sha256(), hashlib.sha256()
    documents = total_tokens = text_bytes = 0
    lengths = Counter()
    started = time.monotonic()
    next_progress = 100_000

    with temporary_tokens.open("wb") as tokens_file:
        with temporary_offsets.open("wb") as offsets_file:
            write_array(array("Q", [0]), offsets_file, offset_digest)
            for path in files:
                first_batch = True
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(
                        batch_size=batch_size, columns=["text"]
                    ):
                        texts = batch.column(0).to_pylist()
                        if not texts:
                            continue
                        if any(not isinstance(text, str) or not text for text in texts):
                            raise ValueError(f"Некорректный текст в {path}")
                        encodings = tokenizer.encode_batch(
                            texts, add_special_tokens=False
                        )
                        if len(encodings) != len(texts):
                            raise RuntimeError("Не совпало число текстов и кодировок")

                        # Проверяем восстановление первого документа каждого файла.
                        if first_batch:
                            restored = tokenizer.decode(
                                encodings[0].ids, skip_special_tokens=False
                            )
                            if restored != texts[0]:
                                raise RuntimeError(f"Ошибка восстановления: {path}")
                            first_batch = False

                        token_buffer = array("H")
                        offset_buffer = array("Q")
                        for encoding in encodings:
                            ids = encoding.ids
                            token_buffer.append(bos_id)
                            token_buffer.extend(ids)
                            token_buffer.append(eos_id)
                            document_tokens = len(ids) + 2
                            total_tokens += document_tokens
                            offset_buffer.append(total_tokens)
                            lengths[document_tokens] += 1

                        write_array(token_buffer, tokens_file, token_digest)
                        write_array(offset_buffer, offsets_file, offset_digest)
                        documents += len(texts)
                        text_bytes += sum(len(text.encode("utf-8")) for text in texts)

                        if documents >= next_progress:
                            print(
                                f"{split}: {documents:,}/{expected['kept_documents']:,}"
                                f" документов, {total_tokens:,} токенов,"
                                f" {time.monotonic() - started:.0f} с",
                                flush=True,
                            )
                            next_progress = documents + 100_000
                print(f"Готово: {split}/{path.name}", flush=True)

    if documents != expected["kept_documents"]:
        raise RuntimeError(f"Число документов {split} не совпало со stats.json")
    if text_bytes != expected["kept_text_bytes"]:
        raise RuntimeError(f"Объём текста {split} не совпал со stats.json")
    if temporary_tokens.stat().st_size != total_tokens * 2:
        raise RuntimeError(f"Неверный размер файла токенов: {split}")
    if temporary_offsets.stat().st_size != (documents + 1) * 8:
        raise RuntimeError(f"Неверный размер файла границ: {split}")
    temporary_tokens.replace(token_path)
    temporary_offsets.replace(offset_path)

    return {
        "documents": documents,
        "text_bytes": text_bytes,
        "tokens": total_tokens,
        "boundary_tokens": 2 * documents,
        "body_tokens": total_tokens - 2 * documents,
        "document_tokens": {
            "min": min(lengths),
            "mean": total_tokens / documents,
            "median": quantile(lengths, 0.5),
            "p95": quantile(lengths, 0.95),
            "max": max(lengths),
        },
        "tokens_file": token_path.name,
        "offsets_file": offset_path.name,
        "tokens_file_bytes": token_path.stat().st_size,
        "offsets_file_bytes": offset_path.stat().st_size,
        "tokens_sha256": token_digest.hexdigest(),
        "offsets_sha256": offset_digest.hexdigest(),
        "seconds": time.monotonic() - started,
    }


def main(
    input_dir=PROCESSED_DIR,
    tokenizer_dir=DEFAULT_TOKENIZER_DIR,
    output_dir=None,
    batch_size=512,
    splits=SPLITS,
):
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Нужен непустой список различных выборок")
    if output_dir is None and input_dir.resolve() != PROCESSED_DIR.resolve():
        raise ValueError("Для другого корпуса укажи отдельный --output-dir")
    if array("H").itemsize != 2 or array("Q").itemsize != 8:
        raise RuntimeError("Нужны 2-байтовый unsigned short и 8-байтовый unsigned long long")
    input_stats_path = input_dir / "stats.json"
    processing_stats = json.loads(input_stats_path.read_text(encoding="utf-8"))
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    training_config = json.loads(
        (tokenizer_dir / "training_config.json").read_text(encoding="utf-8")
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vocabulary = tokenizer.get_vocab()
    if not vocabulary or min(vocabulary.values()) < 0 or max(vocabulary.values()) >= 65_536:
        raise ValueError("ID токенов не помещаются в uint16")
    if tokenizer.get_vocab_size() != training_config["vocab_size_actual"]:
        raise ValueError("Размер словаря не совпал с training_config.json")
    special_ids = {
        token: tokenizer.token_to_id(token)
        for token in ("<pad>", "<bos>", "<eos>")
    }
    if None in special_ids.values() or len(set(special_ids.values())) != 3:
        raise ValueError("Нужны три различных специальных токена")
    if special_ids != training_config["special_token_ids"]:
        raise ValueError("Специальные ID не совпали с training_config.json")

    split_files = {}
    for split in splits:
        files = sorted((input_dir / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Нет Parquet-файлов: {input_dir / split}")
        expected = processing_stats["splits"][split]["kept_documents"]
        if expected <= 0 or sum(pq.read_metadata(p).num_rows for p in files) != expected:
            raise ValueError(f"Число строк {split} не совпало со stats.json")
        split_files[split] = files

    if output_dir is None:
        output_dir = TOKENIZED_ROOT / tokenizer_dir.name
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Каталоги исходных текстов и токенов должны различаться")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    metadata_path.unlink(missing_ok=True)
    report = {
        "format": "concatenated_documents",
        "document_layout": "<bos> + text_ids + <eos>",
        "token_dtype": "<u2",
        "offset_dtype": "<u8",
        "offset_unit": "tokens",
        "offset_layout": "N+1 positions; document i is tokens[offsets[i]:offsets[i+1]]",
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": special_ids,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "tokenizer_training_config": training_config,
        "input_dir": str(input_dir.resolve()),
        "input_stats_sha256": file_sha256(input_stats_path),
        "cleaning_config": processing_stats["config"],
        "batch_size": batch_size,
        "split_names": list(splits),
        "versions": {
            "python": platform.python_version(),
            "pyarrow": pyarrow.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "splits": {},
    }

    print(f"Токенизатор: {tokenizer_path.resolve()}", flush=True)
    print("Документы: <bos> + text_ids + <eos>; без обрезания и паддинга", flush=True)
    for split, files in split_files.items():
        report["splits"][split] = encode_split(
            split, files, output_dir, tokenizer,
            processing_stats["splits"][split], batch_size,
        )

    temporary = metadata_path.with_name(metadata_path.name + ".part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)

    print("\nИтог токенизации:")
    for split, stats in report["splits"].items():
        lengths = stats["document_tokens"]
        print(
            f"{split}: {stats['documents']:,} документов, {stats['tokens']:,} токенов,"
            f" {stats['tokens_file_bytes'] / 10**9:.3f} ГБ ID"
        )
        print(
            f"  Длина с BOS/EOS: среднее {lengths['mean']:.1f},"
            f" медиана {lengths['median']:.1f}, p95 {lengths['p95']:.1f},"
            f" максимум {lengths['max']:,}"
        )
    print(f"\nКаталог: {output_dir.resolve()}")
    print(f"Метаданные: {metadata_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--tokenizer-dir", type=Path,
        default=DEFAULT_TOKENIZER_DIR,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    args = parser.parse_args()
    main(args.input_dir, args.tokenizer_dir, args.output_dir, args.batch_size, args.splits)
