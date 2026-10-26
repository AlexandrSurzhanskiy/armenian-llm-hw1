"""Обучение byte-level BPE с нуля и оценка на сохранённых выборках ArmWeb."""

from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import statistics
import time

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from paths import TOKENIZER_DATA_DIR, TOKENIZERS_DIR


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


def read_batches(path, batch_size=256):
    batch = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            row = json.loads(line)
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Некорректный текст: {path}, строка {number}")
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def evaluate(tokenizer, path):
    frequencies = Counter()
    lengths = []
    characters = text_bytes = whitespace_words = 0

    for texts in read_batches(path):
        encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
        for text, encoding in zip(texts, encodings):
            ids = encoding.ids
            restored = tokenizer.decode(ids, skip_special_tokens=False)
            if restored != text:
                raise RuntimeError(f"Текст не восстановился после decode: {path}")
            lengths.append(len(ids))
            frequencies.update(ids)
            characters += len(text)
            text_bytes += len(text.encode("utf-8"))
            whitespace_words += len(text.split())

    if not lengths or not sum(lengths):
        raise ValueError(f"Пустые данные для оценки: {path}")
    total_tokens = sum(lengths)
    p95 = (
        statistics.quantiles(lengths, n=100, method="inclusive")[94]
        if len(lengths) > 1 else lengths[0]
    )
    metrics = {
        "documents": len(lengths),
        "text_bytes": text_bytes,
        "characters": characters,
        "whitespace_words": whitespace_words,
        "tokens": total_tokens,
        "tokens_per_document_mean": total_tokens / len(lengths),
        "tokens_per_document_median": statistics.median(lengths),
        "tokens_per_document_p95": p95,
        "characters_per_token": characters / total_tokens,
        "bytes_per_token": text_bytes / total_tokens,
        "tokens_per_whitespace_word": (
            total_tokens / whitespace_words if whitespace_words else None
        ),
        "roundtrip_errors": 0,
    }
    return metrics, frequencies


def run(data_dir, vocab_size=16_000, min_frequency=2, output_dir=None):
    if vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("Словарь должен вмещать 256 байтов и специальные токены")
    if min_frequency < 1:
        raise ValueError("min_frequency должна быть положительной")

    with (data_dir / "sample_info.json").open(encoding="utf-8") as f:
        sample_info = json.load(f)
    train_documents = sample_info["splits"]["train"]["sample_documents"]
    paths = {
        split: data_dir / f"{split}.jsonl"
        for split in ("train", "validation")
    }
    fingerprints = {split: file_sha256(path) for split, path in paths.items()}
    if output_dir is None:
        output_dir = (
            TOKENIZERS_DIR / f"bpe_{vocab_size}_{train_documents}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.unlink(missing_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"Обучаем BPE: {train_documents:,} документов, словарь {vocab_size:,}", flush=True)
    started = time.monotonic()
    tokenizer.train_from_iterator(
        read_batches(paths["train"]), trainer=trainer, length=train_documents
    )
    training_seconds = time.monotonic() - started

    temporary = output_dir / "tokenizer.json.part"
    tokenizer.save(str(temporary))
    tokenizer_path = output_dir / "tokenizer.json"
    temporary.replace(tokenizer_path)
    tokenizer.model.save(str(output_dir))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    probes = [
        "Հայաստանը գտնվում է Հարավային Կովկասում։",
        " Բարև\n\nաշխարհ։ ",
        "English Русский العربية 🙂",
        "<bos>Բարև<eos>",
        "Cafe\u0301",
        "",
    ]
    for text in probes:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise RuntimeError("Не прошла проверка обратимости токенизации")

    evaluation = {}
    frequencies = {}
    for split, path in paths.items():
        print(f"Оцениваем токенизатор: {split}", flush=True)
        evaluation[split], frequencies[split] = evaluate(tokenizer, path)
        expected = sample_info["splits"][split]["sample_documents"]
        if evaluation[split]["documents"] != expected:
            raise RuntimeError(f"Число документов {split} не совпало с sample_info.json")
        expected_bytes = sample_info["splits"][split]["sample_text_bytes"]
        if evaluation[split]["text_bytes"] != expected_bytes:
            raise RuntimeError(f"Объём текста {split} не совпал с sample_info.json")

    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    vocabulary = tokenizer.get_vocab()
    ordinary_ids = set(vocabulary.values()) - set(special_ids.values())
    unused = sum(frequencies["train"][i] == 0 for i in ordinary_ids)
    rare = sum(0 < frequencies["train"][i] < 10 for i in ordinary_ids)

    with (output_dir / "token_frequencies.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["id", "token", "is_special", "train_count", "validation_count"])
        for token, index in sorted(vocabulary.items(), key=lambda item: item[1]):
            writer.writerow([
                index, token, index in special_ids.values(),
                frequencies["train"][index], frequencies["validation"][index],
            ])

    config = {
        "algorithm": "byte_level_bpe",
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "initial_alphabet_size": 256,
        "add_prefix_space": False,
        "special_token_ids": special_ids,
        "tokenizers_version": tokenizers.__version__,
        "python_version": platform.python_version(),
        "data_sha256": fingerprints,
        "sample_info": sample_info,
    }
    with (output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    report = {
        "training_seconds": training_seconds,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "train_sample": evaluation["train"],
        "validation": evaluation["validation"],
        "unused_tokens_on_train_sample": unused,
        "tokens_seen_1_to_9_times_on_train_sample": rare,
        "probes_roundtrip_errors": 0,
    }
    temporary_metrics = output_dir / "metrics.json.part"
    with temporary_metrics.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    temporary_metrics.replace(metrics_path)

    val = evaluation["validation"]
    print("\nИтог обучения токенизатора:")
    print(f"Словарь: {tokenizer.get_vocab_size():,} токенов")
    print(f"Время обучения: {training_seconds:.1f} с")
    print("Специальные токены:", special_ids)
    print(f"Validation: {val['documents']:,} документов")
    print(f"Токенов на документ, среднее: {val['tokens_per_document_mean']:.1f}")
    print(f"Токенов на документ, медиана: {val['tokens_per_document_median']:.1f}")
    print(f"Токенов на документ, p95: {val['tokens_per_document_p95']:.1f}")
    print(f"Символов на токен: {val['characters_per_token']:.3f}")
    print(f"Байтов UTF-8 на токен: {val['bytes_per_token']:.3f}")
    print(f"Токенов на слово по пробелам: {val['tokens_per_whitespace_word']:.3f}")
    print(f"Не использованы в обучающей выборке: {unused:,} обычных токенов")
    print(f"Встречаются 1–9 раз: {rare:,} обычных токенов")
    print("Ошибок восстановления текста: 0")
    example = probes[0]
    encoding = tokenizer.encode(example, add_special_tokens=False)
    print("\nПример:", example)
    print("ID токенов:", encoding.ids)
    print("После decode:", tokenizer.decode(encoding.ids, skip_special_tokens=False))
    print(f"\nКаталог: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=TOKENIZER_DATA_DIR)
    parser.add_argument("--vocab-size", type=int, default=16_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.vocab_size, args.min_frequency, args.output_dir)
