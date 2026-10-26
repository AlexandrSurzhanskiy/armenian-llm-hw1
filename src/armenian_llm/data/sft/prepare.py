"""Подготовка коротких армянских QA-пар для SFT нашей модели.

В labels вопрос и BOS замаскированы значением -100. Ответ и EOS остаются
целями обучения. В train затем подаем x=input_ids[:-1], y=labels[1:].
"""

from collections import Counter, defaultdict
import hashlib
import html
import json
import random
import re
from statistics import mean, median
import unicodedata

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from armenian_llm.paths import DATA_DIR, DEFAULT_TOKENIZER_DIR, FACTS_CONFIG


CONFIG = {
    "seed": 42, "validation_fraction": 0.1, "test_fraction": 0.1,
    "max_question_chars": 800, "max_answer_chars": 600,
    "min_armenian_share": 0.8, "max_sequence_tokens": 1024,
    "max_answer_tokens_including_eos": 128,
}
PROMPT_TEMPLATE = "Հարց՝ {question}\nՊատասխան՝"
# Исключаем целиком сущности контрольных фактов, включая склонения по префиксу.
CONTROL_STEMS = [
    "հայաստան", "վրաստան", "ֆրանսիա", "իտալիա", "երևան", "թբիլիսի", "թիֆլիս",
    "փարիզ", "հռոմ", "թումանյան", "րաֆֆ", "ռաֆֆ", "աբովյան", "պուշկին",
    "անուշ", "սամվել", "օնեգին", "մադրիդ", "իսպանիա", "շեքսպիր", "համլետ",
    "armenia", "georgia", "france", "italy", "yerevan", "erevan", "tbilisi", "tiflis",
    "paris", "rome", "tumanyan", "toumanian", "raffi", "abovyan", "abovian", "pushkin",
    "anush", "samvel", "onegin", "madrid", "spain", "shakespeare", "hamlet",
]
SUBJECTIVE_STEMS = [
    "լավագույն", "ամենալավ", "ամենասիրելի", "սիրելի", "ամենագեղեցիկ", "կարծիք",
    "best", "favorite", "favourite", "opinion",
]


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", html.unescape(text))).strip()


def comparison_key(text):
    text = unicodedata.normalize("NFC", text).casefold().replace("եւ", "և")
    text = re.sub(r"[^\w\s]|_", "", text)
    return re.sub(r"\s+", " ", text).strip()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def armenian_share(text):
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return None
    return sum("\u0530" <= char <= "\u058f" for char in letters) / len(letters)


def main():
    raw_dir = DATA_DIR / "raw" / "aya_dolly"
    raw_path = raw_dir / "armenian.parquet"
    output_dir = DATA_DIR / "sft" / "aya_dolly_qa"
    tokenizer_path = DEFAULT_TOKENIZER_DIR / "tokenizer.json"
    benchmark_path = FACTS_CONFIG
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    control_terms = set(CONTROL_STEMS)
    for pool in benchmark["candidate_pools"].values():
        for candidate in pool:
            control_terms.update(candidate["aliases"])
    control_terms = sorted({comparison_key(term) for term in control_terms}, key=lambda term: (-len(term), term))
    control_pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(term) for term in control_terms) + ")")
    subjective_pattern = re.compile(r"(?<!\w)(?:" + "|".join(SUBJECTIVE_STEMS) + ")")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    bos_id, eos_id = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    if bos_id is None or eos_id is None:
        raise ValueError("В токенизаторе отсутствуют BOS/EOS")
    rows = pq.read_table(raw_path).to_pylist()
    rejected = Counter()
    control_matches = Counter()
    groups = defaultdict(list)

    for row in rows:
        question, answer = row.get("inputs"), row.get("targets")
        if not isinstance(question, str) or not isinstance(answer, str):
            rejected["missing_text"] += 1
            continue
        question, answer = normalize(question), normalize(answer)
        if not (("՞" in question or "?" in question)
                and 5 <= len(question) <= CONFIG["max_question_chars"]
                and 1 <= len(answer) <= CONFIG["max_answer_chars"]):
            rejected["qa_format_or_char_length"] += 1
            continue
        key = comparison_key(question)
        if "context:" in question.casefold() or any(
            marker in key for marker in ("համատեքստ", "ըստ տեքստի", "տրված տեքստ", "վերոնշյալ", "հետևյալ տեքստ")
        ):
            rejected["requires_context"] += 1
            continue
        q_share, a_share = armenian_share(question), armenian_share(answer)
        if (q_share is None or q_share < CONFIG["min_armenian_share"]
                or (a_share is not None and a_share < CONFIG["min_armenian_share"])
                or (a_share is None and not any(char.isdigit() for char in answer))):
            rejected["language_or_empty_answer"] += 1
            continue
        if any(token in question + answer for token in ("<pad>", "<bos>", "<eos>")):
            rejected["special_token_literal"] += 1
            continue
        if subjective_pattern.search(key):
            rejected["subjective_question"] += 1
            continue
        match = control_pattern.search(comparison_key(question + " " + answer))
        if match:
            rejected["control_entity"] += 1
            control_matches[match.group()] += 1
            continue
        groups[key].append({"source_id": row["id"], "question": question, "answer": answer})

    prepared = []
    for key, group in groups.items():
        if len({row["answer"].casefold().replace("եւ", "և") for row in group}) != 1:
            rejected["conflicting_question"] += len(group)
            continue
        rejected["duplicate_question"] += len(group) - 1
        row = group[0]
        prompt = PROMPT_TEMPLATE.format(question=row["question"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        full_ids = tokenizer.encode(prompt + " " + row["answer"], add_special_tokens=False).ids
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Токенизация пересекла границу вопрос/ответ: source_id={row['source_id']}")
        answer_ids = full_ids[len(prompt_ids):] + [eos_id]
        input_ids = [bos_id] + full_ids + [eos_id]
        if len(answer_ids) > CONFIG["max_answer_tokens_including_eos"]:
            rejected["answer_token_length"] += 1
            continue
        if len(input_ids) > CONFIG["max_sequence_tokens"]:
            rejected["sequence_token_length"] += 1
            continue
        masked = 1 + len(prompt_ids)
        prepared.append({
            "id": hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "source_ids": [item["source_id"] for item in group],
            "question": row["question"], "answer": row["answer"], "prompt": prompt,
            "input_ids": input_ids, "labels": [-100] * masked + answer_ids,
            "prompt_tokens_including_bos": masked, "answer_tokens_including_eos": len(answer_ids),
        })

    if len(prepared) < 10:
        raise ValueError(f"После фильтров осталось слишком мало пар: {len(prepared)}")
    random.Random(CONFIG["seed"]).shuffle(prepared)
    n_val = int(len(prepared) * CONFIG["validation_fraction"])
    n_test = int(len(prepared) * CONFIG["test_fraction"])
    splits = {
        "validation": prepared[:n_val],
        "test": prepared[n_val:n_val + n_test],
        "train": prepared[n_val + n_test:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": CONFIG, "prompt_template": PROMPT_TEMPLATE, "ignore_index": -100,
        "labels": "unshifted; use x=input_ids[:-1], y=labels[1:]; only answer and EOS supervised",
        "tokenizer_sha256": file_sha256(tokenizer_path), "vocab_size": tokenizer.get_vocab_size(),
        "source": json.loads((raw_dir / "download_info.json").read_text(encoding="utf-8")),
        "source_armenian_parquet_sha256": file_sha256(raw_path),
        "benchmark_sha256": file_sha256(benchmark_path), "control_terms": control_terms,
        "control_matches": dict(control_matches), "subjective_stems": SUBJECTIVE_STEMS,
        "input_rows": len(rows), "kept_rows": len(prepared), "rejections": dict(rejected),
        "split_method": "shuffle normalized-question groups with seed 42; 80/10/10",
        "limitations_ru": [
            "Отбор QA, языка, субъективности и контрольных сущностей основан на эвристиках; истинность каждого ответа не проверена.",
            "Уникальные нормализованные вопросы разделены между SFT-выборками; семантические переформулировки автоматически не кластеризовались.",
            "Пересечения с претрейном не исключались. Контрольные сущности удалены по записанному списку префиксов и вариантов написания.",
        ],
        "splits": {},
    }
    print(f"Исходных пар: {len(rows):,}; после очистки: {len(prepared):,}")
    print("\nИсключено (первая подходящая причина):")
    for reason, count in rejected.items():
        print(f"  {reason}: {count:,}")
    print("\nВыборки SFT:")
    for split, records in splits.items():
        path = output_dir / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.part")
        with temporary.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
        lengths = [len(record["input_ids"]) for record in records]
        answer_lengths = [record["answer_tokens_including_eos"] for record in records]
        metadata["splits"][split] = {
            "file": path.name, "sha256": file_sha256(path), "examples": len(records),
            "sequence_tokens_mean": mean(lengths), "sequence_tokens_median": median(lengths),
            "sequence_tokens_max": max(lengths), "answer_tokens_mean": mean(answer_lengths),
            "supervised_tokens": sum(answer_lengths),
        }
        print(f"{split}: {len(records):,} пар; средняя длина {mean(lengths):.1f}; токенов ответа с EOS {mean(answer_lengths):.1f}")

    metadata_path = output_dir / "metadata.json"
    temporary = metadata_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    example = splits["train"][0]
    print("\nПример train:")
    print("Вопрос:", example["question"])
    print("Ответ:", example["answer"])
    print("Позиций с label=-100:", example["prompt_tokens_including_bos"])
    print("Целевых токенов с EOS:", example["answer_tokens_including_eos"])
    print("Текст целевых токенов:", tokenizer.decode(
        [label for label in example["labels"] if label != -100], skip_special_tokens=False
    ))
    print(f"\nКаталог: {output_dir}")
    print(f"Метаданные: {metadata_path}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
