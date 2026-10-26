"""Сопоставление коротких армянских кандидатов с английскими исходниками.

%run /workdir/compare_sft_originals.py

Результат предназначен для ручного разбора. Совпадение позиции в языковом
блоке — кандидат на соответствие переводов, а не подтверждение его качества.
Ни одна пара автоматически не исправляется и не добавляется в SFT.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from paths import DATA_DIR, PROJECT_DIR


AYA_REVISION = "09069079fad96ad9c7f8781be8c1d70471dfa768"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
# Проверены по публичным строкам указанного снимка Aya; это ID, не индексы Dolly.
LANGUAGE_ROWS = 14808
ENGLISH_FIRST_ID = 444241
ARMENIAN_FIRST_ID = 666361
ID_SHIFT = ARMENIAN_FIRST_ID - ENGLISH_FIRST_ID
PREVIEW_IDS = (670523, 673477, 669226, 669593)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize(text):
    # Некоторые исходные поля содержат буквальные \n вместо перевода строки.
    text = unicodedata.normalize("NFC", text).replace(r"\n", "\n")
    return re.sub(r"\s+", " ", text).strip()


def dolly_input(row):
    context = row.get("context", "")
    return row["instruction"] + ("\nContext:" + context if context else "")


def load_english(aya_dir, english_path):
    if english_path is not None:
        rows = read_jsonl(english_path)
    else:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        info = json.loads((aya_dir / "download_info.json").read_text(encoding="utf-8"))
        if info.get("revision") != AYA_REVISION:
            raise ValueError("Версия локального Aya отличается от версии сопоставления")
        files = sorted((aya_dir / "translated_dolly").glob("*.parquet"))
        if not files:
            raise FileNotFoundError("Не найдены исходные translated_dolly/*.parquet")
        rows = []
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=4096, columns=["id", "inputs", "targets", "language"]):
                table = pa.Table.from_batches([batch])
                rows.extend(table.filter(pc.equal(table["language"], "eng")).to_pylist())
            print(f"Прочитан {path.name}; английских строк: {len(rows):,}", flush=True)
        if len(rows) != LANGUAGE_ROWS:
            raise ValueError(f"Ожидалось {LANGUAGE_ROWS} английских строк, получено {len(rows)}")
    rows.sort(key=lambda row: int(row["id"]))
    if len({int(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Повторяющиеся английские ID")
    if any(row["language"] != "eng" or not ENGLISH_FIRST_ID <= int(row["id"]) < ENGLISH_FIRST_ID + LANGUAGE_ROWS for row in rows):
        raise ValueError("Неожиданный язык или диапазон ID; позиционное сопоставление неприменимо")
    return rows


def main(args):
    candidates = read_jsonl(args.candidates)
    english = load_english(args.aya_dir, args.english_aya)
    if not args.dolly.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="databricks/databricks-dolly-15k", repo_type="dataset",
            revision=DOLLY_REVISION, filename="databricks-dolly-15k.jsonl",
            local_dir=args.dolly.parent, token=False,
        )
    dolly = read_jsonl(args.dolly)
    index = defaultdict(list)
    for line_number, row in enumerate(dolly, start=1):
        index[(normalize(dolly_input(row)), normalize(row["response"]))].append(
            {"line_1based": line_number, **row})
    by_id = {int(row["id"]): row for row in english}
    paired = []
    for candidate in candidates:
        sources = []
        for source_id in candidate["source_ids"]:
            source_id = int(source_id)
            if not ARMENIAN_FIRST_ID <= source_id < ARMENIAN_FIRST_ID + LANGUAGE_ROWS:
                raise ValueError(f"Неожиданный армянский ID: {source_id}")
            row = by_id.get(source_id - ID_SHIFT)
            matches = [] if row is None else index[(normalize(row["inputs"]), normalize(row["targets"]))]
            sources.append({
                "armenian_source_id": source_id,
                "aya_english_id": source_id - ID_SHIFT,
                "alignment_status": "positional_candidate_requires_manual_review" if row else "english_unavailable",
                "english_input": row["inputs"] if row else None,
                "english_answer": row["targets"] if row else None,
                "dolly_exact_matches": matches,
            })
        paired.append({**candidate, "english_sources": sources})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "paired_candidates.jsonl", paired)
    write_jsonl(args.output_dir / "aya_english.jsonl", english)
    counts = Counter()
    for row in paired:
        sources = row["english_sources"]
        counts["with_any_english"] += any(s["english_input"] is not None for s in sources)
        counts["with_all_english"] += all(s["english_input"] is not None for s in sources)
        counts["with_any_exact_dolly_match"] += any(s["dolly_exact_matches"] for s in sources)
    report = {
        "candidates": len(candidates), "english_aya_rows": len(english), "original_dolly_rows": len(dolly),
        **counts, "aya_revision": AYA_REVISION, "dolly_revision": DOLLY_REVISION,
        "candidate_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "dolly_sha256": hashlib.sha256(args.dolly.read_bytes()).hexdigest(),
        "alignment_method": "Armenian source ID minus pinned language-block shift; proposed links require manual bilingual review",
        "dolly_match_method": "Exact input+context and answer after NFC/whitespace/literal-newline normalization; all duplicate matches retained",
        "status": "reference_pairs_for_review_only",
        "limitations_ru": [
            "Позиционное соответствие не доказывает смысловую эквивалентность перевода.",
            "Совпадение с Dolly не подтверждает истинность исходного ответа.",
            "Отсутствие точного совпадения не означает ошибку: тексты могли измениться при подготовке Aya.",
            "Армянские тексты, прежняя разметка и обучающие выборки не изменены.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nКандидатов: {len(candidates):,}; английских строк Aya: {len(english):,}; строк Dolly: {len(dolly):,}")
    print(f"Есть английский кандидат: {counts['with_any_english']:,}; есть точное совпадение с Dolly: {counts['with_any_exact_dolly_match']:,}")
    print("Это исходники для ручной сверки, а не автоматически очищенный набор.")
    print("\nЧетыре специально выбранных примера — не случайная выборка:")
    for source_id in PREVIEW_IDS:
        for row in paired:
            source = next((s for s in row["english_sources"] if s["armenian_source_id"] == source_id), None)
            if source is None:
                continue
            print(f"\nsource_id={source_id}")
            print("HY вопрос:", row["question"])
            print("HY ответ:", row["answer"])
            print("EN вопрос:", source["english_input"])
            print("EN ответ:", source["english_answer"])
            print("Точных совпадений в Dolly:", len(source["dolly_exact_matches"]))
    print(f"\nВсе пары: {args.output_dir / 'paired_candidates.jsonl'}")
    print(f"Сводка: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=PROJECT_DIR / "short_sft_candidates.jsonl")
    parser.add_argument("--aya-dir", type=Path, default=DATA_DIR / "raw" / "aya_dolly")
    parser.add_argument("--english-aya", type=Path, default=None, help="Необязательный готовый JSONL английского Aya; допустим неполный кэш")
    parser.add_argument("--dolly", type=Path, default=DATA_DIR / "raw" / "dolly_original" / "databricks-dolly-15k.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "analysis" / "sft_aya_dolly_originals")
    main(parser.parse_args())
