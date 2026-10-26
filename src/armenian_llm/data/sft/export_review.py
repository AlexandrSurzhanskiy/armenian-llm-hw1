"""Экспорт коротких SFT-кандидатов для содержательного ручного разбора.

%run -m armenian_llm.data.sft.export_review
Небольшой файл создаётся рядом со скриптами, чтобы скачать его из Jupyter.
Отбор по длине не означает, что пары прошли проверку качества.
"""

from pathlib import Path
import argparse
import json

from armenian_llm.data.sft.inspect import file_sha256
from armenian_llm.paths import DATA_DIR, PROJECT_DIR


def main(output_path=DATA_DIR / "analysis" / "sft_aya_dolly_qa" / "short_sft_candidates.jsonl"):
    source_path = DATA_DIR / "sft" / "aya_dolly_qa" / "train.jsonl"
    with source_path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    candidates = []
    for row in rows:
        length = row["answer_tokens_including_eos"] - 1
        if 1 <= length <= 24:
            candidates.append({
                "id": row["id"], "source_ids": row["source_ids"], "source_split": "train",
                "question": row["question"], "answer": row["answer"],
                "answer_tokens_without_eos": length,
            })
    if not candidates:
        raise ValueError("Коротких кандидатов не найдено")
    output_path = Path(output_path)
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Для экспорта нужен отдельный файл")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".jsonl.part")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates), encoding="utf-8")
    temporary.replace(output_path)
    report = {
        "status": "candidates_pending_review", "source_split": "train",
        "source_file": str(source_path.resolve()), "source_sha256": file_sha256(source_path),
        "source_examples": len(rows), "exported_examples": len(candidates),
        "selection": "1–24 answer tokens without EOS; original train order",
        "export_file": str(output_path.resolve()), "export_sha256": file_sha256(output_path),
        "script_sha256": file_sha256(Path(__file__)),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    temporary = metadata_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    print(f"Экспортировано кандидатов: {len(candidates):,}")
    print(f"Размер: {output_path.stat().st_size / 10**6:.2f} МБ")
    print(f"Файл для скачивания: {output_path.resolve()}")
    print(f"Метаданные экспорта: {metadata_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=DATA_DIR / "analysis" / "sft_aya_dolly_qa" / "short_sft_candidates.jsonl")
    args = parser.parse_args()
    main(args.output_path)
