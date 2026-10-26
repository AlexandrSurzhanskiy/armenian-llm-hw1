"""Мягкая фильтрация очищенной выборки Википедии, без обработки всего корпуса.

Запуск: %run -m armenian_llm.data.wikipedia.filter
При низкой доле армянского сначала пытаемся сохранить армянские блоки.
Отсеянные статьи сохраняются отдельно вместе с причиной.
"""

from collections import Counter
import hashlib
import json
import re
import unicodedata

from armenian_llm.data.wikipedia.clean import summarize
from armenian_llm.paths import DATA_DIR
from armenian_llm.data.armweb.prepare import WORD_PATTERN, armenian_share, repetition_fraction


CONFIG = {
    "version": "wikipedia_filter_v1",
    "min_characters": 50,
    "min_armenian_letters": 20,
    "min_armenian_share": 0.5,
    "salvage_min_words": 3,
    "ngram_size": 10,
    "max_repetition_fraction": 0.2,
    "deduplication": "exact SHA256 of cleaned UTF-8 text within the sample",
}


def armenian_letters(text):
    return sum(count for char, count in Counter(text).items()
               if char.isalpha() and unicodedata.name(char, "").startswith("ARMENIAN"))


def salvage_armenian_blocks(text, config):
    def suitable(block):
        return (armenian_share(block) >= config["min_armenian_share"]
                and len(WORD_PATTERN.findall(block)) >= config["salvage_min_words"])

    kept = []
    for paragraph in re.split(r"\n[ \t]*\n", text):
        if suitable(paragraph):
            kept.append(paragraph.strip())
        elif armenian_share(paragraph) < config["min_armenian_share"]:
            # Если абзац смешанный, сохраняем подходящие строки по отдельности.
            lines = [line.strip() for line in paragraph.splitlines() if suitable(line)]
            if lines:
                kept.append("\n".join(lines))
    return "\n\n".join(kept)


def filter_text(text, config=None):
    """Вход уже прошёл clean_wikipedia.clean_text; возвращаем текст, причину и метрики."""
    config = config or CONFIG
    before_share = armenian_share(text)
    attempted = before_share < config["min_armenian_share"]
    original_length = len(text)
    if attempted:
        text = salvage_armenian_blocks(text, config)
    details = {
        "armenian_share_before": before_share, "armenian_share_after": armenian_share(text),
        "language_salvage_attempted": attempted,
        "characters_before": original_length, "characters_after": len(text),
        "armenian_letters": armenian_letters(text),
        "repetition_fraction": repetition_fraction(text, config["ngram_size"]),
    }
    reason = None
    if not text:
        reason = "empty_after_language_salvage" if attempted else "empty"
    elif len(text) < config["min_characters"]:
        reason = "too_short"
    elif details["armenian_letters"] < config["min_armenian_letters"]:
        reason = "too_few_armenian_letters"
    elif details["armenian_share_after"] < config["min_armenian_share"]:
        reason = "language"
    elif details["repetition_fraction"] > config["max_repetition_fraction"]:
        reason = "repetition"
    return text, reason, details


def write_jsonl(path, rows):
    temporary = path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    analysis_dir = DATA_DIR / "analysis" / "wikipedia_hy_20231101"
    source_path = analysis_dir / "sample_basic_cleaned.jsonl"
    basic_report = json.loads((analysis_dir / "basic_cleaning_stats.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept, rejected, reasons = [], [], Counter()
    seen = {}
    for row in rows:
        text, reason, details = filter_text(row["text"])
        if reason is None:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            details["text_sha256"] = digest
            if digest in seen:
                reason = "exact_duplicate"
                details["duplicate_of_id"] = seen[digest]
            else:
                seen[digest] = row["id"]
        if reason is not None:
            reasons[reason] += 1
            rejected.append({**row, "candidate_text": text, "filtering": details, "reason": reason})
        else:
            kept.append({**row, "text": text, "filtering": details})
    if not kept:
        raise ValueError("После фильтров не осталось статей")

    accepted_path = analysis_dir / "sample_filtered.jsonl"
    rejected_path = analysis_dir / "sample_rejected.jsonl"
    write_jsonl(accepted_path, kept)
    write_jsonl(rejected_path, rejected)
    before = basic_report["after"]
    after = summarize([row["text"] for row in kept])
    recovered = [row for row in kept if row["filtering"]["language_salvage_attempted"]]
    report = {
        "config": CONFIG, "basic_cleaning_config": basic_report["config"],
        "source": basic_report["source"], "sample_role": basic_report["sample_role"],
        "source_sample_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "input_documents": len(rows), "kept_documents": len(kept), "rejections": dict(reasons),
        "language_salvage_attempts": sum(row["filtering"]["language_salvage_attempted"] for row in kept + rejected),
        "kept_after_language_salvage": len(recovered), "before": before, "after": after,
        "limitations_ru": [
            "Фильтры являются эвристиками и не проверяют фактическую истинность или полноту статьи.",
            "При извлечении армянских блоков из смешанного текста часть контекста может потеряться.",
            "Дедупликация на этом шаге выполняется только внутри случайной выборки.",
        ],
    }
    report_path = analysis_dir / "filtering_stats.json"
    temporary = report_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)

    print(f"Статей до фильтра: {len(rows):,}; после: {len(kept):,}")
    print("Исключено — первая подходящая причина:")
    for reason, count in reasons.items():
        print(f"  {reason}: {count:,}")
    print(f"Попыток сохранить армянские блоки: {report['language_salvage_attempts']}; сохранено статей после этого: {len(recovered)}")
    print(f"Медианная длина: {before['median_characters']:.1f} → {after['median_characters']:.1f}")
    print(f"Сохранено символов относительно базовой очистки: {after['characters'] / max(1, before['characters']):.2%}")
    print("\nПризнак                                  До    После")
    for name, count in before["counts"].items():
        print(f"{name:<39} {count:>5,} {after['counts'][name]:>8,}")

    print("\nДва примера сохранённых армянских блоков:")
    for row in sorted(recovered, key=lambda row: row["filtering"]["armenian_share_before"])[:2]:
        details = row["filtering"]
        print(f"\n{row['title']}\n{row['url']}")
        print(f"Армянских букв: {details['armenian_share_before']:.2%} → {details['armenian_share_after']:.2%}")
        print(f"Символов: {details['characters_before']:,} → {details['characters_after']:,}")
        print(row["text"][:900])

    print("\nОтсеянные примеры — по одному на причину:")
    shown = set()
    for row in rejected:
        if row["reason"] in shown:
            continue
        shown.add(row["reason"])
        print(f"\n{row['title']} | {row['reason']}\n{row['url']}")
        print(f"Повторяемость: {row['filtering']['repetition_fraction']:.2%}")
        print(row["text"][:500])
    print(f"\nСохранённые: {accepted_path}\nОтсеянные: {rejected_path}\nСтатистика: {report_path}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
