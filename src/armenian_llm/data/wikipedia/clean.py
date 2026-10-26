"""Первый вариант очистки только на выборке из 5000 статей Википедии.

Запуск: %run -m armenian_llm.data.wikipedia.clean
Исходный корпус и sample.jsonl не меняются. Статьи пока не отбрасываем.
"""

from collections import Counter
import hashlib
import json
import random
import re
from statistics import median

from armenian_llm.paths import DATA_DIR
from armenian_llm.data.armweb.prepare import armenian_share, normalize_text, repetition_fraction
from armenian_llm.data.wikipedia.inspect import HEADING_PATTERN, SERVICE_HEADINGS, URL_PATTERN


# Удаляем артефакты вроде «Название (), ...» и «Имя (, ), ...».
# Скобки внутри f() и самостоятельные () этот шаблон не затрагивает.
EMPTY_PARENS_PATTERN = re.compile(r"[ \t]+\([ \t,;՝։]*\)(?=[,՝։])")
CONFIG = {
    "version": "wikipedia_basic_v1",
    "normalization": "html.unescape, Unicode NFC, whitespace normalization",
    "cut_from_first_service_heading": list(SERVICE_HEADINGS),
    "service_heading_matching": "whole line, after whitespace normalization",
    "empty_parentheses_pattern": EMPTY_PARENS_PATTERN.pattern,
    "document_filtering": False,
}


def clean_text(original):
    text = normalize_text(original)
    heading = HEADING_PATTERN.search(text)
    changes = {
        "normalized": text != (original or ""),
        "service_heading": heading.group().strip() if heading else None,
        "service_start": heading.start() if heading else None,
        "removed_tail_characters": len(text) - heading.start() if heading else 0,
    }
    if heading:
        text = text[:heading.start()].rstrip()
    text, replacements = EMPTY_PARENS_PATTERN.subn("", text)
    changes["empty_parentheses_removed"] = replacements
    return text.strip(), changes


def summarize(texts):
    counts = Counter()
    shares = []
    lengths = []
    for text in texts:
        length, share = len(text), armenian_share(text)
        lengths.append(length)
        shares.append(share)
        counts["empty"] += length == 0
        counts["shorter_than_200"] += length < 200
        counts["shorter_than_500"] += length < 500
        counts["armenian_share_below_50_percent"] += share < 0.5
        counts["armenian_share_below_80_percent"] += share < 0.8
        counts["has_service_heading"] += HEADING_PATTERN.search(text) is not None
        counts["has_url"] += URL_PATTERN.search(text) is not None
        counts["has_empty_parentheses"] += re.search(r"\(\s*\)", text) is not None
        counts["has_wiki_markup"] += re.search(r"\{\{|\}\}|\[\[|\]\]", text) is not None
        counts["repetition_above_20_percent"] += repetition_fraction(text, 10) > 0.2
    return {
        "documents": len(texts), "characters": sum(lengths),
        "utf8_bytes": sum(len(text.encode("utf-8")) for text in texts),
        "median_characters": median(lengths), "median_armenian_share": median(shares),
        "counts": dict(counts),
    }


def main():
    analysis_dir = DATA_DIR / "analysis" / "wikipedia_hy_20231101"
    source_path = analysis_dir / "sample.jsonl"
    sample_report = json.loads((analysis_dir / "stats.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Выборка пуста; сначала запусти inspect_wikipedia.py")
    normalized = [normalize_text(row.get("text")) for row in rows]
    cleaned, changes = [], []
    for row in rows:
        text, change = clean_text(row.get("text"))
        cleaned.append(text)
        changes.append(change)
    before, after = summarize(normalized), summarize(cleaned)
    report = {
        "config": CONFIG,
        "source": sample_report["source"],
        "source_sample_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "sample_role": sample_report["sample_role"],
        "before": before, "after": after,
        "articles_normalized": sum(item["normalized"] for item in changes),
        "articles_with_tail_removed": sum(item["service_heading"] is not None for item in changes),
        "articles_with_empty_parentheses_removed": sum(item["empty_parentheses_removed"] > 0 for item in changes),
        "empty_parentheses_removed": sum(item["empty_parentheses_removed"] for item in changes),
        "characters_removed_after_normalization": before["characters"] - after["characters"],
        "limitations_ru": [
            "Обрезка по заголовкам является эвристикой; содержимое удаляемых хвостов нужно просмотреть на примерах.",
            "Длинные иноязычные списки, остатки шаблонов и повторяемость пока только диагностируются.",
            "Ни одна статья на этом этапе не исключается. Пустые результаты также оставлены для анализа.",
        ],
    }
    output_path = analysis_dir / "sample_basic_cleaned.jsonl"
    temporary = output_path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as output:
        for row, text, change in zip(rows, cleaned, changes):
            result = {key: row[key] for key in ("id", "url", "title")}
            result.update(text=text, cleaning=change)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    report_path = analysis_dir / "basic_cleaning_stats.json"
    temporary = report_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)

    print(f"Статей до / после: {before['documents']:,} / {after['documents']:,}")
    print(f"Обрезан служебный хвост: {report['articles_with_tail_removed']:,} статей")
    print(f"Удалены пустые скобки: {report['empty_parentheses_removed']:,} вхождений в {report['articles_with_empty_parentheses_removed']:,} статьях")
    print(f"Символов после нормализации: {before['characters']:,} → {after['characters']:,}")
    print(f"Убрано символов: {report['characters_removed_after_normalization'] / max(1, before['characters']):.2%}")
    print(f"Медианная длина: {before['median_characters']:.1f} → {after['median_characters']:.1f}")
    print(f"Медианная доля армянских букв: {before['median_armenian_share']:.2%} → {after['median_armenian_share']:.2%}")
    print("\nПризнак                                  До    После")
    for name, count in before["counts"].items():
        print(f"{name:<39} {count:>5,} {after['counts'][name]:>8,}")

    # Один большой срез, один случайный срез, один пример удаления скобок.
    tail_indices = [i for i, item in enumerate(changes) if item["service_heading"] is not None]
    chosen = []
    if tail_indices:
        chosen.append(max(tail_indices, key=lambda i: changes[i]["removed_tail_characters"] / max(1, len(normalized[i]))))
        others = [i for i in tail_indices if i not in chosen]
        if others:
            chosen.append(random.Random(43).choice(others))
    parentheses = [i for i, item in enumerate(changes) if item["empty_parentheses_removed"] and i not in chosen]
    if parentheses:
        chosen.append(random.Random(42).choice(parentheses))
    print("\nПримеры очистки:")
    for number, index in enumerate(chosen, start=1):
        row, change = rows[index], changes[index]
        print(f"\nПример {number}: {row['title']}")
        print(row["url"])
        print(f"Символов: {len(normalized[index]):,} → {len(cleaned[index]):,}")
        print("До — начало текста:")
        print(normalized[index][:450])
        if change["service_start"] is not None:
            print("Удаляемый хвост — первые 450 символов:")
            print(normalized[index][change["service_start"]:change["service_start"] + 450])
        print("После — первые 700 символов:")
        print(cleaned[index][:700] or "<пустой текст>")
    print(f"\nОчищенная выборка: {output_path}\nСтатистика: {report_path}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
