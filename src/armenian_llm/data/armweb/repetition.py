from armenian_llm.paths import DATA_DIR, RAW_DIR
from pathlib import Path
from collections import Counter
import json
import random
import re


def main():
    sample_path = (DATA_DIR / "train_sample.jsonl")
    with sample_path.open(encoding="utf-8") as f:
        sample = [json.loads(line) for line in f]

    word_pattern = re.compile(r"[^\W\d_]+")
    cases = []

    for row in sample:
        text = row["text"] or ""
        words = [m.group().lower() for m in word_pattern.finditer(text)]
        grams = [tuple(words[i:i + 10]) for i in range(len(words) - 9)]

        if not grams:
            continue

        counts = Counter(grams)
        repeat_fraction = 1 - len(counts) / len(grams)

        if repeat_fraction > 0.2:
            gram, count = counts.most_common(1)[0]
            cases.append((row, repeat_fraction, gram, count))

    print("Документов с признаком повторов:", len(cases))

    rng = random.Random(42)
    examples = rng.sample(cases, min(3, len(cases)))

    for number, (row, fraction, gram, count) in enumerate(examples, start=1):
        text = row["text"]
        matches = list(word_pattern.finditer(text))
        words = [m.group().lower() for m in matches]

        positions = [
            i for i in range(len(words) - 9)
            if tuple(words[i:i + 10]) == gram
        ]

        first = positions[0]
        second = next((i for i in positions if i >= first + 10), None)

        print(f"\nПример {number}")
        print("Источник:", row["source"])
        print("Ссылка:", row["url"])
        print("Длина:", len(text))
        print(f"Доля повторных 10-грамм: {fraction:.2%}")
        print("Самый частый фрагмент:", " ".join(gram))
        print("Число вхождений:", count)

        if second is None:
            print("Вхождения этого фрагмента перекрываются.")

        for i, position in enumerate(
            [first] if second is None else [first, second], start=1
        ):
            start = max(0, matches[position].start() - 120)
            end = matches[position + 9].end() + 120
            print(f"\nКонтекст {i}:")
            print(text[start:end])


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
