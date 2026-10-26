"""Смесь ArmWeb и Википедии: воспроизводимая выборка окон по источникам.

%run -m armenian_llm.data.mixture
Этот запуск сохраняет план смеси и читает один CPU-батч; обучение отдельно.
"""

from collections import Counter
from fractions import Fraction
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from armenian_llm.paths import ARTIFACTS_DIR, DEFAULT_TOKENIZED_DIR, DEFAULT_TOKENIZER_DIR, TOKENIZED_ROOT
from armenian_llm.training.utils import file_sha256, load_metadata, open_token_stream


SOURCE_NAMES = ("armweb", "wikipedia")
WIKIPEDIA_DIR = TOKENIZED_ROOT / "wikipedia_hy_20231101" / "bpe_16000_500000"
DEFAULT_PLAN = ARTIFACTS_DIR / "data_mixture" / "news70_wiki30" / "plan.json"


class MixedWindowSampler:
    """Доля задаётся по окнам одинаковой длины, поэтому равна доле токенов.

    Сначала считаем накопленную квоту Википедии. Остальные окна — ArmWeb.
    Внутри каждого источника проходим случайную перестановку всех его окон,
    затем создаём следующую перестановку. batch(step) зависит только от шага
    и настроек: при возобновлении состояние загрузчика восстанавливать не надо.
    """

    def __init__(self, window_counts, batch_size=128, wikipedia_fraction=0.3, seed=42):
        self.window_counts = {name: int(window_counts[name]) for name in SOURCE_NAMES}
        self.batch_size = batch_size
        self.fraction = Fraction(str(wikipedia_fraction))
        self.seed = seed
        self._orders = {}
        if batch_size < 1 or any(count < 1 for count in self.window_counts.values()):
            raise ValueError("Нужны положительный батч и непустые источники")
        if not 0 <= self.fraction <= 1 or seed < 0:
            raise ValueError("Доля должна быть от 0 до 1, seed — неотрицательный")

    def counts_before(self, step):
        """Сколько окон каждого источника использовано до шага step (с нуля)."""
        if step < 0:
            raise ValueError("Шаг должен быть неотрицательным")
        total = step * self.batch_size
        wiki = total * self.fraction.numerator // self.fraction.denominator
        return {"armweb": total - wiki, "wikipedia": wiki}

    def _take(self, source_id, first, count):
        name = SOURCE_NAMES[source_id]
        windows = self.window_counts[name]
        pieces = []
        while count:
            cycle, offset = divmod(first, windows)
            cached = self._orders.get(name)
            if cached is None or cached[0] != cycle:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, source_id, cycle]))
                cached = (cycle, rng.permutation(windows))
                self._orders[name] = cached
            take = min(count, windows - offset)
            pieces.append(cached[1][offset:offset + take])
            first += take
            count -= take
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    def batch(self, step):
        before, after = self.counts_before(step), self.counts_before(step + 1)
        sources, indices = [], []
        for source_id, name in enumerate(SOURCE_NAMES):
            count = after[name] - before[name]
            sources.append(np.full(count, source_id, dtype=np.int64))
            indices.append(self._take(source_id, before[name], count))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, len(SOURCE_NAMES), step]))
        order = rng.permutation(self.batch_size)
        return np.concatenate(sources)[order], np.concatenate(indices)[order]


def load_corpora(news_dir, wikipedia_dir, tokenizer_dir, context_length):
    corpora = {}
    for name, directory in zip(SOURCE_NAMES, (news_dir, wikipedia_dir)):
        directory = Path(directory)
        metadata = load_metadata(directory, tokenizer_dir)
        corpora[name] = {
            "directory": directory, "metadata": metadata,
            "train": open_token_stream(directory, metadata, "train", context_length),
            "validation": open_token_stream(directory, metadata, "validation", context_length),
        }
    first, second = (corpora[name]["metadata"] for name in SOURCE_NAMES)
    for key in ("tokenizer_sha256", "vocab_size", "special_token_ids"):
        if first[key] != second[key]:
            raise ValueError(f"Несовместимые корпуса: {key}")
    return corpora


def make_mixed_batch(streams, source_ids, window_indices, context_length, device):
    """Окно остаётся внутри одного источника; между документами есть BOS/EOS."""
    source_ids = np.asarray(source_ids, dtype=np.int64)
    window_indices = np.asarray(window_indices, dtype=np.int64)
    if context_length < 1 or source_ids.ndim != 1 or source_ids.size == 0:
        raise ValueError("Нужны положительный контекст и непустой батч")
    if source_ids.shape != window_indices.shape or source_ids.min() < 0 or source_ids.max() >= len(SOURCE_NAMES):
        raise ValueError("Некорректные источники или индексы окон")
    windows = np.empty((len(source_ids), context_length + 1), dtype=np.int64)
    for source_id, name in enumerate(SOURCE_NAMES):
        positions = np.flatnonzero(source_ids == source_id)
        if not len(positions):
            continue
        tokens = streams[name]
        starts = window_indices[positions] * context_length
        if starts.min() < 0 or starts.max() + context_length >= len(tokens):
            raise ValueError(f"Окно выходит за границы {name}")
        for position, start in zip(positions, starts):
            windows[position] = tokens[int(start):int(start) + context_length + 1]
    windows = torch.from_numpy(windows)
    x, y = windows[:, :-1].contiguous(), windows[:, 1:].contiguous()
    if device.type == "cuda":
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def main(news_dir=DEFAULT_TOKENIZED_DIR, wikipedia_dir=WIKIPEDIA_DIR,
         tokenizer_dir=DEFAULT_TOKENIZER_DIR, output_path=DEFAULT_PLAN,
         wikipedia_fraction=0.3, context_length=1024, batch_size=128, seed=42):
    if context_length < 1:
        raise ValueError("Контекст должен быть положительным")
    corpora = load_corpora(news_dir, wikipedia_dir, tokenizer_dir, context_length)
    window_counts = {name: (len(corpus["train"]) - 1) // context_length
                     for name, corpus in corpora.items()}
    sampler = MixedWindowSampler(window_counts, batch_size, wikipedia_fraction, seed)
    raw_fraction = len(corpora["wikipedia"]["train"]) / sum(len(c["train"]) for c in corpora.values())
    print(f"Доля Википедии при простом объединении: {raw_fraction:.2%}")
    print(f"Выбранная смесь по целевым токенам: ArmWeb {1 - wikipedia_fraction:.0%}; Википедия {wikipedia_fraction:.0%}")
    print(f"Контекст: {context_length}; effective batch: {batch_size}; токенов/шаг: {batch_size * context_length:,}")
    print("\nИсточники (train и validation остаются раздельными):")
    source_info = {}
    for name, corpus in corpora.items():
        print(f"{name}: train {len(corpus['train']):,} токенов; validation {len(corpus['validation']):,} токенов")
        source_info[name] = {
            "data_dir": str(corpus["directory"].resolve()),
            "metadata_sha256": file_sha256(corpus["directory"] / "metadata.json"),
            "tokenizer_sha256": corpus["metadata"]["tokenizer_sha256"],
            "train_tokens": len(corpus["train"]), "validation_tokens": len(corpus["validation"]),
            "train_windows": window_counts[name],
            "discarded_train_tail_targets": (len(corpus["train"]) - 1) % context_length,
        }
    print("\nПервые пять эффективных батчей:")
    for step in range(5):
        source_ids, _ = sampler.batch(step)
        counts = Counter(source_ids.tolist())
        print(f"шаг {step + 1}: ArmWeb {counts[0]}; Википедия {counts[1]}")
    source_ids, indices = sampler.batch(0)
    streams = {name: corpus["train"] for name, corpus in corpora.items()}
    x, y = make_mixed_batch(streams, source_ids, indices, context_length, torch.device("cpu"))
    assert torch.equal(x[:, 1:], y[:, :-1])
    print(f"Батч x/y: {tuple(x.shape)} / {tuple(y.shape)}; сдвиг на один токен: OK")
    plan = {
        "wikipedia_fraction": wikipedia_fraction, "context_length": context_length,
        "effective_batch_size": batch_size, "seed": seed, "sources": source_info,
        "sampling": "cumulative integer quotas; shuffled nonoverlapping windows without replacement per source and cycle; shuffled source positions per step",
        "validation": "each source separately; test splits are not used by this loader",
        "code_sha256": file_sha256(Path(__file__)), "numpy_version": np.__version__,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(f"\nПлан смеси: {output_path}\nОбучение не запускалось.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--wikipedia-dir", type=Path, default=WIKIPEDIA_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--wikipedia-fraction", type=float, default=0.3)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.news_dir, args.wikipedia_dir, args.tokenizer_dir, args.output_path,
         args.wikipedia_fraction, args.context_length, args.batch_size, args.seed)
