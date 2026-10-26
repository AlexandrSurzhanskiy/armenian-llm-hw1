# Armenian GPT

**Небольшая LLM на армянском языке, обученная с нуля.** Домашнее задание №1: от очистки корпуса и своего BPE до претрейна, экспериментов и оценки.

**97.2M параметров · контекст 1024 · BPE 16k · H100 80 GB**

[Инференс](notebooks/02_inference.ipynb) · [Обучение](notebooks/01_reproduce.ipynb) · [Результаты](notebooks/03_results.ipynb) · [Устройство проекта](docs/architecture.md)

---

## Главное

Финальная модель прошла две стадии: претрейн на ArmWeb, затем продолжение на смеси **70% ArmWeb / 30% армянской Wikipedia**. Веса и токенизатор обучены самостоятельно. Предобученные компоненты не использовались; финальный checkpoint не проходил SFT.

| Модель | ArmWeb IID · PPL ↓ | ArmWeb tail · PPL ↓ | Wikipedia · PPL ↓ |
| :--- | ---: | ---: | ---: |
| 35M · 1 эпоха | 18.11 | 18.25 | — |
| 97M · 2 эпохи | 13.22 | 13.52 | 23.95 |
| **97M + смесь** | **13.16** | **13.44** | **13.07** |

На Wikipedia perplexity снизилась на **45.4%**. Свободные ответы остаются слабым местом: **6/24** верных, **0/8** фактов с правильным ответом при всех трёх формулировках. В отчёте отдельно разобраны связность, повторы и фактические ошибки.

## Навигация

| Задача | Где смотреть |
| :--- | :--- |
| Загрузить модель, посмотреть примеры, задать свой запрос | [02 · Inference](notebooks/02_inference.ipynb) |
| Воспроизвести data → tokenizer → train | [01 · Reproduce](notebooks/01_reproduce.ipynb) |
| Сравнить модели, токенизаторы, смесь и SFT | [03 · Results](notebooks/03_results.ipynb) |
| Найти исходные числовые таблицы | [reports/results.json](reports/results.json) |
| Разобраться с корпусами и очисткой | [Данные](docs/data.md) |
| Повторить запуск на сервере | [Запуск и воспроизведение](docs/running.md) |

## Структура

```text
src/armenian_llm/
├── data/            # ArmWeb, Wikipedia, SFT, токенизация и смесь
├── tokenization/    # выборка для BPE, обучение и анализ словаря
├── modeling/        # decoder-only GPT
├── training/        # pretrain, continued pretrain, SFT, AdamW
├── inference/       # авторегрессионная генерация
├── evaluation/      # held-out loss, факты и сравнение генераций
├── experiments/     # размеры моделей, batch benchmark, smoke check
├── reporting/       # таблицы, графики, экспорт модели и отчётов
├── paths.py         # единые пути к данным и артефактам
└── cli.py           # команды этапов
notebooks/           # обучение, инференс, итоговый отчёт
configs/             # правила очистки, вопросы, конфигурации запусков
assets/tokenizer/    # обученный BPE и частоты токенов
reports/             # метрики, HTML-версия отчёта и рисунки
docs/                # архитектура, данные, запуск и оценка
tests/               # импорты, пути, snapshots и совместимость GPT
```

## Быстрый старт

Установка **из корня репозитория** в Jupyter:

```python
%pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
%pip install -e ".[notebooks,reports,logging]"
```

На исходном сервере большие файлы хранятся в `/data/armenian_llm`. Задай путь до импорта пакета:

```python
import os
os.environ["LLM_STORAGE_ROOT"] = "/data/armenian_llm"
```

Команды доступны через пакет, независимо от расположения отдельных исходников:

```python
%run -m armenian_llm --help
%run -m armenian_llm evaluate-facts --help
%run -m armenian_llm export-model
```

Только просмотр результатов: открой [ноутбук отчёта](notebooks/03_results.ipynb). Его рисунки и таблицы уже сохранены. Полный повтор обучения описан в [01_reproduce.ipynb](notebooks/01_reproduce.ipynb) и [инструкции запуска](docs/running.md).

## Архитектура и эксперименты

Decoder-only Transformer: **12 слоёв, ширина 768, 12 голов**, SwiGLU, pre-RMSNorm, RoPE и общая матрица входного embedding и выхода. Обучение — AdamW, FP32-параметры с BF16 autocast и Flash Attention через PyTorch SDPA.

Помимо финальной модели сохранены результаты BPE 8k / 16k / 32k, моделей 35M / 97M, пилота 222M, подбора батча, контрольного продолжения на новостях и SFT. Бюджеты моделей различались; 222M остановлена на пилоте. Подробности и ограничения — в [оценке](docs/evaluation.md).

## Веса

**Checkpoint ещё находится на GPU-сервере.** Скрипт `armenian-llm export-model` собирает `best.pt`, токенизатор, код и SHA256 в архив. Его нужно прикрепить к [Release этого репозитория](https://github.com/AlexandrSurzhanskiy/armenian-llm-hw1/releases).

После скачивания каталог модели размещается в `LLM_STORAGE_ROOT/artifacts/checkpoints/arm_gpt_97m_cpt_mix/`. Загрузка показана в [ноутбуке инференса](notebooks/02_inference.ipynb). Существующие checkpoint совместимы с пакетной архитектурой; экспорт поддерживает прежний и новый форматы снимков кода.
