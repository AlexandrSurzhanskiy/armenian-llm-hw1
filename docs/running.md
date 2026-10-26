# Запуск и воспроизведение

## Установка

Из корня репозитория в Jupyter:

```python
%pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
%pip install -e ".[notebooks,reports,logging]"
```

Исходное окружение: Python 3.12, PyTorch 2.9.1+cu128, tokenizers 0.22.1, NumPy 2.4.2, PyArrow 20.0.0. Для обучения нужна CUDA с BF16. Отчёт и упаковка архива работают на CPU.

Установка editable связывает пакет с `src/`; после переноса отдельных файлов импорты остаются одинаковыми. Если kernel уже импортировал другую версию пакета или библиотек, перезапусти его.

## Хранилище

Задай путь до импорта `armenian_llm.paths`:

```python
import os
os.environ["LLM_STORAGE_ROOT"] = "/data/armenian_llm"
```

Внутри него:

```text
data/
├── raw/
├── processed/
└── tokenized/
artifacts/
├── tokenizers/
├── checkpoints/
├── evaluation/
└── submission/
```

По умолчанию путь — `../data/armenian_llm` относительно репозитория. Конфигурации и пути к notebook/report определяются от корня проекта; текущая директория kernel на них не влияет.

## Полный повтор

[01_reproduce.ipynb](../notebooks/01_reproduce.ipynb) вызывает этапы последовательно:

1. ArmWeb: скачать → проанализировать → очистить.
2. Выборка 500k → обучение BPE 16k → токенизация.
3. Претрейн 97M: 2 эпохи, effective batch 128, warmup 400, LR 6e-4 → 6e-5.
4. Wikipedia: скачать → проверить очистку на выборке → очистить весь корпус → токенизировать.
5. План смеси 70/30 → продолжение на 5000 шагов, warmup 100, LR 1e-4 → 1e-5.
6. Held-out оценка, QA и экспорт.

Это полный повтор загрузки данных и обучения. Для просмотра уже полученных результатов используется [03_results.ipynb](../notebooks/03_results.ipynb).

Команда отдельного этапа:

```python
%run -m armenian_llm download-armweb
%run -m armenian_llm train-tokenizer --help
%run -m armenian_llm continue-pretrain --variant mix --stop-at-step 5000
```

Аргументы этапов передаются без изменений. Из обычной командной строки те же команды начинаются с `armenian-llm` или `python -m armenian_llm`.

## Инференс и веса

Открой [02_inference.ipynb](../notebooks/02_inference.ipynb). Модель ищется в:

```text
LLM_STORAGE_ROOT/artifacts/checkpoints/arm_gpt_97m_cpt_mix/
├── best.pt
└── tokenizer/tokenizer.json
```

Экспорт существующего checkpoint на сервере:

```python
%run -m armenian_llm export-model
```

Архив: `LLM_STORAGE_ROOT/artifacts/submission/arm_gpt_97m_cpt_mix.zip`. После публикации он будет доступен в GitHub Release. Внутри сохраняются код именно исходного запуска, метаданные и SHA256. Поддерживаются прежние плоские snapshots и новый пакет.

## Отчёт и проверки

Пересборка рисунков из уже опубликованных таблиц, без обучения:

```python
from armenian_llm.paths import REPORTS_DIR
%run -m armenian_llm report --from-summary {REPORTS_DIR}/results.json
```

Проверки после установки пакета, из корня репозитория:

```python
!python -m unittest discover -s tests -v
```

Они проверяют импорты, независимость путей от cwd, снимки кода и совместимость GPT с небольшим checkpoint прежней реализации. Корпус и GPU не требуются.
