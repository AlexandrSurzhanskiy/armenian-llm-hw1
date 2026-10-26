"""Все большие файлы проекта хранятся вне каталога с кодом.

На сервере: /workdir/*.py и /data/armenian_llm/{data,artifacts}.
Пути считаются от этого файла, поэтому не зависят от текущей папки ноутбука.
При необходимости корень задаётся через LLM_STORAGE_ROOT до импорта модулей.
"""

from pathlib import Path
import os


PROJECT_DIR = Path(__file__).resolve().parent
configured_root = os.environ.get("LLM_STORAGE_ROOT")
if configured_root:
    STORAGE_ROOT = Path(configured_root).expanduser()
    if not STORAGE_ROOT.is_absolute():
        STORAGE_ROOT = PROJECT_DIR / STORAGE_ROOT
else:
    STORAGE_ROOT = PROJECT_DIR.parent / "data" / "armenian_llm"
STORAGE_ROOT = STORAGE_ROOT.resolve()

DATA_DIR = STORAGE_ROOT / "data"
ARTIFACTS_DIR = STORAGE_ROOT / "artifacts"
RAW_DIR = DATA_DIR / "raw" / "armweb"
PROCESSED_DIR = DATA_DIR / "processed" / "armweb"
TOKENIZER_DATA_DIR = DATA_DIR / "tokenizer"
TOKENIZER_500K_DATA_DIR = DATA_DIR / "tokenizer_500k"
TOKENIZERS_DIR = ARTIFACTS_DIR / "tokenizers"
TOKENIZED_ROOT = DATA_DIR / "tokenized"
DEFAULT_TOKENIZER_DIR = TOKENIZERS_DIR / "bpe_16000_500000"
DEFAULT_TOKENIZED_DIR = TOKENIZED_ROOT / "bpe_16000_500000"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
