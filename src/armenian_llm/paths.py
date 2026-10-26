"""Пути проекта и внешнего хранилища; не зависят от каталога запуска."""

from pathlib import Path
import os


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("LLM_PROJECT_ROOT", PACKAGE_DIR.parents[1])).expanduser().resolve()
CONFIGS_DIR = PROJECT_DIR / "configs"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
REPORTS_DIR = PROJECT_DIR / "reports"
ASSETS_DIR = PROJECT_DIR / "assets"
ARMWEB_CLEANING_CONFIG = CONFIGS_DIR / "data" / "armweb_cleaning.json"
FACTS_CONFIG = CONFIGS_DIR / "evaluation" / "facts.json"
RUN_CONFIGS_DIR = CONFIGS_DIR / "runs"
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
