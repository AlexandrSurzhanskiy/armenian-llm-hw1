"""Сохранение пакета и конфигураций рядом с новым результатом эксперимента."""

import hashlib
from pathlib import Path
import shutil

from armenian_llm.paths import CONFIGS_DIR, PACKAGE_DIR, PROJECT_DIR


def snapshot_code(destination):
    """Копируем src-layout целиком, чтобы snapshot сохранял рабочие импорты."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    package_copy = destination / "src" / "armenian_llm"
    shutil.copytree(PACKAGE_DIR, package_copy, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(CONFIGS_DIR, destination / "configs", dirs_exist_ok=True)
    shutil.copy2(PROJECT_DIR / "pyproject.toml", destination / "pyproject.toml")
    shutil.copy2(PROJECT_DIR / "README.md", destination / "README.md")
    paths = sorted(package_copy.rglob("*.py")) + sorted((destination / "configs").rglob("*.json"))
    paths.append(destination / "pyproject.toml")
    paths.append(destination / "README.md")
    return {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
