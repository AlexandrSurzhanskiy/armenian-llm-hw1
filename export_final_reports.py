"""Небольшой архив кода и сохранённых результатов для подготовки сдачи.

В ноутбуке: %run /workdir/export_final_reports.py
Используется только стандартная библиотека; веса и корпуса не читаются.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
from zipfile import ZipFile, ZIP_DEFLATED

from paths import PROJECT_DIR, STORAGE_ROOT


MAX_FILE_BYTES = 20 * 1024**2
MAX_TOTAL_BYTES = 150 * 1024**2
DATA_REPORT_NAMES = {
    "metadata.json", "stats.json", "filtering_stats.json", "cleaning_config.json",
    "sample_info.json", "download_info.json", "cleaning_report.json",
}
PROJECT_REPORT_NAMES = {"benchmark_facts.json", "cleaning_config.json"}
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".py"}
SKIP_DIRS = {".git", ".cache", "__pycache__", "tensorboard", ".ipynb_checkpoints"}


def eligible(path, root):
    return not path.is_symlink() and not any(part in SKIP_DIRS for part in path.relative_to(root).parts)


def collect_files(project_dir, storage_root):
    selected = {}
    # Верхний уровень рабочей папки: код, ноутбуки и небольшие настройки проекта.
    for path in sorted(project_dir.iterdir()):
        if path.is_file() and eligible(path, project_dir) and (
            path.suffix in {".py", ".ipynb"} or path.name in PROJECT_REPORT_NAMES
            or path.name.startswith("requirements") or path.name in {"pyproject.toml", "environment.yml"}
        ):
            selected["project/" + path.name] = path

    # Сюда входят версии кода/токенизаторов из запусков, метрики и генерации.
    artifacts = storage_root / "artifacts"
    if artifacts.is_dir():
        for path in sorted(artifacts.rglob("*")):
            if path.is_file() and path.suffix in TEXT_SUFFIXES and eligible(path, artifacts):
                selected["artifacts/" + path.relative_to(artifacts).as_posix()] = path

    # В data берём только отчёты, без текстовых выборок и токенов.
    data = storage_root / "data"
    if data.is_dir():
        for path in sorted(data.rglob("*.json")):
            if path.is_file() and path.name in DATA_REPORT_NAMES and eligible(path, data):
                selected["data_reports/" + path.relative_to(data).as_posix()] = path
    return selected


def checkpoint_inventory(storage_root):
    root = storage_root / "artifacts" / "checkpoints"
    runs = []
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                continue
            weights = {}
            for name in ("best.pt", "last.pt", "initial.pt"):
                path = directory / name
                if path.is_file() and not path.is_symlink():
                    weights[name] = {"bytes": path.stat().st_size, "path": str(path)}
            runs.append({
                "run": directory.name, "checkpoint_files": weights,
                "report_files": [name for name in ("config.json", "metrics.csv", "summary.json", "progress.json")
                                 if (directory / name).is_file()],
            })
    return runs


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "numpy", "tokenizers", "pyarrow", "huggingface-hub", "tensorboard", "matplotlib"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main(project_dir=PROJECT_DIR, storage_root=STORAGE_ROOT, output=None):
    project_dir, storage_root = Path(project_dir).resolve(), Path(storage_root).resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(project_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(output).resolve() if output else project_dir / f"final_reports_{stamp}.zip"
    if output.exists():
        raise FileExistsError(f"Архив уже существует: {output}")
    selected = collect_files(project_dir, storage_root)
    included, skipped, total = [], [], 0
    for name, path in selected.items():
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped.append({"archive_path": name, "bytes": size, "reason": "file_exceeds_20_MiB"})
        else:
            included.append({"archive_path": name, "source": str(path), "bytes": size})
            total += size
    if total > MAX_TOTAL_BYTES:
        raise ValueError(f"Выбрано {total / 1024**2:.1f} МиБ; лимит небольшого архива 150 МиБ")
    manifest = {
        "purpose": "Code and saved reports for the final homework analysis; not a standalone model release",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(project_dir), "storage_root": str(storage_root),
        "package_versions": package_versions(), "runs": checkpoint_inventory(storage_root),
        "included_files": included, "skipped_large_files": skipped,
        "uncompressed_bytes": total,
        "notes_ru": [
            "Веса не включены; наличие и размеры указаны в runs без загрузки чекпоинтов.",
            "Корпуса, токенизированные данные, кэши и события TensorBoard не включены.",
            "Файлы проекта и сохранённые результаты не изменялись.",
            "Наличие файлов и метрик само по себе не подтверждает качество или завершённость эксперимента.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.part")
    if temporary.exists():
        raise FileExistsError(f"Найден незавершённый архив: {temporary}; выбери другое имя")
    try:
        with ZipFile(temporary, "x", compression=ZIP_DEFLATED) as archive:
            for row in included:
                archive.write(row["source"], arcname=row["archive_path"])
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    groups = Counter(row["archive_path"].split("/")[0] for row in included)
    print(f"Файлов в архиве: {len(included):,} + manifest.json")
    for name, count in sorted(groups.items()):
        print(f"  {name}: {count:,}")
    print(f"Каталогов запусков: {len(manifest['runs'])}")
    print(f"Размер архива: {output.stat().st_size / 10**6:.2f} МБ")
    if skipped:
        print("Пропущены большие файлы:")
        for row in skipped:
            print(f"  {row['archive_path']}: {row['bytes'] / 10**6:.1f} МБ")
    if not groups["artifacts"]:
        print("Не найдены отчёты artifacts. Проверь storage_root перед отправкой архива.")
    print(f"\nФайл для скачивания: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--storage-root", type=Path, default=STORAGE_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    main(args.project_dir, args.storage_root, args.output)
