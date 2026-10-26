"""Перенос data/ и artifacts/ во внешнее хранилище с совместимостью старых путей.

Запускать после завершения ячеек, которые записывают данные или артефакты.
При копировании между файловыми системами сверяем SHA-256 до удаления оригинала.
Повторный запуск продолжает перенос; отличающиеся файлы не перезаписываются.
"""

from pathlib import Path
import argparse
import errno
import hashlib
import os
import shutil
import tempfile

from paths import PROJECT_DIR, STORAGE_ROOT


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identical(first, second):
    return (
        first.stat().st_size == second.stat().st_size
        and file_sha256(first) == file_sha256(second)
    )


def preflight(source, destination):
    if destination.is_symlink():
        raise ValueError(f"Вместо каталога назначения найдена ссылка: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Назначение не является каталогом: {destination}")
    if source.is_symlink():
        if source.resolve() != destination.resolve() or not destination.is_dir():
            raise ValueError(f"Источник уже является ссылкой на другой путь: {source}")
        return None
    if not source.exists():
        return []
    if not source.is_dir():
        raise ValueError(f"Источник не является каталогом: {source}")

    files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink() or target.is_symlink():
            raise ValueError(f"Внутри дерева найдена символическая ссылка: {relative}")
        if path.is_dir():
            if target.exists() and not target.is_dir():
                raise ValueError(f"Конфликт каталога и файла: {target}")
        elif path.is_file():
            if target.exists() and (not target.is_file() or not identical(path, target)):
                raise FileExistsError(f"В назначении уже есть другой файл: {target}")
            files.append((relative, path.stat().st_size))
        else:
            raise ValueError(f"Неподдерживаемый тип файла: {path}")
    return files


def transfer_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or not identical(source, destination):
            raise FileExistsError(f"Конфликт файлов: {destination}")
        source.unlink()
        return "existing"

    try:
        # Внутри одной файловой системы rename сохраняет содержимое без копирования.
        source.rename(destination)
        return "rename"
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise

    before = source.stat()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".transfer-", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        if before.st_size >= 100 * 10**6:
            print(f"Копирование и сверка: {source.name} ({before.st_size / 10**9:.3f} ГБ)", flush=True)
        with os.fdopen(descriptor, "wb") as output:
            with source.open("rb") as input_file:
                while block := input_file.read(8 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Источник изменился во время переноса: {source}")
        if temporary.stat().st_size != before.st_size or file_sha256(temporary) != digest.hexdigest():
            raise RuntimeError(f"Не прошла сверка содержимого: {source}")
        shutil.copystat(source, temporary)
        if destination.exists():
            raise FileExistsError(f"Во время переноса появился файл назначения: {destination}")
        temporary.replace(destination)
        source.unlink()
        return "copy_verified"
    finally:
        temporary.unlink(missing_ok=True)


def move_tree(source, destination, files):
    if files is None:
        print(f"Уже перенесено: {source} -> {destination}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(size for _, size in files)
    print(f"\n{source} -> {destination}: {len(files):,} файлов, {total_bytes / 10**9:.3f} ГБ", flush=True)
    processed_bytes = 0
    next_progress = 500 * 10**6
    for number, (relative, size) in enumerate(files, start=1):
        transfer_file(source / relative, destination / relative)
        processed_bytes += size
        if processed_bytes >= next_progress or number == len(files):
            print(
                f"  Обработано {number:,}/{len(files):,} файлов,"
                f" {processed_bytes / 10**9:.3f}/{total_bytes / 10**9:.3f} ГБ",
                flush=True,
            )
            next_progress = processed_bytes + 500 * 10**6

    # Сохраняем и пустые каталоги; удаляем только освободившиеся папки источника.
    if source.exists():
        for path in sorted(source.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError(f"Во время переноса появился новый файл: {path}")
            (destination / path.relative_to(source)).mkdir(parents=True, exist_ok=True)
            path.rmdir()
        source.rmdir()
    source.symlink_to(destination, target_is_directory=True)
    print(f"Ссылка для старого кода: {source} -> {source.resolve()}", flush=True)


def main(project_dir=PROJECT_DIR, storage_root=STORAGE_ROOT):
    project_dir = project_dir.resolve()
    storage_root = storage_root.expanduser().resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Нет каталога проекта: {project_dir}")
    if storage_root == project_dir or project_dir in storage_root.parents:
        raise ValueError("Хранилище должно находиться вне каталога с кодом")
    if storage_root in project_dir.parents:
        raise ValueError("Корень хранилища не должен быть родительским каталогом проекта")

    moves = [
        (project_dir / "data", storage_root / "data"),
        (project_dir / "artifacts", storage_root / "artifacts"),
    ]
    # Проверяем оба дерева до начала переноса, включая конфликты существующих файлов.
    plans = [(source, destination, preflight(source, destination)) for source, destination in moves]
    for source, destination, files in plans:
        move_tree(source, destination, files)

    print("\nПеренос завершён:")
    print(f"Код и ноутбуки: {project_dir}")
    print(f"Данные: {storage_root / 'data'}")
    print(f"Токенизаторы, отчёты и будущие чекпоинты: {storage_root / 'artifacts'}")
    print("Содержимое файлов, включая JSON-метаданные, не изменялось.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--storage-root", type=Path, default=STORAGE_ROOT)
    args = parser.parse_args()
    main(args.project_dir, args.storage_root)
