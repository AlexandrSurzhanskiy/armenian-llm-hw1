"""Скачивание фиксированной версии ArmWeb во внешнее хранилище."""

from pathlib import Path
import argparse

from armenian_llm.paths import RAW_DIR
from huggingface_hub import snapshot_download


REPO_ID = "COPA-AI/armweb"
REVISION = "804790c6cd783532641b76e27381f92695505ce4"


def main(output_dir=RAW_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=output_dir,
        token=False,
    )
    files = sorted(output_dir.rglob("*.parquet"))
    print(f"Каталог: {output_dir.resolve()}")
    print(f"Количество файлов Parquet: {len(files)}")
    for path in files:
        print(f"{path.relative_to(output_dir)}: {path.stat().st_size / 10**6:.1f} МБ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    main(args.output_dir)
