"""Собрать архив финальной модели для сдачи, без запуска GPU.

В Jupyter: %run /workdir/export_submission_model.py
Источник — готовый best.pt модели 97M после смеси. Исходные файлы не меняются.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

from paths import ARTIFACTS_DIR, CHECKPOINTS_DIR


RUN_NAME = "arm_gpt_97m_cpt_mix"


def main(overwrite=False):
    run_dir = CHECKPOINTS_DIR / RUN_NAME
    destination = ARTIFACTS_DIR / "submission"
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"{RUN_NAME}.zip"
    if archive_path.exists() and not overwrite:
        print("Архив уже существует:", archive_path)
        print(f"Размер: {archive_path.stat().st_size / 1e6:.1f} МБ")
        print("Для повторной сборки добавь --overwrite.")
        return archive_path

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    if (config.get("variant") != "mix" or config["parameters"] != 97_241_856
            or progress.get("status") != "complete"):
        raise ValueError("Ожидался завершённый запуск 97M после смеси ArmWeb/Wikipedia")
    sources = {
        "best.pt": run_dir / "best.pt",
        "tokenizer/tokenizer.json": run_dir / "tokenizer" / "tokenizer.json",
        "code/model.py": run_dir / "code" / "model.py",
        "code/generate.py": run_dir / "code" / "generate.py",
        "config.json": run_dir / "config.json",
        "progress.json": run_dir / "progress.json",
    }
    for relative in ("tokenizer/training_config.json", "tokenizer/metrics.json"):
        path = run_dir / relative
        if path.exists():
            sources[relative] = path
    final_test = ARTIFACTS_DIR / "evaluation" / "final_97m_comparison" / "results.json"
    if final_test.exists():
        sources["evaluation/final_test.json"] = final_test
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": RUN_NAME, "parameters": config["parameters"],
        "model_config": config["model"], "files": {},
        "purpose": "Inference checkpoint; full training pipeline is submitted separately",
    }
    readme = """# Финальная армянская GPT 97M

Веса и токенизатор обучены с нуля. После ArmWeb модель продолжила претрейн
на смеси 70% ArmWeb / 30% Wikipedia. Этот checkpoint не проходил SFT.

best.pt — выбранный по validation checkpoint; tokenizer/tokenizer.json —
его BPE-токенизатор. Код архитектуры и генерации находится в code/.
config.json описывает запуск, manifest.json содержит SHA256 файлов архива.
Для обучения использовался PyTorch 2.9.1+cu128, tokenizers 0.22.1.

Загрузка в Python из распакованного каталога:

```python
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

root = Path.cwd()  # каталог с best.pt
sys.path.insert(0, str(root / 'code'))
from model import GPT, GPTConfig
from generate import generate_text

checkpoint = torch.load(root / 'best.pt', map_location='cpu', weights_only=True, mmap=True)
model = GPT(GPTConfig(**checkpoint['model_config']))
model.load_state_dict(checkpoint['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()
tokenizer = Tokenizer.from_file(str(root / 'tokenizer' / 'tokenizer.json'))
tokenizer.no_padding()
tokenizer.no_truncation()
del checkpoint

prompt = 'Այսօր Երևանում'
print(prompt + generate_text(model, tokenizer, prompt, device,
                            max_new_tokens=80, temperature=0.8, top_k=50, seed=2026))
```

Смысловые противоречия и ошибки фактов в генерации сохраняются.
Отдельные ноутбуки демонстрации и результатов входят в комплект отчётов.
Полный код подготовки данных, токенизатора и обучения сдаётся отдельно.
"""
    partial = archive_path.with_suffix(".zip.part")
    try:
        # ZIP_STORED быстро упаковывает большие веса без их перекодирования.
        with ZipFile(partial, "w", compression=ZIP_STORED) as archive:
            for name, source in sources.items():
                digest = hashlib.sha256()
                size = 0
                with source.open("rb") as incoming, archive.open(f"{RUN_NAME}/{name}", "w", force_zip64=True) as outgoing:
                    while block := incoming.read(8 * 1024 * 1024):
                        outgoing.write(block)
                        digest.update(block)
                        size += len(block)
                manifest["files"][name] = {"sha256": digest.hexdigest(), "bytes": size, "source": str(source)}
                print(f"Добавлен {name}: {size / 1e6:.1f} МБ", flush=True)
            expected = config["fingerprints"]["tokenizer_sha256"]
            if manifest["files"]["tokenizer/tokenizer.json"]["sha256"] != expected:
                raise ValueError("Токенизатор не совпадает с конфигурацией обучения")
            archive.writestr(f"{RUN_NAME}/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr(f"{RUN_NAME}/README.md", readme)
        partial.replace(archive_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print("\nАрхив модели:", archive_path)
    print(f"Размер: {archive_path.stat().st_size / 1e6:.1f} МБ")
    print("SHA256 best.pt:", manifest["files"]["best.pt"]["sha256"])
    print("Модель и токенизатор готовы к скачиванию через файловый браузер Jupyter.")
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Заменить ранее собранный архив")
    args = parser.parse_args()
    main(overwrite=args.overwrite)
