"""Архив финальной модели: веса, токенизатор, код и контрольные суммы."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from zipfile import ZipFile, ZIP_STORED

from armenian_llm.paths import ARTIFACTS_DIR, CHECKPOINTS_DIR


RUN_NAME = "arm_gpt_97m_cpt_mix"


def checkpoint_sources(run_dir):
    """Поддерживаем как прежний плоский snapshot, так и новый пакет."""
    sources = {
        "best.pt": run_dir / "best.pt",
        "tokenizer/tokenizer.json": run_dir / "tokenizer/tokenizer.json",
        "config.json": run_dir / "config.json",
        "progress.json": run_dir / "progress.json",
    }
    code = run_dir / "code"
    if (code / "src/armenian_llm/modeling/gpt.py").is_file():
        layout = "src_package"
        for directory, suffix in (("src", ".py"), ("configs", ".json")):
            for path in sorted((code / directory).rglob("*" + suffix)):
                sources["code/" + path.relative_to(code).as_posix()] = path
        for name in ("pyproject.toml", "README.md"):
            if (code / name).is_file():
                sources["code/" + name] = code / name
    else:
        layout = "legacy_flat"
        sources["code/model.py"] = code / "model.py"
        sources["code/generate.py"] = code / "generate.py"
    for relative in ("tokenizer/training_config.json", "tokenizer/metrics.json"):
        if (run_dir / relative).is_file():
            sources[relative] = run_dir / relative
    return sources, layout


def loading_instructions(layout):
    imports = (
        "sys.path.insert(0, str(root / 'code' / 'src'))\n"
        "from armenian_llm.modeling.gpt import GPT, GPTConfig\n"
        "from armenian_llm.inference.generate import generate_text"
        if layout == "src_package" else
        "sys.path.insert(0, str(root / 'code'))\n"
        "from model import GPT, GPTConfig\n"
        "from generate import generate_text"
    )
    return f"""# Финальная армянская GPT 97M

Веса и токенизатор обучены с нуля. После ArmWeb модель продолжила претрейн
на смеси 70% ArmWeb / 30% Wikipedia. Этот checkpoint не проходил SFT.

best.pt выбран по validation. manifest.json содержит SHA256 всех файлов.
Код из исходного запуска сохранён в code/; раскладка: {layout}.
PyTorch 2.9.1+cu128, tokenizers 0.22.1.

Запуск из распакованного каталога с best.pt:

```python
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

root = Path.cwd()
{imports}

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

Генерация может содержать смысловые противоречия и ошибки фактов.
Полный код data/tokenizer/train и ноутбуки находятся в репозитории проекта.
"""


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
    sources, layout = checkpoint_sources(run_dir)
    final_test = ARTIFACTS_DIR / "evaluation/final_97m_comparison/results.json"
    if final_test.is_file():
        sources["evaluation/final_test.json"] = final_test
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": RUN_NAME, "parameters": config["parameters"],
        "model_config": config["model"], "code_layout": layout, "files": {},
        "purpose": "Inference checkpoint; full training pipeline is submitted separately",
    }
    partial = archive_path.with_suffix(".zip.part")
    try:
        with ZipFile(partial, "w", compression=ZIP_STORED) as archive:
            for name, source in sources.items():
                digest, size = hashlib.sha256(), 0
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
            archive.writestr(f"{RUN_NAME}/README.md", loading_instructions(layout))
        partial.replace(archive_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    print("\nАрхив модели:", archive_path)
    print(f"Размер: {archive_path.stat().st_size / 1e6:.1f} МБ")
    print("SHA256 best.pt:", manifest["files"]["best.pt"]["sha256"])
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Заменить ранее собранный архив")
    main(overwrite=parser.parse_args().overwrite)
