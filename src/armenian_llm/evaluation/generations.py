"""Генерации исходной 97M и её продолжения на смеси, с одинаковыми настройками.

%run -m armenian_llm.evaluation.generations
Вопросы и численные QA-метрики оцениваем следующим отдельным шагом.
"""

from pathlib import Path
import argparse
import json

import torch
from tokenizers import Tokenizer

from armenian_llm.inference.generate import generate_text
from armenian_llm.inference import generate as generation_module
from armenian_llm.modeling.gpt import GPT, GPTConfig
from armenian_llm.paths import ARTIFACTS_DIR, CHECKPOINTS_DIR, DEFAULT_TOKENIZER_DIR
from armenian_llm.training.utils import file_sha256


PROMPTS = [
    ("Հայաստանի մայրաքաղաքը", "Столица Армении"),
    ("Այսօր Երևանում", "Сегодня в Ереване"),
    ("Հայ գրականության պատմության մեջ", "В истории армянской литературы"),
    ("Հայաստանի հավաքականը", "Сборная Армении"),
]
SEEDS = (2026, 2027)
GENERATION = {"max_new_tokens": 80, "temperature": 0.8, "top_k": 50}


def show_training_summary(run_dir):
    path = run_dir / "progress.json"
    if not path.exists():
        print("Финальная сводка progress.json не найдена; ниже покажем шаг выбранного чекпоинта.", flush=True)
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(f"Сохранённая сводка смеси: {summary['status']}; шаг {summary['step']:,}/{summary['total_steps']:,}", flush=True)
    print(f"Дополнительных токенов: {summary['tokens_seen']:,}", flush=True)
    for name, tokens in summary["tokens_by_source"].items():
        print(f"  {name}: {tokens:,}", flush=True)
    print("Validation на последнем сохранённом шаге:", flush=True)
    for name, loss in summary["last_validation_losses"].items():
        initial = summary["initial_validation_losses"][name]
        print(f"  {name}: {initial:.4f} → {loss:.4f}", flush=True)
    return summary


def main(base_checkpoint=CHECKPOINTS_DIR / "arm_gpt_97m" / "best.pt",
         mix_checkpoint=CHECKPOINTS_DIR / "arm_gpt_97m_cpt_mix" / "best.pt",
         tokenizer_dir=DEFAULT_TOKENIZER_DIR,
         output_dir=ARTIFACTS_DIR / "evaluation" / "generation_comparison_97m_cpt_mix",
         device_name="cuda"):
    base_checkpoint, mix_checkpoint, tokenizer_dir, output_dir = map(
        Path, (base_checkpoint, mix_checkpoint, tokenizer_dir, output_dir))
    training_summary = show_training_summary(mix_checkpoint.parent)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Для этого запуска нужны CUDA и BF16")
    elif device.type != "cpu":
        raise ValueError("Поддерживаются cuda и cpu")
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_hash = file_sha256(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    checkpoints = {"arm_gpt_97m": base_checkpoint, "arm_gpt_97m_cpt_mix": mix_checkpoint}
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    temporary_path = predictions_path.with_suffix(".jsonl.part")
    model_info, predictions = {}, {}
    base_config = None
    print(f"\nУстройство: {device}; генерация: {GENERATION}; seeds: {SEEDS}", flush=True)
    with temporary_path.open("w", encoding="utf-8") as output:
        for name, path in checkpoints.items():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["fingerprints"]["tokenizer_sha256"] != tokenizer_hash:
                raise ValueError(f"Токенизатор не совпал с {name}")
            checkpoint_hash = file_sha256(path)
            if base_config is None:
                base_config = checkpoint["model_config"]
            elif checkpoint["model_config"] != base_config:
                raise ValueError("Архитектуры сравниваемых моделей различаются")
            if name == "arm_gpt_97m_cpt_mix":
                if checkpoint.get("variant") != "mix":
                    raise ValueError("Ожидался чекпоинт продолжения на смеси")
                if checkpoint["fingerprints"]["initial_checkpoint_sha256"] != model_info["arm_gpt_97m"]["checkpoint_sha256"]:
                    raise ValueError("Смесь обучалась из другого исходного чекпоинта")
            info = {"checkpoint": str(path.resolve()), "checkpoint_sha256": checkpoint_hash,
                    "step": checkpoint["step"], "tokens_seen": checkpoint["tokens_seen"],
                    "model_config": checkpoint["model_config"]}
            if "last_validation_losses" in checkpoint:
                info["validation_losses"] = checkpoint["last_validation_losses"]
            model = GPT(GPTConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            model = model.to(device).eval()
            model_info[name] = info
            print(f"\n{name}: выбран {path.name}, шаг {info['step']:,}", flush=True)
            if "validation_losses" in info:
                print(f"Validation выбранного чекпоинта: {info['validation_losses']}", flush=True)
            for prompt, translation in PROMPTS:
                for seed in SEEDS:
                    completion = generate_text(model, tokenizer, prompt, device, seed=seed, **GENERATION)
                    row = {"model": name, "prompt": prompt, "prompt_ru": translation,
                           "seed": seed, "completion": completion, "text": prompt + completion}
                    predictions[(name, prompt, seed)] = row
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Сохранено генераций: {len(PROMPTS) * len(SEEDS)}", flush=True)
    temporary_path.replace(predictions_path)
    report = {"status": "complete", "models": model_info, "generation": GENERATION,
              "prompts": [{"text": text, "translation_ru": ru} for text, ru in PROMPTS],
              "seeds": list(SEEDS), "tokenizer_sha256": tokenizer_hash,
              "training_summary": training_summary, "torch_version": str(torch.__version__),
              "device": str(device), "script_sha256": file_sha256(Path(__file__)),
              "generator_sha256": file_sha256(Path(generation_module.__file__))}
    metadata_path = output_dir / "metadata.json"
    temporary = metadata_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    print("\nСРАВНЕНИЕ ГЕНЕРАЦИЙ", flush=True)
    for prompt, translation in PROMPTS:
        print(f"\nНачало: {prompt} ({translation})", flush=True)
        for seed in SEEDS:
            print(f"\nseed={seed}", flush=True)
            for name in checkpoints:
                print(f"\n{name}:\n{predictions[(name, prompt, seed)]['text']}", flush=True)
    print(f"\nВсе генерации: {predictions_path}\nНастройки и чекпоинты: {metadata_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, default=CHECKPOINTS_DIR / "arm_gpt_97m" / "best.pt")
    parser.add_argument("--mix-checkpoint", type=Path, default=CHECKPOINTS_DIR / "arm_gpt_97m_cpt_mix" / "best.pt")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "evaluation" / "generation_comparison_97m_cpt_mix")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.base_checkpoint, args.mix_checkpoint, args.tokenizer_dir, args.output_dir, args.device)
