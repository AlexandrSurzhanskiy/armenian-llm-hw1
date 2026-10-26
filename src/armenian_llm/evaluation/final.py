"""Финальная оценка исходной 97M и смеси на одинаковых отложенных текстах.

В ноутбуке: %run -m armenian_llm.evaluation.final
Считает все next-token targets, включая остаток последнего окна.
Обучение и выбор checkpoint по этим тестам не выполняются.
"""

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from armenian_llm.modeling.gpt import GPT, GPTConfig
from armenian_llm.paths import ARTIFACTS_DIR, CHECKPOINTS_DIR, DATA_DIR, DEFAULT_TOKENIZED_DIR, DEFAULT_TOKENIZER_DIR
from armenian_llm.training.pretrain import evaluate
from armenian_llm.training.utils import file_sha256, load_metadata, open_token_stream


MODELS = ("arm_gpt_97m", "arm_gpt_97m_cpt_mix")


def save_json(path, value):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def print_results(report):
    print("Модель              | Корпус    | Split     | Loss   | Perplexity", flush=True)
    for row in report["results"]:
        print(f"{row['model']:<19} | {row['corpus']:<9} | {row['split']:<9} | {row['loss']:.4f} | {row['perplexity']:.2f}", flush=True)


def export_saved_curves():
    """Небольшие CSV из серверной копии: для восстановления полных графиков."""
    result = {}
    for name in ("arm_gpt_35m", "arm_gpt_97m"):
        path = CHECKPOINTS_DIR / name / "metrics.csv"
        try:
            text = path.read_text(encoding="utf-8")
            result[name] = {"source": str(path), "sha256": file_sha256(path),
                            "rows": list(csv.DictReader(text.splitlines()))}
        except OSError as error:
            result[name] = {"source": str(path), "error_type": type(error).__name__}
    return result


def main(batch_size=32, overwrite=False):
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    output_dir = ARTIFACTS_DIR / "evaluation" / "final_97m_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    if output_path.exists() and not overwrite:
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        print("Сохранённые результаты; повторная оценка не запускалась. Для нового замера добавь --overwrite.")
        print("Статус:", saved.get("status", "unknown"))
        print_results(saved)
        print("\nРезультаты:", output_path)
        return saved
    if not torch.cuda.is_available():
        raise RuntimeError("Для этого запуска нужна GPU в текущем kernel Jupyter")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Ожидалась GPU с BF16, как в сохранённых запусках")
    device = torch.device("cuda")
    corpora = {
        "armweb": {"directory": DEFAULT_TOKENIZED_DIR, "splits": ("test_iid", "test_tail")},
        "wikipedia": {"directory": DATA_DIR / "tokenized" / "wikipedia_hy_20231101" / "bpe_16000_500000", "splits": ("test",)},
    }
    tokenizer_hash = file_sha256(DEFAULT_TOKENIZER_DIR / "tokenizer.json")
    for corpus in corpora.values():
        corpus["metadata"] = load_metadata(corpus["directory"], DEFAULT_TOKENIZER_DIR)
    report = {
        "status": "running", "gpu": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__), "batch_size": batch_size,
        "precision": "bf16_autocast_fp32_parameters", "attention": "pytorch_flash_attention",
        "tokenizer_sha256": tokenizer_hash,
        "method": "all_next_token_targets_in_nonoverlapping_context_windows_including_remainder; token-weighted mean; perplexity=exp(loss)",
        "checkpoint_selection": "existing best.pt chosen on validation before this test evaluation",
        "corpora": {name: {"directory": str(info["directory"]), "metadata_sha256": file_sha256(info["directory"] / "metadata.json"),
                           "splits": {split: {key: info["metadata"]["splits"][split][key] for key in ("documents", "tokens", "tokens_sha256")} for split in info["splits"]}}
                    for name, info in corpora.items()},
        "models": {}, "results": [], "saved_pretrain_curves": export_saved_curves(),
        "limitations_ru": [
            "Test split не использовался для выбора checkpoint в этом скрипте.",
            "Похожие тексты между корпусами могли оставаться; отсутствие near-duplicate утечек не доказано.",
            "Perplexity оценивает предсказание текста, а не истинность фактических ответов или выполнение инструкций.",
        ],
    }
    save_json(output_path, report)
    print("Устройство:", report["gpu"], flush=True)
    expected_architecture = None
    try:
        for name in MODELS:
            path = CHECKPOINTS_DIR / name / "best.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            config = checkpoint["model_config"]
            if checkpoint["fingerprints"]["tokenizer_sha256"] != tokenizer_hash:
                raise ValueError(f"{name}: другой токенизатор")
            if any(info["metadata"]["vocab_size"] != config["vocab_size"] for info in corpora.values()):
                raise ValueError(f"{name}: размер словаря не совпадает с данными")
            if expected_architecture is not None and config != expected_architecture:
                raise ValueError("Для этого сравнения ожидалась одна архитектура 97M")
            expected_architecture = config
            model = GPT(GPTConfig(**config))
            model.load_state_dict(checkpoint["model"])
            model = model.to(device).eval()
            origin = checkpoint.get("origin", {})
            report["models"][name] = {
                "checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
                "step": checkpoint["step"], "parameters": model.num_parameters(), "model_config": config,
                "stage_tokens_seen": checkpoint["tokens_seen"], "origin": origin,
                "cumulative_pretrain_tokens_seen": checkpoint["tokens_seen"] + origin.get("tokens_seen", 0),
            }
            del checkpoint
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for corpus_name, corpus in corpora.items():
                    for split in corpus["splits"]:
                        tokens = open_token_stream(corpus["directory"], corpus["metadata"], split, model.config.context_length)
                        loss = evaluate(model, tokens, batch_size, device)
                        row = {"model": name, "corpus": corpus_name, "split": split,
                               "scored_tokens": len(tokens) - 1, "loss": loss, "perplexity": math.exp(loss)}
                        report["results"].append(row)
                        save_json(output_path, report)
                        print(f"{name:<19} | {corpus_name:<9} | {split:<9} | loss {loss:.4f} | perplexity {row['perplexity']:.2f}", flush=True)
                        del tokens
            del model
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException as error:
        report["status"] = "incomplete"
        report["error_type"] = type(error).__name__
        save_json(output_path, report)
        raise
    report["status"] = "complete"
    save_json(output_path, report)
    print("\nИТОГ", flush=True)
    print_results(report)
    print("\nРезультаты:", output_path, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true", help="Повторить оценку вместо показа сохранённых результатов")
    args = parser.parse_args()
    main(args.batch_size, args.overwrite)
