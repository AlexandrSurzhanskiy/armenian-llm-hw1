"""Оценка лучших чекпоинтов 35M и 97M на test_iid и test_tail.

Считаем loss на всех next-token targets тем же способом, что на validation:
неперекрывающиеся окна контекста и последний короткий остаток. Веса не меняются.
"""

import argparse
import gc
import json
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from armenian_llm.modeling.gpt import GPT, GPTConfig
from armenian_llm.paths import ARTIFACTS_DIR, CHECKPOINTS_DIR, DEFAULT_TOKENIZED_DIR, DEFAULT_TOKENIZER_DIR
from armenian_llm.training.pretrain import evaluate
from armenian_llm.training.utils import file_sha256, load_metadata, open_token_stream


def main(batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    device = torch.device("cuda")
    metadata = load_metadata(DEFAULT_TOKENIZED_DIR, DEFAULT_TOKENIZER_DIR)
    fingerprints = {
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "data_metadata_sha256": file_sha256(DEFAULT_TOKENIZED_DIR / "metadata.json"),
    }
    output_dir = ARTIFACTS_DIR / "evaluation" / "heldout_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    report = {
        "status": "running",
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "precision": "bf16_autocast_fp32_parameters",
        "batch_size": batch_size,
        "fingerprints": fingerprints,
        "method": "all_next_token_targets_in_nonoverlapping_context_windows_including_remainder",
        "results": [],
    }

    def save_report():
        temporary = output_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_path)

    save_report()
    print("Устройство:", report["gpu"], flush=True)
    print("Модель         | Split     | Loss   | Perplexity", flush=True)
    for name in ("arm_gpt_35m", "arm_gpt_97m"):
        checkpoint_path = CHECKPOINTS_DIR / name / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        if checkpoint["fingerprints"] != fingerprints:
            raise ValueError(f"{name}: данные или токенизатор отличаются от обучающих")
        model = GPT(GPTConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model"])
        model = model.to(device).eval()
        model_info = {
            "model": name,
            "checkpoint": str(checkpoint_path),
            "step": checkpoint["step"],
            "training_tokens_seen": checkpoint["tokens_seen"],
            "parameters": model.num_parameters(),
            "model_config": checkpoint["model_config"],
            "train_config": checkpoint["train_config"],
        }
        del checkpoint

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for split in ("test_iid", "test_tail"):
                tokens = open_token_stream(DEFAULT_TOKENIZED_DIR, metadata, split, model.config.context_length)
                loss = evaluate(model, tokens, batch_size, device)
                perplexity = math.exp(loss)
                report["results"].append({
                    **model_info, "split": split, "scored_tokens": len(tokens) - 1,
                    "loss": loss, "perplexity": perplexity,
                })
                save_report()
                print(f"{name:<14} | {split:<9} | {loss:.4f} | {perplexity:.2f}", flush=True)
                del tokens

        del model
        gc.collect()
        torch.cuda.empty_cache()

    report["status"] = "complete"
    save_report()
    print(f"\nРезультаты: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(batch_size=args.batch_size)
