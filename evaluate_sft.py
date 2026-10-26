"""Сравнение претрейна 97M и лучшего SFT-чекпоинта на SFT test.

Запуск из ноутбука: %run /workdir/evaluate_sft.py
Считаем loss только на эталонных ответах и EOS, с teacher forcing.
Это оценка вероятности текстов ответов; она не равна точности свободного QA.
"""

import argparse
import gc
import json
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from paths import ARTIFACTS_DIR, CHECKPOINTS_DIR, DATA_DIR, DEFAULT_TOKENIZER_DIR
from train_sft import evaluate_answers, load_records
from training_utils import file_sha256


def main(batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Нужны CUDA и поддержка BF16")
    device = torch.device("cuda")
    data_dir = DATA_DIR / "sft" / "aya_dolly_qa"
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_hash = file_sha256(metadata_path)
    tokenizer_path = DEFAULT_TOKENIZER_DIR / "tokenizer.json"
    tokenizer_hash = file_sha256(tokenizer_path)
    if tokenizer_hash != metadata["tokenizer_sha256"] or metadata["ignore_index"] != -100:
        raise ValueError("Токенизатор или ignore_index отличаются от подготовки SFT")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id, bos_id, eos_id = [tokenizer.token_to_id(name) for name in ("<pad>", "<bos>", "<eos>")]
    if None in (pad_id, bos_id, eos_id):
        raise ValueError("В токенизаторе отсутствуют специальные токены")

    test_info = metadata["splits"]["test"]
    output_dir = ARTIFACTS_DIR / "evaluation" / "sft_heldout_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    report = {
        "status": "running", "split": "test", "data_dir": str(data_dir.resolve()),
        "sft_metadata_sha256": metadata_hash, "test_sha256": test_info["sha256"],
        "tokenizer_sha256": tokenizer_hash, "batch_size": batch_size,
        "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name(device),
        "precision": "bf16_autocast_fp32_parameters",
        "method": "token-weighted conditional cross-entropy on all test answer and EOS tokens; teacher forcing",
        "checkpoint_selection": "pretrain best by ArmWeb validation; SFT best by SFT validation answer loss",
        "limitations_ru": metadata["limitations_ru"] + [
            "Loss измеряет вероятность эталонных текстов при подаче предыдущих эталонных токенов, а не фактическую точность свободной генерации.",
            "Этот условный perplexity не сопоставим напрямую с perplexity на новостном корпусе ArmWeb.",
        ],
        "results": [],
    }

    def save_report():
        temporary = output_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output_path)

    save_report()
    print(f"Устройство: {report['gpu']}", flush=True)
    print(f"SFT test: {test_info['examples']:,} пар; loss только по ответу и EOS", flush=True)
    print("Модель          | Answer loss | Answer perplexity", flush=True)
    for name in ("arm_gpt_97m", "arm_gpt_97m_sft"):
        checkpoint_path = CHECKPOINTS_DIR / name / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        if checkpoint["fingerprints"]["tokenizer_sha256"] != tokenizer_hash:
            raise ValueError(f"{name}: токенизатор отличается от тестовой выборки")
        stage = checkpoint.get("stage", "pretrain")
        if stage == "sft" and checkpoint["fingerprints"]["data_metadata_sha256"] != metadata_hash:
            raise ValueError("Тестовые данные отличаются от подготовленных перед SFT")
        config = GPTConfig(**checkpoint["model_config"])
        if config.vocab_size != tokenizer.get_vocab_size():
            raise ValueError(f"{name}: размер словаря не совпал")
        records = load_records(data_dir / test_info["file"], test_info, config.vocab_size,
                               config.context_length, bos_id, eos_id)
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device).eval()
        result = {
            "model": name, "checkpoint": str(checkpoint_path.resolve()), "stage": stage,
            "step": checkpoint["step"], "parameters": model.num_parameters(),
            "model_config": checkpoint["model_config"], "train_config": checkpoint["train_config"],
            "fingerprints": checkpoint["fingerprints"], "provenance": checkpoint.get("provenance"),
            "examples": len(records),
            "scored_answer_tokens_including_eos": sum(row["answer_tokens_including_eos"] for row in records),
        }
        del checkpoint
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            loss = evaluate_answers(model, records, batch_size, pad_id, device)
        result.update(answer_loss=loss, answer_perplexity=math.exp(loss))
        report["results"].append(result)
        save_report()
        print(f"{name:<15} | {loss:>11.4f} | {math.exp(loss):>17.2f}", flush=True)
        del model, records
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
