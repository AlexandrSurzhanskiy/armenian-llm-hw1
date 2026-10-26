"""Короткий SFT нашей 97M GPT на армянских QA-парах.

Запуск из ноутбука: %run -m armenian_llm.training.sft
Продолжение:        %run -m armenian_llm.training.sft --resume
Loss считаем только на ответе и EOS. Test при обучении не читается.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tokenizers import Tokenizer

from armenian_llm.inference.generate import generate_text
from armenian_llm.modeling.gpt import GPT, GPTConfig
from armenian_llm.paths import CHECKPOINTS_DIR, DATA_DIR, DEFAULT_TOKENIZER_DIR
from armenian_llm.provenance import snapshot_code
from armenian_llm.training.pretrain import LOG_FIELDS, atomic_save, learning_rate, trim_logs
from armenian_llm.training.utils import file_sha256, make_optimizer


@dataclass(frozen=True)
class SFTConfig:
    epochs: int = 3
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    max_learning_rate: float = 5e-5
    min_learning_rate: float = 5e-6
    warmup_steps: int = 10
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    log_interval: int = 10
    sample_tokens: int = 128
    seed: int = 42

    def __post_init__(self):
        for name in ("epochs", "micro_batch_size", "gradient_accumulation_steps", "log_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} должен быть положительным")
        if not 0 <= self.min_learning_rate <= self.max_learning_rate or self.max_learning_rate <= 0:
            raise ValueError("Нужны 0 <= min_learning_rate <= max_learning_rate и max_learning_rate > 0")
        if self.warmup_steps < 0 or self.sample_tokens < 0 or self.gradient_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Некорректные настройки SFT")


def load_records(path, info, vocab_size, context_length, bos_id, eos_id):
    if file_sha256(path) != info["sha256"]:
        raise ValueError(f"{path}: содержимое отличается от metadata.json")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len(records) != info["examples"]:
        raise ValueError(f"{path}: неверное число примеров")
    for row in records:
        ids, labels = row["input_ids"], row["labels"]
        prefix = row["prompt_tokens_including_bos"]
        if not (2 <= len(ids) <= context_length + 1 and len(ids) == len(labels)
                and 1 <= prefix < len(ids) and ids[0] == bos_id and ids[-1] == eos_id
                and min(ids) >= 0 and max(ids) < vocab_size
                and labels[:prefix] == [-100] * prefix and labels[prefix:] == ids[prefix:]
                and len(ids) - prefix == row["answer_tokens_including_eos"]):
            raise ValueError(f"{path}: неверная маска или длина в примере {row['id']}")
    return records


def make_sft_batch(records, pad_id, device):
    """labels в файле не сдвинуты. Сдвигаем ровно один раз, затем дополняем справа."""
    length = max(len(row["input_ids"]) - 1 for row in records)
    x = torch.full((len(records), length), pad_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    targets = 0
    for index, row in enumerate(records):
        size = len(row["input_ids"]) - 1
        x[index, :size] = torch.tensor(row["input_ids"][:-1], dtype=torch.long)
        y[index, :size] = torch.tensor(row["labels"][1:], dtype=torch.long)
        targets += row["answer_tokens_including_eos"]
    # При causal attention реальные токены не видят правый padding.
    # Его собственные предсказания исключены из loss через -100.
    if device.type == "cuda":
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), targets


@torch.inference_mode()
def evaluate_answers(model, records, batch_size, pad_id, device):
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    scored = 0
    try:
        for first in range(0, len(records), batch_size):
            x, y, count = make_sft_batch(records[first:first + batch_size], pad_id, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, loss = model(x, y)
            loss_sum.add_(loss.double() * count)
            scored += count
            del logits, loss, x, y
        value = (loss_sum / scored).item()
        if not math.isfinite(value):
            raise RuntimeError("Validation loss не является конечным")
        return value
    finally:
        model.train(was_training)


def main(
    data_dir=DATA_DIR / "sft" / "aya_dolly_qa",
    tokenizer_dir=DEFAULT_TOKENIZER_DIR,
    source_checkpoint=CHECKPOINTS_DIR / "arm_gpt_97m" / "best.pt",
    run_dir=CHECKPOINTS_DIR / "arm_gpt_97m_sft",
    config=None,
    resume=False,
    tensorboard=True,
    device_name="cuda",
):
    config = config or SFTConfig()
    data_dir, tokenizer_dir, source_checkpoint, run_dir = map(
        Path, (data_dir, tokenizer_dir, source_checkpoint, run_dir)
    )
    if source_checkpoint.parent.resolve() == run_dir.resolve():
        raise ValueError("Для SFT нужен отдельный каталог от исходного претрейна")
    if not resume and any((run_dir / name).exists() for name in ("config.json", "best.pt", "last.pt")):
        raise FileExistsError(f"В {run_dir} уже есть запуск. Продолжение: --resume")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Нужны CUDA и поддержка BF16")
    elif device.type != "cpu":
        raise ValueError("Поддерживаются cuda и cpu")

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    tokenizer_hash = file_sha256(tokenizer_dir / "tokenizer.json")
    if tokenizer_hash != metadata["tokenizer_sha256"] or metadata["ignore_index"] != -100:
        raise ValueError("Токенизатор или ignore_index отличаются от подготовки SFT")
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    pad_id, bos_id, eos_id = [tokenizer.token_to_id(name) for name in ("<pad>", "<bos>", "<eos>")]
    if None in (pad_id, bos_id, eos_id):
        raise ValueError("В токенизаторе отсутствуют специальные токены")

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model_config = GPTConfig(**source["model_config"])
    if source["fingerprints"]["tokenizer_sha256"] != tokenizer_hash or model_config.vocab_size != tokenizer.get_vocab_size():
        raise ValueError("Исходный чекпоинт обучался с другим токенизатором")
    provenance = {
        "source_checkpoint": str(source_checkpoint.resolve()), "source_step": source["step"],
        "source_tokens_seen": source["tokens_seen"], "source_fingerprints": source["fingerprints"],
    }
    fingerprints = {
        "tokenizer_sha256": tokenizer_hash,
        "data_metadata_sha256": file_sha256(data_dir / "metadata.json"),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
    }
    splits = {}
    for split in ("train", "validation"):
        info = metadata["splits"][split]
        splits[split] = load_records(
            data_dir / info["file"], info, model_config.vocab_size, model_config.context_length, bos_id, eos_id
        )
    train_rows, validation_rows = splits["train"], splits["validation"]
    effective_batch = config.micro_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_rows) / effective_batch)
    total_steps = steps_per_epoch * config.epochs
    torch.manual_seed(config.seed)
    model = GPT(model_config).to(device)
    model.load_state_dict(source["model"])
    del source
    # Новая фаза обучения: новый оптимизатор, без моментов AdamW из претрейна.
    optimizer = make_optimizer(model, config.max_learning_rate, config.weight_decay)
    step = tokens_seen = examples_seen = best_step = 0
    best_val = math.inf
    previous_elapsed = 0.0
    if resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if (checkpoint.get("stage") != "sft" or checkpoint["model_config"] != asdict(model_config)
                or checkpoint["train_config"] != asdict(config) or checkpoint["fingerprints"] != fingerprints):
            raise ValueError("Для resume нужны те же исходные веса, данные и настройки SFT")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, tokens_seen, examples_seen = checkpoint["step"], checkpoint["tokens_seen"], checkpoint["examples_seen"]
        best_val, best_step = checkpoint["best_val_loss"], checkpoint["best_step"]
        previous_elapsed = checkpoint["elapsed_seconds"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        del checkpoint
        trim_logs(run_dir, step)

    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        info = {
            "stage": "sft", "model": asdict(model_config), "training": asdict(config),
            "parameters": model.num_parameters(), "fingerprints": fingerprints, "provenance": provenance,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "train_examples": len(train_rows), "validation_examples": len(validation_rows),
            "prompt_template": metadata["prompt_template"],
            "objective": "mean next-token cross-entropy over answer and EOS tokens only",
            "tokens_seen_definition": "supervised answer and EOS tokens in SFT only",
            "batching": "shuffle without replacement each epoch; dynamic right padding; token-weighted accumulation",
            "checkpoint_selection": "minimum full SFT validation answer loss, evaluated before training and after each epoch",
            "precision": "bf16_autocast_fp32_parameters" if device.type == "cuda" else "fp32",
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "data_dir": str(data_dir.resolve()),
        }
        (run_dir / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copytree(tokenizer_dir, run_dir / "tokenizer", dirs_exist_ok=True)
        shutil.copy2(data_dir / "metadata.json", run_dir / "sft_data_metadata.json")
        snapshot_code(run_dir / "code")

    tb_writer = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=step + 1 if resume else None)
        except ImportError:
            print("TensorBoard не установлен; метрики сохраняются в CSV.", flush=True)
    csv_path = run_dir / "metrics.csv"
    write_header = not csv_path.exists()
    csv_file = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
    if write_header:
        writer.writeheader()
    started = time.perf_counter()

    def elapsed():
        return previous_elapsed + time.perf_counter() - started

    def log(split, loss, lr=0.0, norm=None, speed=None):
        writer.writerow(dict(zip(LOG_FIELDS, (
            step, step / steps_per_epoch, tokens_seen, split, loss, math.exp(loss), lr, norm, speed, elapsed()
        ))))
        csv_file.flush()
        if tb_writer:
            tb_writer.add_scalar(f"answer_loss/{split}", loss, step)
            if split == "train":
                tb_writer.add_scalar("train/learning_rate", lr, step)
                tb_writer.add_scalar("train/gradient_norm", norm, step)

    def payload(include_optimizer):
        state = {
            "stage": "sft", "model": model.state_dict(), "model_config": asdict(model_config),
            "train_config": asdict(config), "step": step, "tokens_seen": tokens_seen,
            "examples_seen": examples_seen, "best_val_loss": best_val, "best_step": best_step,
            "fingerprints": fingerprints, "provenance": provenance, "elapsed_seconds": elapsed(),
            "validation_objective": "answer_and_eos_only",
        }
        if include_optimizer:
            state.update(optimizer=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(),
                         cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        return state

    def validate_and_save():
        nonlocal best_val, best_step
        loss = evaluate_answers(model, validation_rows, config.micro_batch_size, pad_id, device)
        log("validation", loss)
        print(f"VALIDATION step {step:,}: answer loss {loss:.4f}; answer perplexity {math.exp(loss):.2f}", flush=True)
        if loss < best_val:
            best_val, best_step = loss, step
            atomic_save(payload(False), run_dir / "best.pt")
        atomic_save(payload(True), run_dir / "last.pt")
        if tb_writer:
            tb_writer.flush()
        if config.sample_tokens:
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as output:
                for row in validation_rows[:2]:
                    completion = generate_text(model, tokenizer, row["prompt"], device,
                                               max_new_tokens=config.sample_tokens, temperature=1.0, top_k=1, seed=42)
                    output.write(json.dumps({"step": step, "id": row["id"], "prompt": row["prompt"],
                                             "reference": row["answer"], "completion": completion}, ensure_ascii=False) + "\n")
                    print(f"  Вопрос: {row['question']}\n  Эталон: {row['answer']}\n  Модель: {completion}", flush=True)

    print(f"Устройство: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}", flush=True)
    print(f"SFT из: {source_checkpoint}; параметров: {model.num_parameters():,}", flush=True)
    print(f"Train: {len(train_rows):,}; validation: {len(validation_rows):,}; эпох: {config.epochs}", flush=True)
    print(f"Micro batch: {config.micro_batch_size}; накопление: {config.gradient_accumulation_steps}; effective batch: {effective_batch}", flush=True)
    print(f"Шагов/эпоху: {steps_per_epoch}; всего шагов: {total_steps}; LR: {config.max_learning_rate:g} → {config.min_learning_rate:g}", flush=True)
    print(f"Каталог: {run_dir.resolve()}", flush=True)
    attention = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device.type == "cuda" else nullcontext()
    try:
        with attention:
            if not resume:
                validate_and_save()
            model.train()
            current_epoch, order = None, None
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            interval_tokens = 0
            interval_started = time.perf_counter()
            while step < total_steps:
                epoch = step // steps_per_epoch
                if epoch != current_epoch:
                    order = np.random.default_rng(config.seed + epoch).permutation(len(train_rows))
                    current_epoch = epoch
                first = (step % steps_per_epoch) * effective_batch
                rows = [train_rows[int(index)] for index in order[first:first + effective_batch]]
                batch_targets = sum(row["answer_tokens_including_eos"] for row in rows)
                lr = learning_rate(step, total_steps, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                for first_micro in range(0, len(rows), config.micro_batch_size):
                    x, y, count = make_sft_batch(rows[first_micro:first_micro + config.micro_batch_size], pad_id, device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits, loss = model(x, y)
                    loss_sum.add_(loss.detach().double() * count)
                    interval_tokens += count
                    # Средний loss микробатча взвешиваем по числу целевых токенов.
                    (loss * (count / batch_targets)).backward()
                    del logits, loss, x, y
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                tokens_seen += batch_targets
                examples_seen += len(rows)
                if step % config.log_interval == 0 or step % steps_per_epoch == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    value = (loss_sum / interval_tokens).item()
                    if not math.isfinite(value):
                        raise RuntimeError("Train loss не является конечным")
                    speed = interval_tokens / (time.perf_counter() - interval_started)
                    log("train", value, lr, norm.item(), speed)
                    print(f"step {step:,}/{total_steps:,} | train answer loss {value:.4f} | lr {lr:.2e}", flush=True)
                    loss_sum.zero_()
                    interval_tokens = 0
                    interval_started = time.perf_counter()
                if step % steps_per_epoch == 0:
                    validate_and_save()
                    interval_started = time.perf_counter()
    except KeyboardInterrupt:
        print("\nПрервано. Продолжение с последней сохранённой эпохи: %run -m armenian_llm.training.sft --resume", flush=True)
        return
    finally:
        csv_file.close()
        if tb_writer:
            tb_writer.close()

    summary = {
        "stage": "sft", "steps": step, "epochs": config.epochs, "examples_seen": examples_seen,
        "supervised_tokens_seen": tokens_seen, "best_step": best_step,
        "best_validation_answer_loss": best_val, "best_validation_answer_perplexity": math.exp(best_val),
        "elapsed_seconds": elapsed(), "provenance": provenance,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSFT завершён. Лучший шаг: {best_step}/{total_steps}; validation answer loss: {best_val:.4f}", flush=True)
    print(f"Чекпоинт для инференса: {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    main(resume=args.resume, tensorboard=not args.no_tensorboard)
