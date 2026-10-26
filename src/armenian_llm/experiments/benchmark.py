"""Сравнение размеров батча при одинаковом числе токенов на шаг AdamW.

Время включает чтение окон из memmap, перенос на GPU, forward, backward,
накопление градиентов, clipping и обновление весов. Разогрев исключён.
Память: https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.max_memory_allocated.html
"""

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import argparse
import gc
import json
import math
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from armenian_llm.modeling.gpt import GPT, GPTConfig
from armenian_llm.paths import ARTIFACTS_DIR, DEFAULT_TOKENIZED_DIR, DEFAULT_TOKENIZER_DIR
from armenian_llm.training.utils import file_sha256, load_metadata, make_batch, make_optimizer, open_token_stream


def optimizer_step(model, optimizer, tokens, rng, micro_batch_size, accumulation_steps, device):
    optimizer.zero_grad(set_to_none=True)
    context_length = model.config.context_length
    mean_loss = torch.zeros((), device=device)
    for _ in range(accumulation_steps):
        starts = rng.integers(0, len(tokens) - context_length, size=micro_batch_size)
        x, y = make_batch(tokens, starts, context_length, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits, loss = model(x, y)
        mean_loss.add_(loss.detach() / accumulation_steps)
        (loss / accumulation_steps).backward()
        del logits, loss, x, y
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()
    return mean_loss


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_case(config, tokens, micro_batch_size, effective_batch_size, warmup_steps, steps, device):
    accumulation_steps = effective_batch_size // micro_batch_size
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = GPT(config).to(device)
    model.train()
    optimizer = make_optimizer(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    attention_context = (
        sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device.type == "cuda" else nullcontext()
    )
    with attention_context:
        for _ in range(warmup_steps):
            optimizer_step(
                model, optimizer, tokens, rng, micro_batch_size, accumulation_steps, device
            )
        synchronize(device)
        started = time.perf_counter()
        for _ in range(steps):
            last_loss = optimizer_step(
                model, optimizer, tokens, rng, micro_batch_size, accumulation_steps, device
            )
        synchronize(device)
        seconds = time.perf_counter() - started

    last_loss = last_loss.item()
    if not math.isfinite(last_loss):
        raise RuntimeError("Loss не является конечным")
    timed_tokens = steps * effective_batch_size * config.context_length
    tokens_per_second = timed_tokens / seconds
    return {
        "status": "ok",
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "tokens_per_optimizer_step": effective_batch_size * config.context_length,
        "timed_optimizer_steps": steps,
        "timed_tokens": timed_tokens,
        "timed_seconds": seconds,
        "seconds_per_optimizer_step": seconds / steps,
        "tokens_per_second": tokens_per_second,
        "last_loss": last_loss,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "estimated_train_pass_hours": len(tokens) / tokens_per_second / 3600,
    }


def main(
    data_dir=DEFAULT_TOKENIZED_DIR,
    tokenizer_dir=DEFAULT_TOKENIZER_DIR,
    output_dir=ARTIFACTS_DIR / "training_benchmark",
    batch_sizes=(16, 32, 64, 128),
    effective_batch_size=128,
    warmup_steps=2,
    steps=5,
    device_name="cuda",
    model_config=None,
):
    if warmup_steps < 1 or steps < 1 or effective_batch_size < 1 or not batch_sizes:
        raise ValueError("Число шагов, эффективный батч и список батчей должны быть положительными")
    batch_sizes = sorted(set(batch_sizes))
    if any(size < 1 or effective_batch_size % size for size in batch_sizes):
        raise ValueError("Каждый micro batch должен быть положительным делителем effective batch")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Нужны CUDA и поддержка BF16")
    elif device.type != "cpu":
        raise ValueError("Поддерживаются cuda и cpu")

    metadata = load_metadata(data_dir, tokenizer_dir)
    config = model_config or GPTConfig(vocab_size=metadata["vocab_size"])
    if config.vocab_size != metadata["vocab_size"]:
        raise ValueError("Размер словаря модели не совпадает с токенизированными данными")
    tokens = open_token_stream(data_dir, metadata, "train", config.context_length)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "results.json"
    report_path.unlink(missing_ok=True)
    report = {
        "config": asdict(config),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "seed": 42,
        "precision": "bf16_autocast_fp32_parameters" if device.type == "cuda" else "fp32",
        "attention_backend": "pytorch_flash_attention" if device.type == "cuda" else "pytorch_cpu_sdpa",
        "torch_compile": False,
        "effective_batch_size": effective_batch_size,
        "tokens_per_optimizer_step": effective_batch_size * config.context_length,
        "warmup_optimizer_steps": warmup_steps,
        "timed_optimizer_steps": steps,
        "timing_includes": ["data_loading", "host_to_device", "forward", "backward", "gradient_clipping", "adamw"],
        "memory_includes_warmup": True,
        "train_tokens": len(tokens),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "data_metadata_sha256": file_sha256(data_dir / "metadata.json"),
        "checkpoint_saved": False,
        "results": [],
    }
    print("Устройство:", report["gpu_name"] or "CPU", flush=True)
    print("Конфигурация:", asdict(config), flush=True)
    print(f"Эффективный батч: {effective_batch_size} последовательностей", flush=True)
    print(f"Токенов на шаг AdamW: {report['tokens_per_optimizer_step']:,}", flush=True)
    print(f"Для каждого варианта: {warmup_steps} шага разогрева + {steps} шагов замера", flush=True)

    for size in batch_sizes:
        print(
            f"\nMicro batch {size}, накопление {effective_batch_size // size}...",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            result = benchmark_case(
                config, tokens, size, effective_batch_size, warmup_steps, steps, device
            )
        except torch.cuda.OutOfMemoryError:
            result = {"status": "oom", "micro_batch_size": size}
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["results"].append(result)
        temporary = report_path.with_name(report_path.name + ".part")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        if result["status"] == "oom":
            print("Не хватило GPU-памяти; большие батчи пропускаем.", flush=True)
            break
        print(f"Скорость: {result['tokens_per_second']:,.0f} токенов/с", flush=True)

    print("\nИтог замера обучения:")
    print("Micro | Накопление | Токенов/с | Секунд/шаг | Пик allocated, ГиБ | Пик reserved, ГиБ")
    for result in report["results"]:
        if result["status"] == "oom":
            print(f"{result['micro_batch_size']:>5} | OOM")
            continue
        allocated = result["peak_cuda_allocated_bytes"]
        reserved = result["peak_cuda_reserved_bytes"]
        allocated_text = f"{allocated / 2**30:.3f}" if allocated is not None else "—"
        reserved_text = f"{reserved / 2**30:.3f}" if reserved is not None else "—"
        print(
            f"{result['micro_batch_size']:>5} | {result['gradient_accumulation_steps']:>10} |"
            f" {result['tokens_per_second']:>9,.0f} | {result['seconds_per_optimizer_step']:>10.3f} |"
            f" {allocated_text:>18} | {reserved_text:>17}"
        )
        print(f"      Оценка прохода по train без валидации и сохранений: {result['estimated_train_pass_hours']:.2f} ч")
    print(f"\nОтчёт: {report_path.resolve()}")
    print("Веса после замеров не сохраняются.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "training_benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--effective-batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(
        args.data_dir, args.tokenizer_dir, args.output_dir, args.batch_sizes,
        args.effective_batch_size, args.warmup_steps, args.steps, args.device,
    )
