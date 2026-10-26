"""Один проверочный шаг на реальных токенах перед длительным обучением."""

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from armenian_llm.modeling.gpt import GPT, GPTConfig
from armenian_llm.paths import ARTIFACTS_DIR, DEFAULT_TOKENIZED_DIR, DEFAULT_TOKENIZER_DIR


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_batch(data_dir, metadata, batch_size, context_length, seed):
    info = metadata["splits"]["train"]
    if metadata["token_dtype"] != "<u2" or metadata["offset_dtype"] != "<u8":
        raise ValueError("Ожидались uint16 ID и uint64 границы документов")
    token_path = data_dir / info["tokens_file"]
    offset_path = data_dir / info["offsets_file"]
    if token_path.stat().st_size != info["tokens"] * 2:
        raise ValueError("Размер train.bin не совпал с metadata.json")
    if offset_path.stat().st_size != (info["documents"] + 1) * 8:
        raise ValueError("Размер файла границ не совпал с metadata.json")
    tokens = np.memmap(token_path, dtype="<u2", mode="r")
    offsets = np.memmap(offset_path, dtype="<u8", mode="r")
    if offsets[0] != 0 or offsets[-1] != len(tokens):
        raise ValueError("Некорректные крайние границы документов")
    first_end = int(offsets[1])
    special_ids = metadata["special_token_ids"]
    if not 2 <= first_end <= len(tokens):
        raise ValueError("Некорректная граница первого документа")
    if tokens[0] != special_ids["<bos>"] or tokens[first_end - 1] != special_ids["<eos>"]:
        raise ValueError("Первый документ не имеет ожидаемых BOS/EOS")
    if len(tokens) <= context_length:
        raise ValueError("В train недостаточно токенов для заданного контекста")

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - context_length, size=batch_size)
    windows = np.stack([
        tokens[int(start):int(start) + context_length + 1] for start in starts
    ]).astype(np.int64)
    if windows.min() < 0 or windows.max() >= metadata["vocab_size"]:
        raise ValueError("В батче встретился ID за пределами словаря")
    windows = torch.from_numpy(windows)
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def block_memory(block, optimizer):
    parameters = list(block.parameters())
    moments = sum(
        value.numel() * value.element_size()
        for parameter in parameters
        for key, value in optimizer.state[parameter].items()
        if key in ("exp_avg", "exp_avg_sq")
    )
    steps = sum(
        value.numel() * value.element_size()
        for parameter in parameters
        for key, value in optimizer.state[parameter].items()
        if key == "step" and isinstance(value, torch.Tensor)
    )
    return {
        "parameters": sum(p.numel() for p in parameters),
        "weights_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "gradients_bytes": sum(
            p.grad.numel() * p.grad.element_size()
            for p in parameters if p.grad is not None
        ),
        "adam_moments_bytes": moments,
        "adam_step_counters_bytes": steps,
        "weight_dtypes": sorted({str(p.dtype) for p in parameters}),
        "gradient_dtypes": sorted({str(p.grad.dtype) for p in parameters if p.grad is not None}),
        "moment_dtypes": sorted({
            str(value.dtype) for p in parameters
            for key, value in optimizer.state[p].items()
            if key in ("exp_avg", "exp_avg_sq")
        }),
    }


def main(
    data_dir=DEFAULT_TOKENIZED_DIR,
    tokenizer_dir=DEFAULT_TOKENIZER_DIR,
    output_dir=ARTIFACTS_DIR / "model_check",
    batch_size=2,
    device_name="cuda",
):
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Для этой проверки нужны CUDA и поддержка BF16")
    elif device.type != "cpu":
        raise ValueError("Поддерживаются cuda и cpu")

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if file_sha256(tokenizer_dir / "tokenizer.json") != metadata["tokenizer_sha256"]:
        raise ValueError("Токенизатор не совпал с использованным при кодировании корпуса")

    config = GPTConfig(vocab_size=metadata["vocab_size"])
    x, y = read_batch(data_dir, metadata, batch_size, config.context_length, seed=42)
    assert torch.equal(x[:, 1:], y[:, :-1])
    x, y = x.to(device), y.to(device)
    torch.manual_seed(42)
    model = GPT(config).to(device)
    model.train()
    parameter_groups = [
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.1},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=3e-4, betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("Устройство:", torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", flush=True)
    print("Конфигурация:", asdict(config), flush=True)
    print(f"Обучаемых параметров: {model.num_parameters():,}", flush=True)
    print("Форма x / y:", tuple(x.shape), "/", tuple(y.shape), flush=True)

    # На H100 явно проверяем Flash Attention из PyTorch, без стороннего flash-attn.
    attention_context = (
        sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device.type == "cuda" else nullcontext()
    )
    with attention_context:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits, loss = model(x, y)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError("Начальный loss не является конечным")
        loss.backward()

    if any(p.grad is None for p in model.parameters()):
        raise RuntimeError("Не все параметры получили градиенты")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0, error_if_nonfinite=True
    ).item()
    optimizer.step()
    third_block = block_memory(model.blocks[2], optimizer)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    report = {
        "config": asdict(config),
        "parameters": model.num_parameters(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "precision": "bf16_autocast_fp32_parameters" if device.type == "cuda" else "fp32",
        "attention_backend": "pytorch_flash_attention" if device.type == "cuda" else "pytorch_cpu_sdpa",
        "batch_shape": list(x.shape),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "initial_loss": loss_value,
        "uniform_loss_log_vocab": math.log(config.vocab_size),
        "gradient_norm_before_clipping": gradient_norm,
        "peak_cuda_allocated_bytes": peak_bytes,
        "third_block": third_block,
        "data_metadata_sha256": file_sha256(metadata_path),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "optimizer_steps": 1,
        "checkpoint_saved": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    temporary = report_path.with_name(report_path.name + ".part")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)

    print("\nИтог проверки модели:")
    print(f"Параметров: {model.num_parameters():,}")
    print("Форма logits:", tuple(logits.shape), "dtype:", logits.dtype)
    print(f"Начальный loss: {loss_value:.4f}; log(vocab_size): {math.log(config.vocab_size):.4f}")
    print(f"Норма градиентов до clipping: {gradient_norm:.4f}")
    print("Forward / backward / AdamW: OK")
    print("Attention:", report["attention_backend"])
    if peak_bytes is not None:
        print(f"Пик выделенной CUDA-памяти: {peak_bytes / 2**30:.3f} ГиБ")
    print("Третий блок Transformer:")
    print(f"  Параметров: {third_block['parameters']:,}")
    print(f"  Веса FP32: {third_block['weights_bytes']:,} байт")
    print(f"  Градиенты FP32: {third_block['gradients_bytes']:,} байт")
    print(f"  Два момента AdamW: {third_block['adam_moments_bytes']:,} байт")
    print(f"  Счётчики шагов AdamW: {third_block['adam_step_counters_bytes']:,} байт")
    print(f"Отчёт: {report_path.resolve()}")
    print("Проверочный шаг завершён; чекпоинт не сохранялся.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "model_check")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.data_dir, args.tokenizer_dir, args.output_dir, args.batch_size, args.device)
