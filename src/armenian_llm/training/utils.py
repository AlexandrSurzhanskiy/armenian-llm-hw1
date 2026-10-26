"""Чтение токенизированного корпуса, создание батчей и настройка AdamW."""

import hashlib
import json

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(data_dir, tokenizer_dir):
    path = data_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata["token_dtype"] != "<u2":
        raise ValueError("Ожидались ID в формате little-endian uint16")
    if file_sha256(tokenizer_dir / "tokenizer.json") != metadata["tokenizer_sha256"]:
        raise ValueError("Токенизатор не совпал с использованным для кодирования корпуса")
    return metadata


def open_token_stream(data_dir, metadata, split, context_length):
    info = metadata["splits"][split]
    path = data_dir / info["tokens_file"]
    if path.stat().st_size != info["tokens"] * 2:
        raise ValueError(f"Размер {path} не совпал с metadata.json")
    if info["tokens"] <= context_length:
        raise ValueError(f"В {split} недостаточно токенов для заданного контекста")
    return np.memmap(path, dtype="<u2", mode="r")


def make_batch(tokens, starts, context_length, device):
    """Каждое окно длины T+1 даёт x длины T и сдвинутые на токен targets."""
    if context_length < 1:
        raise ValueError("context_length должен быть положительным")
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("Нужен непустой одномерный массив начал окон")
    if starts.min() < 0 or starts.max() + context_length >= len(tokens):
        raise ValueError("Окно выходит за границы потока токенов")
    windows = np.stack([
        tokens[int(start):int(start) + context_length + 1] for start in starts
    ]).astype(np.int64)
    windows = torch.from_numpy(windows)
    x, y = windows[:, :-1].contiguous(), windows[:, 1:].contiguous()
    if device.type == "cuda":
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def make_optimizer(model, learning_rate=3e-4, weight_decay=0.1):
    parameters = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=learning_rate, betas=(0.9, 0.95),
        fused=parameters[0].device.type == "cuda",
    )
