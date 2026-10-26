"""Обучение GPT с нуля: полный проход по окнам train, validation и чекпоинты.

Все большие файлы сохраняются по путям из paths.py.
Продолжение после остановки: %run train.py --resume
Пилот: --stop-at-step N останавливает запуск, сохраняя полный график LR.
Формат чекпоинта: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
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

from generate import generate_text
from model import GPT, GPTConfig
from paths import CHECKPOINTS_DIR, DEFAULT_TOKENIZED_DIR, DEFAULT_TOKENIZER_DIR
from training_utils import file_sha256, load_metadata, make_batch, make_optimizer, open_token_stream


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    eval_interval: int = 500
    log_interval: int = 50
    sample_tokens: int = 96
    seed: int = 42
    max_steps: int | None = None

    def __post_init__(self):
        for name in ("epochs", "micro_batch_size", "gradient_accumulation_steps", "eval_interval", "log_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} должен быть положительным")
        if self.warmup_steps < 0 or self.sample_tokens < 0:
            raise ValueError("warmup_steps и sample_tokens должны быть неотрицательными")
        if not 0 <= self.min_learning_rate <= self.max_learning_rate or self.max_learning_rate <= 0:
            raise ValueError("Нужны 0 <= min_learning_rate <= max_learning_rate и max_learning_rate > 0")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps должен быть положительным")


PROMPTS = ["Հայաստանի մայրաքաղաքը", "Այսօր Երևանում", "Հայ գրականության պատմության մեջ"]
LOG_FIELDS = ["step", "epoch", "tokens_seen", "split", "loss", "perplexity", "learning_rate", "gradient_norm", "tokens_per_second", "elapsed_seconds"]


def learning_rate(step, total_steps, config):
    warmup = min(config.warmup_steps, total_steps)
    if step < warmup:
        return config.max_learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    return config.min_learning_rate + 0.5 * (config.max_learning_rate - config.min_learning_rate) * (1 + math.cos(math.pi * progress))


def atomic_save(payload, path):
    temporary = path.with_name(path.name + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, tokens, batch_size, device):
    was_training = model.training
    model.eval()
    length = model.config.context_length
    full_windows = (len(tokens) - 1) // length
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    scored = 0
    try:
        for first in range(0, full_windows, batch_size):
            starts = np.arange(first, min(first + batch_size, full_windows)) * length
            x, y = make_batch(tokens, starts, length, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, loss = model(x, y)
            total_loss.add_(loss.double() * y.numel())
            scored += y.numel()
            del logits, loss, x, y
        remainder = len(tokens) - 1 - full_windows * length
        if remainder:
            x, y = make_batch(tokens, [full_windows * length], remainder, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, loss = model(x, y)
            total_loss.add_(loss.double() * y.numel())
            scored += y.numel()
        value = (total_loss / scored).item()
        if not math.isfinite(value):
            raise RuntimeError("Validation loss не является конечным")
        return value
    finally:
        model.train(was_training)


def trim_logs(run_dir, step):
    csv_path = run_dir / "metrics.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = [row for row in csv.DictReader(f) if int(row["step"]) <= step]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    sample_path = run_dir / "samples.jsonl"
    if sample_path.exists():
        rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
        with sample_path.open("w", encoding="utf-8") as f:
            for row in rows:
                if row["step"] <= step:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(
    data_dir=DEFAULT_TOKENIZED_DIR,
    tokenizer_dir=DEFAULT_TOKENIZER_DIR,
    run_dir=CHECKPOINTS_DIR / "arm_gpt_35m",
    train_config=None,
    model_config=None,
    resume=False,
    tensorboard=True,
    device_name="cuda",
    stop_at_step=None,
):
    config = train_config or TrainConfig()
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Нужны CUDA и поддержка BF16")
    elif device.type != "cpu":
        raise ValueError("Поддерживаются cuda и cpu")

    metadata = load_metadata(data_dir, tokenizer_dir)
    model_config = model_config or GPTConfig(vocab_size=metadata["vocab_size"])
    if model_config.vocab_size != metadata["vocab_size"]:
        raise ValueError("Размер словаря модели не совпал с корпусом")
    train_tokens = open_token_stream(data_dir, metadata, "train", model_config.context_length)
    val_tokens = open_token_stream(data_dir, metadata, "validation", model_config.context_length)
    windows = (len(train_tokens) - 1) // model_config.context_length
    effective_batch = config.micro_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(windows / effective_batch)
    total_steps = steps_per_epoch * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    if stop_at_step is not None and stop_at_step < 1:
        raise ValueError("stop_at_step должен быть положительным")
    # Горизонт LR остаётся total_steps. Пилот ограничивает только текущий вызов.
    stopping_step = min(total_steps, stop_at_step) if stop_at_step is not None else total_steps
    fingerprints = {
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "data_metadata_sha256": file_sha256(data_dir / "metadata.json"),
    }
    if not resume and any((run_dir / name).exists() for name in ("config.json", "last.pt", "best.pt", "initial.pt")):
        raise FileExistsError(f"В {run_dir} уже есть запуск. Используй --resume или другой --run-dir")

    checkpoint = None
    if resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != asdict(model_config) or checkpoint["train_config"] != asdict(config):
            raise ValueError("Для resume нужны те же настройки модели и обучения")
        if checkpoint["fingerprints"] != fingerprints:
            raise ValueError("Данные или токенизатор отличаются от сохранённого запуска")

    torch.manual_seed(config.seed)
    model = GPT(model_config).to(device)
    optimizer = make_optimizer(model, config.max_learning_rate, config.weight_decay)
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    step = tokens_seen = 0
    best_val = math.inf
    previous_elapsed = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, tokens_seen, best_val = checkpoint["step"], checkpoint["tokens_seen"], checkpoint["best_val_loss"]
        previous_elapsed = checkpoint["elapsed_seconds"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        del checkpoint
        trim_logs(run_dir, step)
    if step > stopping_step:
        raise ValueError(f"В чекпоинте уже шаг {step}; stop_at_step должен быть не меньше")

    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        info = {
            "model": asdict(model_config), "training": asdict(config), "fingerprints": fingerprints,
            "parameters": model.num_parameters(), "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "train_windows": windows, "discarded_train_tail_targets": (len(train_tokens) - 1) % model_config.context_length,
            "sampling": "shuffled_nonoverlapping_windows_without_replacement_each_epoch",
            "validation": "all_next_token_targets_in_separate_nonoverlapping_context_windows",
            "precision": "bf16_autocast_fp32_parameters" if device.type == "cuda" else "fp32",
            "torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "data_dir": str(data_dir.resolve()), "tokenizer_dir": str(tokenizer_dir.resolve()),
        }
        (run_dir / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copytree(tokenizer_dir, run_dir / "tokenizer", dirs_exist_ok=True)
        source_dir = Path(__file__).resolve().parent
        (run_dir / "code").mkdir(exist_ok=True)
        for name in ("model.py", "generate.py", "train.py", "training_utils.py", "paths.py"):
            shutil.copy2(source_dir / name, run_dir / "code" / name)

    tb_writer = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=step + 1 if resume else None)
        except ImportError:
            print("TensorBoard не установлен; метрики сохраняются в CSV. Установка: %pip install tensorboard", flush=True)

    csv_path = run_dir / "metrics.csv"
    write_header = not csv_path.exists()
    csv_file = csv_path.open("a", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
    if write_header:
        csv_writer.writeheader()
    started = time.perf_counter()

    def elapsed():
        return previous_elapsed + time.perf_counter() - started

    def log(split, loss, lr=0.0, gradient_norm=None, speed=None):
        row = {
            "step": step, "epoch": step / steps_per_epoch, "tokens_seen": tokens_seen,
            "split": split, "loss": loss, "perplexity": math.exp(loss), "learning_rate": lr,
            "gradient_norm": gradient_norm, "tokens_per_second": speed, "elapsed_seconds": elapsed(),
        }
        csv_writer.writerow(row)
        csv_file.flush()
        if tb_writer:
            tb_writer.add_scalar(f"loss/{split}", loss, step)
            tb_writer.add_scalar(f"perplexity/{split}", math.exp(loss), step)
            if split == "train":
                tb_writer.add_scalar("train/learning_rate", lr, step)
                tb_writer.add_scalar("train/gradient_norm", gradient_norm, step)
                tb_writer.add_scalar("train/tokens_per_second", speed, step)

    def payload(include_optimizer):
        state = {
            "model": model.state_dict(), "model_config": asdict(model_config), "train_config": asdict(config),
            "step": step, "tokens_seen": tokens_seen, "best_val_loss": best_val,
            "fingerprints": fingerprints, "elapsed_seconds": elapsed(),
        }
        if include_optimizer:
            state.update(
                optimizer=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            )
        return state

    def validation_and_save():
        nonlocal best_val
        loss = evaluate(model, val_tokens, config.micro_batch_size, device)
        log("validation", loss)
        print(f"VALIDATION step {step:,}: loss {loss:.4f}, perplexity {math.exp(loss):.2f}", flush=True)
        if loss < best_val:
            best_val = loss
            atomic_save(payload(False), run_dir / "best.pt")
        if config.sample_tokens:
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as f:
                for index, prompt in enumerate(PROMPTS):
                    completion = generate_text(
                        model, tokenizer, prompt, device, max_new_tokens=config.sample_tokens, seed=2026 + index
                    )
                    f.write(json.dumps({"step": step, "prompt": prompt, "completion": completion, "seed": 2026 + index}, ensure_ascii=False) + "\n")
                    if index == 0:
                        print(f"Пример: {prompt}{completion[:180]}", flush=True)
        atomic_save(payload(True), run_dir / "last.pt")
        if tb_writer:
            tb_writer.flush()

    print(f"Параметров: {model.num_parameters():,}; контекст: {model_config.context_length}", flush=True)
    print(f"Micro batch: {config.micro_batch_size}; накопление: {config.gradient_accumulation_steps}; effective batch: {effective_batch}", flush=True)
    print(f"Окон train: {windows:,}; шагов/эпоху: {steps_per_epoch:,}; всего шагов: {total_steps:,}", flush=True)
    print(f"LR: warmup {min(config.warmup_steps, total_steps)} шагов до {config.max_learning_rate:g}, cosine до {config.min_learning_rate:g}", flush=True)
    print(f"Каталог запуска: {run_dir.resolve()}", flush=True)
    if stopping_step < total_steps:
        print(f"Пилот до шага {stopping_step:,}; расписание LR рассчитано на все {total_steps:,} шагов", flush=True)
    attention_context = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device.type == "cuda" else nullcontext()
    try:
        with attention_context:
            if not resume:
                atomic_save(payload(False), run_dir / "initial.pt")
                validation_and_save()
            model.train()
            current_epoch = None
            order = None
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            interval_tokens = 0
            last_log_time = time.perf_counter()
            while step < stopping_step:
                epoch = step // steps_per_epoch
                if epoch != current_epoch:
                    order = np.random.default_rng(config.seed + epoch).permutation(windows)
                    current_epoch = epoch
                first = (step % steps_per_epoch) * effective_batch
                indices = order[first:first + effective_batch]
                lr = learning_rate(step, total_steps, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                for first_micro in range(0, len(indices), config.micro_batch_size):
                    micro = indices[first_micro:first_micro + config.micro_batch_size]
                    x, y = make_batch(train_tokens, micro * model_config.context_length, model_config.context_length, device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits, loss = model(x, y)
                    loss_sum.add_(loss.detach().double() * y.numel())
                    interval_tokens += y.numel()
                    (loss * (len(micro) / len(indices))).backward()
                    del logits, loss, x, y
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                tokens_seen += len(indices) * model_config.context_length
                step += 1
                if step % config.log_interval == 0 or step == stopping_step:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    seconds = time.perf_counter() - last_log_time
                    mean_loss = (loss_sum / interval_tokens).item()
                    if not math.isfinite(mean_loss):
                        raise RuntimeError("Train loss не является конечным")
                    speed = interval_tokens / seconds
                    log("train", mean_loss, lr, norm.item(), speed)
                    print(f"step {step:,}/{total_steps:,} | train loss {mean_loss:.4f} | lr {lr:.2e} | {speed:,.0f} токенов/с", flush=True)
                    loss_sum.zero_()
                    interval_tokens = 0
                    last_log_time = time.perf_counter()
                if step % config.eval_interval == 0 or step == stopping_step:
                    eval_started = time.perf_counter()
                    validation_and_save()
                    last_log_time += time.perf_counter() - eval_started
    except KeyboardInterrupt:
        print("\nОбучение прервано.", flush=True)
        if (run_dir / "last.pt").exists():
            print(f"Последний сохранённый шаг находится в {run_dir / 'last.pt'}", flush=True)
            print("Для продолжения повтори прежний вызов с resume=True; в CLI сохрани прежние аргументы и добавь --resume.", flush=True)
        else:
            print("Первый чекпоинт ещё не сохранён. Для нового запуска укажи другой --run-dir.", flush=True)
        return
    finally:
        csv_file.close()
        if tb_writer:
            tb_writer.close()

    summary = {
        "status": "complete" if step == total_steps else "pilot_complete",
        "steps": step, "total_steps": total_steps, "tokens_seen": tokens_seen,
        "best_validation_loss": best_val, "best_validation_perplexity": math.exp(best_val),
        "elapsed_seconds": elapsed(),
    }
    (run_dir / "progress.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if step < total_steps:
        print(f"\nПилот завершён: {step:,}/{total_steps:,} шагов; лучший validation loss {best_val:.4f}", flush=True)
        print(f"Состояние модели и оптимизатора: {run_dir / 'last.pt'}", flush=True)
        print("Для продолжения повтори вызов main с теми же настройками, resume=True и без stop_at_step.", flush=True)
        return
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nОбучение завершено: {step:,} шагов, {tokens_seen:,} токенов", flush=True)
    print(f"Лучший validation loss: {best_val:.4f}; perplexity: {math.exp(best_val):.2f}")
    print(f"Чекпоинт для инференса: {run_dir / 'best.pt'}")
    print(f"Чекпоинт для продолжения: {run_dir / 'last.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TOKENIZED_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--run-dir", type=Path, default=CHECKPOINTS_DIR / "arm_gpt_35m")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-learning-rate", type=float, default=6e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-at-step", type=int, help="Остановить текущий запуск на шаге N, сохранив полный горизонт LR; затем --resume без этого флага")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(
        args.data_dir, args.tokenizer_dir, args.run_dir,
        TrainConfig(epochs=args.epochs, micro_batch_size=args.micro_batch_size, gradient_accumulation_steps=args.grad_accum,
                    max_learning_rate=args.learning_rate, min_learning_rate=args.min_learning_rate, warmup_steps=args.warmup_steps,
                    eval_interval=args.eval_interval, log_interval=args.log_interval, max_steps=args.max_steps),
        resume=args.resume, tensorboard=not args.no_tensorboard, device_name=args.device, stop_at_step=args.stop_at_step,
    )
