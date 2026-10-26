"""Контрольный эксперимент: продолжение нашего претрейна на новостях и смеси.

%run /workdir/continue_pretrain.py --variant news
%run /workdir/continue_pretrain.py --variant mix

По умолчанию планируем 5000 шагов и останавливаем пилот после 1000.
Продолжение: те же аргументы, --resume --stop-at-step 5000.
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
import mixed_data
from mixed_data import DEFAULT_PLAN, SOURCE_NAMES, MixedWindowSampler, load_corpora, make_mixed_batch
from model import GPT, GPTConfig
from paths import CHECKPOINTS_DIR, DEFAULT_TOKENIZER_DIR
from train import atomic_save, evaluate, learning_rate
from training_utils import file_sha256, make_optimizer


@dataclass(frozen=True)
class ContinuationConfig:
    total_steps: int = 5000
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    eval_interval: int = 500
    log_interval: int = 50
    sample_tokens: int = 64
    seed: int = 42

    def __post_init__(self):
        for name in ("total_steps", "micro_batch_size", "gradient_accumulation_steps", "eval_interval", "log_interval"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} должен быть положительным")
        if not 0 <= self.min_learning_rate <= self.max_learning_rate or self.max_learning_rate <= 0:
            raise ValueError("Некорректные границы LR")
        if self.warmup_steps < 0 or self.sample_tokens < 0 or self.seed < 0:
            raise ValueError("warmup_steps, sample_tokens и seed должны быть неотрицательными")
        if self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Некорректные weight_decay или gradient_clip")


LOG_FIELDS = ["step", "tokens_seen", "armweb_tokens_seen", "wikipedia_tokens_seen", "split",
              "loss", "perplexity", "learning_rate", "gradient_norm", "tokens_per_second", "elapsed_seconds"]
PROMPTS = ["Հայաստանի մայրաքաղաքը", "Հայաստանը գտնվում է", "Այսօր Երևանում"]


def trim_logs(run_dir, step):
    path = run_dir / "metrics.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as source:
            rows = [row for row in csv.DictReader(source) if int(row["step"]) <= step]
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    path = run_dir / "samples.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in rows if row["step"] <= step), encoding="utf-8")


def main(variant, plan_path=DEFAULT_PLAN,
         initial_checkpoint=CHECKPOINTS_DIR / "arm_gpt_97m" / "best.pt",
         tokenizer_dir=DEFAULT_TOKENIZER_DIR, run_dir=None, config=None,
         resume=False, stop_at_step=1000, device_name="cuda", tensorboard=True):
    config = config or ContinuationConfig()
    if variant not in ("news", "mix"):
        raise ValueError("variant должен быть news или mix")
    if stop_at_step is not None and stop_at_step < 1:
        raise ValueError("stop_at_step должен быть положительным")
    stopping_step = min(config.total_steps, stop_at_step) if stop_at_step is not None else config.total_steps
    plan_path, initial_checkpoint, tokenizer_dir = map(Path, (plan_path, initial_checkpoint, tokenizer_dir))
    run_dir = Path(run_dir) if run_dir is not None else CHECKPOINTS_DIR / f"{initial_checkpoint.parent.name}_cpt_{variant}"
    if not resume and any((run_dir / name).exists() for name in ("config.json", "last.pt", "best.pt", "metrics.csv")):
        raise FileExistsError(f"В {run_dir} уже есть запуск: добавь --resume или выбери другой --run-dir")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Нужны CUDA и поддержка BF16")
    elif device.type != "cpu":
        raise ValueError("Поддерживаются cuda и cpu")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    effective_batch = config.micro_batch_size * config.gradient_accumulation_steps
    if effective_batch != plan["effective_batch_size"] or config.seed != plan["seed"]:
        raise ValueError("Батч и seed должны совпадать с планом смеси")
    if plan["code_sha256"] != file_sha256(Path(mixed_data.__file__)) or plan["numpy_version"] != np.__version__:
        raise ValueError("Код загрузчика или NumPy отличается от сохранённого плана смеси")
    source = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
    model_config = GPTConfig(**source["model_config"])
    if model_config.context_length != plan["context_length"]:
        raise ValueError("Контекст модели отличается от плана смеси")
    corpora = load_corpora(Path(plan["sources"]["armweb"]["data_dir"]),
                          Path(plan["sources"]["wikipedia"]["data_dir"]), tokenizer_dir, model_config.context_length)
    tokenizer_hash = file_sha256(tokenizer_dir / "tokenizer.json")
    if source["fingerprints"]["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("Начальная модель обучалась с другим токенизатором")
    for name, corpus in corpora.items():
        if file_sha256(corpus["directory"] / "metadata.json") != plan["sources"][name]["metadata_sha256"]:
            raise ValueError(f"Метаданные {name} отличаются от плана смеси")
        if corpus["metadata"]["vocab_size"] != model_config.vocab_size:
            raise ValueError("Словарь модели отличается от корпуса")
    fraction = 0.0 if variant == "news" else plan["wikipedia_fraction"]
    windows = {name: (len(corpus["train"]) - 1) // model_config.context_length
               for name, corpus in corpora.items()}
    sampler = MixedWindowSampler(windows, effective_batch, fraction, config.seed)
    streams = {name: corpus["train"] for name, corpus in corpora.items()}
    fingerprints = {
        "tokenizer_sha256": tokenizer_hash,
        "initial_checkpoint_sha256": file_sha256(initial_checkpoint),
        "mixture_plan_sha256": file_sha256(plan_path),
        "trainer_sha256": file_sha256(Path(__file__)),
        "shared_training_code_sha256": file_sha256(Path(__file__).with_name("train.py")),
    }
    origin = {"checkpoint": str(initial_checkpoint.resolve()), "step": source["step"],
              "tokens_seen": source["tokens_seen"]}
    checkpoint = None
    if resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if (checkpoint["train_config"] != asdict(config) or checkpoint["model_config"] != asdict(model_config)
                or checkpoint["fingerprints"] != fingerprints or checkpoint["variant"] != variant):
            raise ValueError("Для resume нужны те же модель, данные, код и настройки обучения")
        if checkpoint["step"] > stopping_step:
            raise ValueError("В чекпоинте уже более поздний шаг; увеличь stop_at_step")

    torch.manual_seed(config.seed)
    model = GPT(model_config).to(device)
    model.load_state_dict(source["model"] if checkpoint is None else checkpoint["model"])
    del source
    optimizer = make_optimizer(model, config.max_learning_rate, config.weight_decay)
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    step, best_val, previous_elapsed = 0, math.inf, 0.0
    initial_losses, last_losses = {}, {}
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, best_val = checkpoint["step"], checkpoint["best_val_loss"]
        previous_elapsed = checkpoint["elapsed_seconds"]
        initial_losses, last_losses = checkpoint["initial_validation_losses"], checkpoint["last_validation_losses"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        del checkpoint
        trim_logs(run_dir, step)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        info = {"model": asdict(model_config), "training": asdict(config), "variant": variant,
                "wikipedia_fraction": fraction, "fingerprints": fingerprints, "origin": origin,
                "optimizer_initialization": "fresh AdamW in both branches",
                "selection": "same validation loss weighted by the saved mixture plan in both branches",
                "parameters": model.num_parameters(), "torch_version": str(torch.__version__)}
        (run_dir / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(plan_path, run_dir / "mixture_plan.json")
        shutil.copytree(tokenizer_dir, run_dir / "tokenizer", dirs_exist_ok=True)
        code_dir = run_dir / "code"
        code_dir.mkdir(exist_ok=True)
        for filename in ("continue_pretrain.py", "mixed_data.py", "train.py", "training_utils.py", "model.py", "generate.py", "paths.py"):
            shutil.copy2(Path(__file__).with_name(filename), code_dir / filename)
    tb_writer = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=step + 1 if resume else None)
        except ImportError:
            print("TensorBoard не установлен; метрики сохраняются в CSV.", flush=True)
    csv_path = run_dir / "metrics.csv"
    header = not csv_path.exists()
    csv_file = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=LOG_FIELDS)
    if header:
        writer.writeheader()
    started = time.perf_counter()

    def elapsed():
        return previous_elapsed + time.perf_counter() - started

    def token_counts():
        return {name: count * model_config.context_length for name, count in sampler.counts_before(step).items()}

    def log(split, loss, lr=0.0, gradient_norm=None, speed=None):
        counts = token_counts()
        writer.writerow({"step": step, "tokens_seen": sum(counts.values()),
                         "armweb_tokens_seen": counts["armweb"], "wikipedia_tokens_seen": counts["wikipedia"],
                         "split": split, "loss": loss, "perplexity": math.exp(loss), "learning_rate": lr,
                         "gradient_norm": gradient_norm, "tokens_per_second": speed, "elapsed_seconds": elapsed()})
        csv_file.flush()
        if tb_writer:
            tb_writer.add_scalar(f"loss/{split}", loss, step)
            if split == "train":
                tb_writer.add_scalar("train/learning_rate", lr, step)
                tb_writer.add_scalar("train/gradient_norm", gradient_norm, step)
                tb_writer.add_scalar("train/tokens_per_second", speed, step)

    def payload(include_optimizer):
        counts = token_counts()
        state = {"model": model.state_dict(), "model_config": asdict(model_config), "train_config": asdict(config),
                 "variant": variant, "wikipedia_fraction": fraction, "step": step, "tokens_seen": sum(counts.values()),
                 "tokens_by_source": counts, "origin": origin, "best_val_loss": best_val,
                 "initial_validation_losses": initial_losses, "last_validation_losses": last_losses,
                 "fingerprints": fingerprints, "elapsed_seconds": elapsed()}
        if include_optimizer:
            state.update(optimizer=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(),
                         cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        return state

    def validation_and_save():
        nonlocal best_val, initial_losses, last_losses
        last_losses = {name: evaluate(model, corpus["validation"], config.micro_batch_size, device)
                       for name, corpus in corpora.items()}
        if step == 0:
            initial_losses = dict(last_losses)
        # Вес для выбора чекпоинта общий для двух веток, независимо от их train-смеси.
        score = ((1 - plan["wikipedia_fraction"]) * last_losses["armweb"]
                 + plan["wikipedia_fraction"] * last_losses["wikipedia"])
        for name, loss in last_losses.items():
            log(f"validation_{name}", loss)
        log("validation_mix", score)
        print(f"VALIDATION step {step:,}: ArmWeb {last_losses['armweb']:.4f}; "
              f"Wikipedia {last_losses['wikipedia']:.4f}; общий loss {score:.4f}", flush=True)
        if score < best_val:
            best_val = score
            atomic_save(payload(False), run_dir / "best.pt")
        if config.sample_tokens:
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as output:
                for index, prompt in enumerate(PROMPTS):
                    completion = generate_text(model, tokenizer, prompt, device,
                                               max_new_tokens=config.sample_tokens, seed=2026 + index)
                    output.write(json.dumps({"step": step, "prompt": prompt, "completion": completion,
                                             "seed": 2026 + index}, ensure_ascii=False) + "\n")
                    if index == 0:
                        print(f"Пример: {prompt}{completion[:180]}", flush=True)
        atomic_save(payload(True), run_dir / "last.pt")
        if tb_writer:
            tb_writer.flush()

    print(f"Ветка: {variant}; ArmWeb {1 - fraction:.0%}; Wikipedia {fraction:.0%}", flush=True)
    print(f"Начальные веса: {initial_checkpoint}; исходный шаг {origin['step']:,}", flush=True)
    print(f"Параметров: {model.num_parameters():,}; контекст: {model_config.context_length}; "
          f"micro batch {config.micro_batch_size}; накопление {config.gradient_accumulation_steps}", flush=True)
    print(f"План: {config.total_steps:,} шагов; текущая остановка: {stopping_step:,}; "
          f"токенов/шаг: {effective_batch * model_config.context_length:,}", flush=True)
    print(f"LR: warmup {config.warmup_steps} до {config.max_learning_rate:g}, cosine до {config.min_learning_rate:g} "
          f"на все {config.total_steps:,} шагов\nКаталог: {run_dir}", flush=True)
    attention = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if device.type == "cuda" else nullcontext()
    try:
        with attention:
            if not resume:
                validation_and_save()
            model.train()
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            interval_tokens = 0
            last_log_time = time.perf_counter()
            while step < stopping_step:
                source_ids, indices = sampler.batch(step)
                lr = learning_rate(step, config.total_steps, config)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                for first in range(0, effective_batch, config.micro_batch_size):
                    end = min(first + config.micro_batch_size, effective_batch)
                    x, y = make_mixed_batch(streams, source_ids[first:end], indices[first:end],
                                            model_config.context_length, device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logits, loss = model(x, y)
                    loss_sum.add_(loss.detach().double() * y.numel())
                    interval_tokens += y.numel()
                    (loss * ((end - first) / effective_batch)).backward()
                    del logits, loss, x, y
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % config.log_interval == 0 or step == stopping_step:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    mean_loss = (loss_sum / interval_tokens).item()
                    if not math.isfinite(mean_loss):
                        raise RuntimeError("Train loss не является конечным")
                    speed = interval_tokens / (time.perf_counter() - last_log_time)
                    log("train", mean_loss, lr, norm.item(), speed)
                    print(f"step {step:,}/{config.total_steps:,} | train loss {mean_loss:.4f} | "
                          f"lr {lr:.2e} | {speed:,.0f} токенов/с", flush=True)
                    loss_sum.zero_()
                    interval_tokens = 0
                    last_log_time = time.perf_counter()
                if step % config.eval_interval == 0 or step == stopping_step:
                    validation_started = time.perf_counter()
                    validation_and_save()
                    last_log_time += time.perf_counter() - validation_started
    except KeyboardInterrupt:
        print(f"\nЗапуск прерван. Последний сохранённый шаг: {run_dir / 'last.pt'}", flush=True)
        print(f"Продолжение: --variant {variant} --resume --stop-at-step {stopping_step}", flush=True)
        return
    finally:
        csv_file.close()
        if tb_writer:
            tb_writer.close()

    summary = {"status": "complete" if step == config.total_steps else "pilot_complete",
               "variant": variant, "step": step, "total_steps": config.total_steps,
               "tokens_seen": sum(token_counts().values()), "tokens_by_source": token_counts(),
               "initial_validation_losses": initial_losses, "last_validation_losses": last_losses,
               "best_validation_mix_loss": best_val, "elapsed_seconds": elapsed()}
    path = run_dir / "progress.json"
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"\n{summary['status']}: {step:,}/{config.total_steps:,}; новых токенов: {summary['tokens_seen']:,}", flush=True)
    for name in SOURCE_NAMES:
        print(f"{name}: validation loss {initial_losses[name]:.4f} → {last_losses[name]:.4f}", flush=True)
    print(f"Состояние модели и оптимизатора: {run_dir / 'last.pt'}", flush=True)
    if step < config.total_steps:
        print(f"Для продолжения: --variant {variant} --resume --stop-at-step {config.total_steps}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("news", "mix"), required=True)
    parser.add_argument("--plan-path", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--initial-checkpoint", type=Path, default=CHECKPOINTS_DIR / "arm_gpt_97m" / "best.pt")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-at-step", type=int, default=1000)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.variant, args.plan_path, args.initial_checkpoint, args.tokenizer_dir, args.run_dir,
         resume=args.resume, stop_at_step=args.stop_at_step, device_name=args.device, tensorboard=not args.no_tensorboard)
