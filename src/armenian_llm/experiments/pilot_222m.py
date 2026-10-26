"""Претрейн 222M с фиксированными настройками.

Пилот до шага 2000: %run -m armenian_llm.experiments.pilot_222m
Продолжение до конца: %run -m armenian_llm.experiments.pilot_222m --resume
График learning rate в обоих случаях рассчитан на четыре эпохи.
"""

import argparse
import importlib

from armenian_llm.modeling.gpt import GPTConfig
from armenian_llm.paths import CHECKPOINTS_DIR
import armenian_llm.training.pretrain as train
def main(resume=False):
    # В Jupyter мог остаться старый импорт train.py без stop_at_step.
    importlib.reload(train)

    model_config = GPTConfig(
        vocab_size=16_000,
        context_length=1024,
        n_layers=16,
        n_heads=16,
        d_model=1024,
        ffn_hidden_dim=2816,
        dropout=0.0,
    )
    train_config = train.TrainConfig(
        epochs=4,
        micro_batch_size=32,
        gradient_accumulation_steps=4,
        max_learning_rate=4e-4,
        min_learning_rate=4e-5,
        warmup_steps=400,
        weight_decay=0.1,
        gradient_clip=1.0,
        eval_interval=500,
        log_interval=50,
        sample_tokens=96,
        seed=42,
    )

    train.main(
        run_dir=CHECKPOINTS_DIR / "arm_gpt_222m",
        train_config=train_config,
        model_config=model_config,
        resume=resume,
        stop_at_step=None if resume else 2000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(resume=args.resume)
