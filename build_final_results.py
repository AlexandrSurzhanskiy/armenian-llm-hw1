"""Таблицы, графики и ноутбук результатов из сохранённых отчётов ДЗ.

Не загружает веса, не запускает обучение и не выполняет код из отчётов.
Пример: python build_final_results.py --reports-dir final_reports_20261026_115500
"""

import argparse
import base64
from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np


LABELS = {
    "arm_gpt_35m": "35M", "arm_gpt_97m": "97M",
    "arm_gpt_222m": "222M · пилот", "arm_gpt_97m_cpt_news": "97M + news",
    "arm_gpt_97m_cpt_mix": "97M + mix", "arm_gpt_97m_sft": "97M + SFT",
}
RUNS = tuple(LABELS)
COLORS = {"35M": "#326FA8", "97M": "#D16C32", "222M · пилот": "#8357A5"}


class Reports:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}
        self.issues = []

    def text(self, relative):
        path = self.root / relative
        content = path.read_bytes()
        self.sources[str(relative)] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        return content.decode("utf-8")

    def json(self, relative):
        return json.loads(self.text(relative))

    def external_json(self, path):
        path = Path(path).resolve()
        content = path.read_bytes()
        self.sources[str(path)] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        return json.loads(content)


def recover_notebook_metrics(reports, run, config):
    """Считываем только текст сохранённых outputs; ячейки не исполняются."""
    rows = {}
    source = "project/Untitled.ipynb"
    notebook = reports.json(source)
    train_pattern = re.compile(r"^step ([\d,]+)/[\d,]+ \| train loss ([\d.]+) \| lr ([\deE+.-]+)")
    val_pattern = re.compile(r"^VALIDATION step ([\d,]+): loss ([\d.]+), perplexity ([\d.]+)")
    for cell in notebook.get("cells", []):
        text = "".join("".join(out.get("text", [])) for out in cell.get("outputs", []))
        if f"/checkpoints/{run}\n" not in text:
            continue
        for line in text.splitlines():
            train, val = train_pattern.match(line), val_pattern.match(line)
            if not train and not val:
                continue
            match = train or val
            step = int(match[1].replace(",", ""))
            loss = float(match[2])
            completed, remaining = divmod(step, config["steps_per_epoch"])
            targets = completed * config["train_windows"] * config["model"]["context_length"]
            targets += remaining * config["training"]["micro_batch_size"] * config["training"]["gradient_accumulation_steps"] * config["model"]["context_length"]
            row = {"step": step, "tokens_seen": targets, "split": "train" if train else "validation", "loss": loss}
            if train:
                row["learning_rate"] = float(match[3])
            else:
                row["perplexity"] = float(match[3])
            rows[(row["step"], row["split"])] = row
    reports.issues.append({"run": run, "kind": "notebook_log_fallback", "source": source,
                           "note": "Loss округлён в консольном логе до 4 знаков; пропуски не заполняются."})
    return sorted(rows.values(), key=lambda row: (row["step"], row["split"])), source


def numeric_metrics(rows):
    return [{key: value if key == "split" else (float(value) if value not in (None, "") else None)
             for key, value in row.items()} for row in rows]


def load_metrics(reports, run, config, final_evaluation=None):
    relative = f"artifacts/checkpoints/{run}/metrics.csv"
    exported = (final_evaluation or {}).get("saved_pretrain_curves", {}).get(run, {})
    if exported.get("rows"):
        source = "final_evaluation.json#saved_pretrain_curves/" + run
        return numeric_metrics(exported["rows"]), source, "exported_csv"
    try:
        rows = list(csv.DictReader(reports.text(relative).splitlines()))
        return numeric_metrics(rows), relative, "csv"
    except OSError as error:
        reports.issues.append({"run": run, "kind": "csv_unavailable", "path": relative, "error": type(error).__name__})
        rows, source = recover_notebook_metrics(reports, run, config)
        return rows, source, "notebook_console_rounded"


def make_tables(reports, final_evaluation=None, reported_test_rows=None):
    heldout = reports.json("artifacts/evaluation/heldout_comparison/results.json")
    selected_steps = {row["model"]: row["step"] for row in heldout["results"]}
    configs, metrics, run_rows = {}, {}, []
    for run in RUNS:
        prefix = f"artifacts/checkpoints/{run}/"
        config = reports.json(prefix + "config.json")
        summary_name = "progress.json" if run.endswith(("222m", "cpt_news", "cpt_mix")) else "summary.json"
        summary = reports.json(prefix + summary_name)
        curve, curve_source, curve_kind = load_metrics(reports, run, config, final_evaluation)
        configs[run], metrics[run] = config, curve
        step = int(summary.get("steps", summary.get("step")))
        is_sft = config.get("stage") == "sft"
        is_cpt = "variant" in config
        selection = "validation_mix" if is_cpt else "validation"
        validation = [row for row in curve if row["split"] == selection]
        best_step = summary.get("best_step", selected_steps.get(run))
        if best_step is None and validation:
            best_step = int(min(validation, key=lambda row: row["loss"])["step"])
        best_loss = summary.get("best_validation_answer_loss" if is_sft else
                                "best_validation_mix_loss" if is_cpt else "best_validation_loss")
        stage_tokens = summary.get("supervised_tokens_seen" if is_sft else "tokens_seen")
        origin_tokens = config.get("origin", {}).get("tokens_seen", config.get("provenance", {}).get("source_tokens_seen", 0))
        run_rows.append({
            "run": run, "model": LABELS[run], "stage": "sft" if is_sft else "continued_pretrain" if is_cpt else "pretrain",
            "parameters": config["parameters"], "layers": config["model"]["n_layers"],
            "d_model": config["model"]["d_model"], "context": config["model"]["context_length"],
            "last_step": step, "planned_steps": config["total_steps"] if "total_steps" in config else config["training"]["total_steps"],
            "best_step": best_step, "stage_tokens": stage_tokens, "origin_pretrain_tokens": origin_tokens,
            "cumulative_pretrain_tokens": origin_tokens + (0 if is_sft else stage_tokens),
            "status": summary.get("status", "complete"), "selection_split": selection,
            "best_validation_loss": best_loss, "best_validation_perplexity": math.exp(best_loss),
            "recorded_elapsed_seconds": summary.get("elapsed_seconds"), "curve_source": curve_source,
            "curve_format": curve_kind, "last_curve_step": int(max((r["step"] for r in curve), default=0)),
            "last_validation_losses": summary.get("last_validation_losses"),
        })

    corpus_rows = []
    for corpus, directory in [("ArmWeb", "bpe_16000_500000"), ("Wikipedia", "wikipedia_hy_20231101/bpe_16000_500000")]:
        path = f"data_reports/tokenized/{directory}/metadata.json"
        metadata = reports.json(path)
        for split, info in metadata["splits"].items():
            corpus_rows.append({"corpus": corpus, "split": split, "documents": info["documents"],
                                "tokens": info["tokens"], "text_bytes": info["text_bytes"],
                                "median_document_tokens": info["document_tokens"]["median"]})
    arm = reports.json("data_reports/processed/armweb/stats.json")
    wiki = reports.json("data_reports/processed/wikipedia_hy_20231101/stats.json")
    cleaning_rows = [{"corpus": "ArmWeb train", "input_documents": arm["splits"]["train"]["input_documents"],
                      "kept_documents": arm["splits"]["train"]["kept_documents"],
                      "repetition": arm["splits"]["train"]["dropped_repetition"],
                      "language": arm["splits"]["train"]["dropped_language"]},
                     {"corpus": "Wikipedia all splits", "input_documents": wiki["counts"]["input_documents"],
                      "kept_documents": wiki["counts"]["kept_documents"], **wiki["rejections"]}]
    tokenizer_rows = []
    validation_hashes = set()
    for directory in ("bpe_16000_100000", "bpe_8000_500000", "bpe_16000_500000", "bpe_32000_500000"):
        path = "artifacts/tokenizers/" + directory + "/"
        info, cfg = reports.json(path + "metrics.json"), reports.json(path + "training_config.json")
        validation_hashes.add(cfg["data_sha256"]["validation"])
        tokenizer_rows.append({"tokenizer": directory, "train_documents": info["train_sample"]["documents"],
                               "vocab_size": info["vocab_size_actual"], "seconds": info["training_seconds"],
                               **{key: info["validation"][key] for key in ["tokens", "characters_per_token", "bytes_per_token", "tokens_per_whitespace_word", "roundtrip_errors"]},
                               "unused_tokens": info["unused_tokens_on_train_sample"],
                               "tokens_seen_1_to_9": info["tokens_seen_1_to_9_times_on_train_sample"]})
    if len(validation_hashes) != 1:
        raise ValueError("Сравнение токенизаторов сделано на разных validation-файлах")
    frequency_path = "artifacts/tokenizers/bpe_16000_500000/token_frequencies.csv"
    frequency_rows = [row for row in csv.DictReader(reports.text(frequency_path).splitlines()) if row["is_special"].lower() == "false"]
    frequency_table = []
    total_appearances = sum(int(row["train_count"]) for row in frequency_rows)
    for label, low, high in [("0", 0, 0), ("1–9", 1, 9), ("10–99", 10, 99), ("100–999", 100, 999), ("1000+", 1000, float("inf"))]:
        values = [int(row["train_count"]) for row in frequency_rows if low <= int(row["train_count"]) <= high]
        frequency_table.append({"frequency": label, "tokens": len(values), "vocab_fraction": len(values) / len(frequency_rows),
                                "appearance_fraction": sum(values) / total_appearances})
    benchmarks = []
    for directory, label in [("training_benchmark", "35M"), ("training_benchmark_97m", "97M"), ("training_benchmark_222m", "222M")]:
        data = reports.json(f"artifacts/{directory}/results.json")
        for row in data["results"]:
            if row["status"] == "ok":
                benchmarks.append({"model": label, "micro_batch": row["micro_batch_size"],
                                   "accumulation": row["gradient_accumulation_steps"], "tokens_per_second": row["tokens_per_second"],
                                   "allocated_GiB": row["peak_cuda_allocated_bytes"] / 1024**3,
                                   "reserved_GiB": row["peak_cuda_reserved_bytes"] / 1024**3,
                                   "timed_steps": data["timed_optimizer_steps"], "warmup_steps": data["warmup_optimizer_steps"]})
    qa_rows = []
    protocols = [("armenian_facts_v1_zero_shot", ["arm_gpt_35m"]),
                 ("armenian_facts_cpt_mix_zero_shot", ["arm_gpt_97m", "arm_gpt_97m_cpt_mix"]),
                 ("armenian_facts_sft_zero_shot", ["arm_gpt_97m_sft"]),
                 ("armenian_facts_v1", ["arm_gpt_35m"]),
                 ("armenian_facts_cpt_mix_two_shot", ["arm_gpt_97m", "arm_gpt_97m_cpt_mix"])]
    for directory, models in protocols:
        data = reports.json(f"artifacts/evaluation/{directory}/summary.json")
        for name in models:
            m = data["metrics"][name]
            qa_rows.append({"model": LABELS[name], "shots": data["protocol"]["shots"], "questions": m["questions"],
                            "mc_sum": m["choice_sum"]["correct"], "mc_mean": m["choice_mean"]["correct"],
                            "free_exact": m["free_exact"]["correct"],
                            "all_three_mc_sum": m["choice_sum"]["all_paraphrases_correct"],
                            "all_three_free": m["free_exact"]["all_paraphrases_correct"], "source": directory})
    heldout_rows = [{"corpus": "armweb", "source_precision": "original_json", **{key: row[key] for key in ["model", "split", "step", "scored_tokens", "loss", "perplexity"]}}
                    for row in heldout["results"]]
    if final_evaluation is not None:
        expected = {(model, corpus, split)
                    for model in ("arm_gpt_97m", "arm_gpt_97m_cpt_mix")
                    for corpus, split in (("armweb", "test_iid"), ("armweb", "test_tail"), ("wikipedia", "test"))}
        actual = [(row["model"], row["corpus"], row["split"]) for row in final_evaluation["results"]]
        if final_evaluation.get("status") != "complete" or len(actual) != 6 or set(actual) != expected:
            raise ValueError("Финальная оценка должна содержать шесть завершённых model/corpus/split")
        combined = {(row["model"], row["corpus"], row["split"]): row for row in heldout_rows}
        for row in final_evaluation["results"]:
            model_info = final_evaluation["models"][row["model"]]
            if model_info["model_config"] != configs[row["model"]]["model"]:
                raise ValueError("Архитектура финальной оценки не совпадает с сохранённым запуском")
            combined[(row["model"], row["corpus"], row["split"])] = {**row, "step": model_info["step"]}
        heldout_rows = sorted(combined.values(), key=lambda row: (RUNS.index(row["model"]), row["corpus"], row["split"]))
    elif reported_test_rows:
        # Числа из сохранённой таблицы outputs не выдаём за исходный полный JSON.
        expected = {(model, corpus, split)
                    for model in ("arm_gpt_97m", "arm_gpt_97m_cpt_mix")
                    for corpus, split in (("armweb", "test_iid"), ("armweb", "test_tail"), ("wikipedia", "test"))}
        keys = [(row["model"], row["corpus"], row["split"]) for row in reported_test_rows]
        if len(keys) != 6 or set(keys) != expected:
            raise ValueError("В выводе ноутбука ожидались шесть строк финального test")
        combined = {(row["model"], row["corpus"], row["split"]): row for row in heldout_rows}
        for row in reported_test_rows:
            key = (row["model"], row["corpus"], row["split"])
            if key in combined:
                original = combined[key]
                if round(original["loss"], 4) != row["loss"] or round(original["perplexity"], 2) != row["perplexity"]:
                    raise ValueError("Тест исходной 97M в архиве не совпадает с выводом ноутбука")
                original["confirmed_by_notebook_table"] = row["source"]
            else:
                run = next(info for info in run_rows if info["run"] == row["model"])
                combined[key] = {**row, "step": run["best_step"], "scored_tokens": None,
                                 "source_precision": "notebook_table_loss_4dp_perplexity_2dp"}
        heldout_rows = sorted(combined.values(), key=lambda row: (RUNS.index(row["model"]), row["corpus"], row["split"]))
    sft_test = reports.json("artifacts/evaluation/sft_heldout_comparison/results.json")
    sft_rows = [{key: row[key] for key in ["model", "step", "examples", "scored_answer_tokens_including_eos", "answer_loss", "answer_perplexity"]} for row in sft_test["results"]]
    controlled_rows = []
    for name in ["arm_gpt_97m_cpt_news", "arm_gpt_97m_cpt_mix"]:
        for row in metrics[name]:
            if row["split"].startswith("validation") and row["step"] <= 1000:
                controlled_rows.append({"model": LABELS[name], **{k: row[k] for k in ["step", "tokens_seen", "split", "loss"]}})
    tables = {"runs": run_rows, "corpora": corpus_rows, "cleaning": cleaning_rows,
              "tokenizers": tokenizer_rows, "token_frequencies": frequency_table, "batch_benchmark": benchmarks,
              "heldout": heldout_rows, "qa": qa_rows, "sft_heldout": sft_rows, "controlled_continuation": controlled_rows}
    return tables, metrics, configs


def save_plot(fig, directory, name):
    fig.savefig(directory / (name + ".png"), dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def draw_broken_curve(ax, rows, color, label, xkey="tokens_seen", scale=1e9):
    rows = sorted(rows, key=lambda row: row["step"])
    groups, current = [], []
    for row in rows:
        if current and row["step"] - current[-1]["step"] > 1000:
            groups.append(current); current = []
        current.append(row)
    if current:
        groups.append(current)
    for i, group in enumerate(groups):
        ax.plot([r[xkey] / scale for r in group], [r["loss"] for r in group], color=color,
                marker="o", markersize=3.5, linewidth=1.7, label=label if i == 0 else None)


def figures(tables, metrics, destination):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False})
    names = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for run in ["arm_gpt_35m", "arm_gpt_97m", "arm_gpt_222m"]:
        validation = [dict(row) for row in metrics[run] if row["split"] == "validation"]
        info = next(r for r in tables["runs"] if r["run"] == run)
        if max((r["step"] for r in validation), default=-1) < info["last_step"]:
            validation.append({"step": info["last_step"], "tokens_seen": info["stage_tokens"], "loss": info["best_validation_loss"]})
        for ax in axes:
            draw_broken_curve(ax, validation, COLORS[LABELS[run]], LABELS[run])
    axes[0].set(title="Validation при обучении с нуля", xlabel="Обработано целевых токенов, млрд", ylabel="Cross-entropy, нат/токен")
    axes[1].set(title="Те же точки: диапазон loss 2.4–4.5", xlabel="Обработано целевых токенов, млрд", ylim=(2.4, 4.5))
    for ax in axes:
        ax.grid(axis="y", alpha=.18); ax.legend(frameon=False)
    partial = [r for r in tables["runs"] if r["run"] == "arm_gpt_97m" and r["last_curve_step"] < r["last_step"]]
    if partial:
        info = partial[0]
        axes[1].annotate("97M: конечная точка из сводки\nпромежуточный лог недоступен",
                         xy=(info["stage_tokens"] / 1e9, info["best_validation_loss"]),
                         xytext=(1.05, 2.77), fontsize=8, color="#8B451F",
                         arrowprops={"arrowstyle": "->", "color": "#D16C32", "lw": .8})
    names.append(save_plot(fig, destination, "01_pretrain_validation"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, split in enumerate(["validation_armweb", "validation_wikipedia"]):
        title = "ArmWeb" if column == 0 else "Wikipedia"
        for run, label, color in [("arm_gpt_97m_cpt_news", "100% news", "#727B84"), ("arm_gpt_97m_cpt_mix", "70% news + 30% wiki", "#168C83")]:
            rows = [r for r in metrics[run] if r["split"] == split and r["step"] <= 1000]
            axes[0, column].plot([r["step"] for r in rows], [r["loss"] for r in rows], "o-", color=color, label=label)
        rows = [r for r in metrics["arm_gpt_97m_cpt_mix"] if r["split"] == split]
        axes[1, column].plot([r["step"] for r in rows], [r["loss"] for r in rows], "o-", color="#168C83", label="mix")
        axes[1, column].axhline(rows[0]["loss"], linestyle="--", color="#727B84", alpha=.7, label="исходная 97M")
        axes[0, column].set(title=f"{title}: равный бюджет 1000 шагов", xlabel="Дополнительные шаги AdamW", ylabel="Validation loss")
        axes[1, column].set(title=f"{title}: смесь до 5000 шагов", xlabel="Дополнительные шаги AdamW", ylabel="Validation loss")
        for row in range(2):
            axes[row, column].grid(axis="y", alpha=.18); axes[row, column].legend(frameon=False, fontsize=9)
    names.append(save_plot(fig, destination, "02_data_mixture"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    tokenizers = [r for r in tables["tokenizers"] if r["train_documents"] == 500000]
    x = np.arange(len(tokenizers))
    bars = axes[0].bar(x, [r["tokens_per_whitespace_word"] for r in tokenizers], color=["#83AEC9", "#168C83", "#AEC9B8"])
    axes[0].set_xticks(x, [f"{r['vocab_size']//1000}k" for r in tokenizers])
    axes[0].bar_label(bars, fmt="%.3f", padding=3)
    axes[0].set(title="BPE: одинаковые 500k train и 5k validation", xlabel="Размер словаря", ylabel="Токенов на слово по пробелам", ylim=(0, 2.1))
    f = tables["token_frequencies"]
    bars = axes[1].bar([r["frequency"] for r in f], [r["tokens"] for r in f], color="#326FA8")
    axes[1].set_yscale("log"); axes[1].bar_label(bars, padding=3, fontsize=9)
    axes[1].set(title="Выбранный BPE 16k: частоты в train-выборке", xlabel="Число появлений токена", ylabel="Число обычных токенов (лог. шкала)", ylim=(10, 40000))
    for ax in axes: ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    names.append(save_plot(fig, destination, "03_tokenizers"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for label, color in [("35M", "#326FA8"), ("97M", "#D16C32"), ("222M", "#8357A5")]:
        rows = [r for r in tables["batch_benchmark"] if r["model"] == label]
        axes[0].plot([r["micro_batch"] for r in rows], [r["tokens_per_second"] / 1000 for r in rows], "o-", color=color, label=label)
        axes[1].plot([r["micro_batch"] for r in rows], [r["allocated_GiB"] for r in rows], "o-", color=color, label=label)
    axes[0].set(title="Скорость на одной H100 80 GB", ylabel="Тысяч целевых токенов/с")
    axes[1].set(title="Пик выделенной CUDA-памяти", ylabel="Allocated, ГиБ")
    for ax in axes:
        ax.set(xlabel="Micro batch, effective batch = 128", xticks=[16, 32, 64, 128]); ax.grid(axis="y", alpha=.18); ax.legend(frameon=False)
    names.append(save_plot(fig, destination, "04_batch_benchmark"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    qa = [r for r in tables["qa"] if r["shots"] == 0]
    x = np.arange(len(qa))
    for ax, key, title in [(axes[0], "mc_sum", "Выбор из 4 вариантов, sum score"), (axes[1], "free_exact", "Свободный ответ, exact match")]:
        bars = ax.bar(x, [r[key] for r in qa], color=["#83AEC9", "#326FA8", "#168C83", "#8357A5"])
        ax.set_xticks(x, [r["model"] for r in qa], rotation=12)
        ax.bar_label(bars, labels=[f"{r[key]}/24" for r in qa], padding=3)
        ax.set(title=title, ylabel="Верных вопросов из 24", ylim=(0, 25), yticks=[0, 6, 12, 18, 24])
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    axes[0].axhline(6, linestyle="--", color="#777777", label="25%: случайный выбор вариантов")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    names.append(save_plot(fig, destination, "05_qa_zero_shot"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for split, label, color in [("train", "train: среднее между логами", "#326FA8"), ("validation", "validation: только ответы + EOS", "#D16C32")]:
        rows = [r for r in metrics["arm_gpt_97m_sft"] if r["split"] == split]
        axes[0].plot([r["step"] for r in rows], [r["loss"] for r in rows], "o-", color=color, label=label, markersize=4)
    axes[0].axvline(98, linestyle="--", color="#777777", alpha=.7)
    axes[0].set(title="SFT: лучший validation на шаге 98", xlabel="Шаги AdamW", ylabel="Answer loss")
    axes[0].legend(frameon=False, fontsize=8)
    sft = tables["sft_heldout"]
    bars = axes[1].bar(["97M", "97M + SFT"], [r["answer_perplexity"] for r in sft], color=["#326FA8", "#8357A5"])
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(title="SFT test: 391 пара, teacher forcing", ylabel="Answer perplexity", ylim=(0, 35))
    for ax in axes: ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    names.append(save_plot(fig, destination, "06_sft"))
    return names


def table_md(rows, columns):
    def value(x, key):
        if x is None: return "—"
        if isinstance(x, float): return f"{x:.2f}" if "perplexity" in key else f"{x:.4f}"
        if isinstance(x, int): return f"{x:,}".replace(",", "\u202f")
        return str(x).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(title for _, title in columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n" + "\n".join("| " + " | ".join(value(row.get(key), key) for key, _ in columns) + " |" for row in rows)


def inspect_inference(reports, path):
    """Читаем сохранённые outputs; пользовательский код не выполняется."""
    notebook = reports.external_json(path)
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "".join(cell["source"]).strip()]
    streams = "\n".join("".join(out.get("text", [])) for cell in code for out in cell.get("outputs", [])
                        if out["output_type"] == "stream")
    errors = [out for cell in code for out in cell.get("outputs", []) if out["output_type"] == "error"]
    test_rows = []
    for index, cell in enumerate(notebook["cells"]):
        for out in cell.get("outputs", []):
            markdown = "".join(out.get("data", {}).get("text/markdown", []))
            for line in markdown.splitlines():
                fields = [part.strip() for part in line.strip().strip("|").split("|")]
                if len(fields) == 5 and fields[0] in ("arm_gpt_97m", "arm_gpt_97m_cpt_mix"):
                    test_rows.append({"model": fields[0], "corpus": fields[1], "split": fields[2],
                                      "loss": float(fields[3]), "perplexity": float(fields[4]),
                                      "source": f"{Path(path).name}#cell-{index + 1}/outputs/text-markdown"})
    complete = bool(code) and all(cell.get("execution_count") is not None for cell in code) and not errors
    def extract(pattern):
        found = re.search(pattern, streams)
        return int(found[1].replace(",", "")) if found else None
    return {"source": str(Path(path).resolve()), "all_code_cells_executed": complete,
            "code_cells": len(code), "errors": len(errors),
            "samples_saved": extract(r"Примеров сохранено: (\d+)"),
            "free_exact_correct": extract(r"Свободный ответ: (\d+)/24"),
            "all_paraphrases_correct": extract(r"Все три формулировки верны: (\d+)/8"),
            "checkpoint_step": extract(r"Шаг выбранного checkpoint: ([\d,]+)"),
            "reported_test_rows": test_rows}


def make_notebook(tables, figure_names, output, issues, final_evaluation=None, inference=None):
    nb = nbformat.v4.new_notebook()
    cells = []
    def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
    def picture(name):
        data = base64.b64encode((output / "figures" / (name + ".png")).read_bytes()).decode()
        cells.append(nbformat.v4.new_code_cell(
            f"display(Image(filename=str(RESULTS_DIR / 'figures' / '{name}.png')))",
            execution_count=len([c for c in cells if c.cell_type == "code"]) + 1,
            outputs=[nbformat.v4.new_output("display_data", data={"image/png": data, "text/plain": [name]})]))
    md("# Армянская LLM с нуля: результаты ДЗ №1\n\nЗдесь собраны сохранённые результаты, без повторного обучения. Основной кандидат для демонстрации — **97M после continued pretraining на смеси 70% ArmWeb / 30% Wikipedia**.\n\nСвязный текст, выбор ответа из вариантов и правильный свободный ответ — разные критерии. В этом отчёте они не подменяют друг друга.")
    cells.append(nbformat.v4.new_code_cell("from pathlib import Path\nfrom IPython.display import display, Image\nimport json\n\n# Ищем комплект отчёта вместе с его рисунками.\nRESULTS_DIR = next((p for p in (Path.cwd(), Path.cwd() / 'submission_results')\n                    if (p / 'results.json').is_file() and (p / 'figures' / '01_pretrain_validation.png').is_file()), None)\nif RESULTS_DIR is None:\n    raise FileNotFoundError('Открой ноутбук из распакованного каталога submission_results')\nsummary = json.loads((RESULTS_DIR / 'results.json').read_text(encoding='utf-8'))", execution_count=1, outputs=[]))
    md("## 1. Данные\n\nДля ArmWeb удалялись документы с повторяемостью 10-грамм выше 20% и долей армянских букв ниже 80%. Для Wikipedia удалялись служебные хвосты, применялись отдельный языковой фильтр и извлечение армянских блоков; точные дубли удалялись до разбиения на выборки. Просмотренные при настройке очистки статьи принудительно отнесены к train.\n\n" + table_md(tables["cleaning"], [("corpus", "Корпус"), ("input_documents", "До очистки"), ("kept_documents", "После очистки"), ("repetition", "Повторы"), ("language", "Языковой фильтр")]) + "\n\nВ первой строке только train ArmWeb, во второй — вся Wikipedia; причины пропусков в строке Wikipedia отдельно сохранены в JSON. Пересечения Wikipedia и ArmWeb не проверялись; удаление точных дублей не гарантирует отсутствия близких статей между выборками.\n\n" + table_md(tables["corpora"], [("corpus", "Корпус"), ("split", "Split"), ("documents", "Документы"), ("tokens", "Токены с BOS/EOS"), ("median_document_tokens", "Медиана длины")]))
    md("## 2. Токенизатор\n\nByte-level BPE обучен самостоятельно. Все четыре варианта оценены на одном validation-файле из 5000 документов; совпадение SHA256 проверено. В модель выбран словарь 16k, обученный на 500k документов. Это компромисс между длиной последовательностей и размером embedding/output-слоя; оптимальность выбора для качества модели отдельно не доказана.\n\n" + table_md(tables["tokenizers"], [("train_documents", "Train документов"), ("vocab_size", "Словарь"), ("characters_per_token", "Символов/токен"), ("tokens_per_whitespace_word", "Токенов/слово"), ("unused_tokens", "Частота 0 в train"), ("tokens_seen_1_to_9", "Частота 1–9"), ("roundtrip_errors", "Ошибок decode")]) + "\n\nУвеличение выборки со 100k до 500k при словаре 16k почти не изменило сжатие validation-текстов. На 500k уменьшилось число неиспользуемых и редких токенов; частоты при разных размерах train-выборки сами по себе несопоставимы как вероятность появления.")
    picture("03_tokenizers")
    md("## 3. Архитектура и запуски\n\nDecoder-only Transformer: causal self-attention, RoPE, pre-RMSNorm, SwiGLU и общие веса входного embedding и выходной проекции. Веса и токенизатор обучены с нуля. Контекст — 1024 токена. Использованы AdamW, BF16 autocast при FP32-параметрах и Flash Attention через PyTorch SDPA; отдельный пакет Flash Attention 2 не использовался. Основной effective batch — 128 последовательностей, micro batch 32 и накопление 4.\n\n" + table_md(tables["runs"], [("model", "Модель"), ("parameters", "Параметры"), ("stage", "Стадия"), ("last_step", "Последний шаг"), ("planned_steps", "План шагов"), ("best_step", "Лучший шаг"), ("stage_tokens", "Токены стадии"), ("status", "Статус")]) + "\n\nУ SFT токены стадии — только supervised-токены ответа и EOS; их нельзя складывать с pretrain-токенами как одинаковый бюджет. Для 97M + mix суммарный pretrain — **2 992 502 784 целевых токена**, включая 2 337 142 784 токена исходной модели. **222M — пилот 2000/35664 шагов**, не завершённая четырёхэпоховая модель. Новостной контроль завершён как пилот 1000/5000; лучший checkpoint этой ветки соответствует исходному шагу 0.\n\n" + table_md(tables["runs"], [("model", "Модель"), ("selection_split", "Validation для выбора"), ("best_validation_loss", "Лучший loss"), ("best_validation_perplexity", "Perplexity")]) + "\n\nУ continued pretraining `validation_mix` — фиксированная сумма 0.7 × ArmWeb loss + 0.3 × Wikipedia loss; у SFT `validation` относится только к ответам. Эти столбцы не являются одним общим тестом для всех стадий.")
    fallback_runs = [r for r in tables["runs"] if r["curve_format"] == "notebook_console_rounded"]
    if fallback_runs:
        curve_note = "Для недоступных CSV использован сохранённый текст outputs из `Untitled.ipynb`, с округлением loss до четырёх знаков. "
        for row in fallback_runs:
            curve_note += f"**{row['model']}: доступный лог до шага {row['last_curve_step']}, последний шаг обучения {row['last_step']}.** "
        curve_note += "Конечная точка при отсутствии полного лога взята отдельно из `summary.json` и метаданных выбранного checkpoint; линия через отсутствующий участок не проводится. Если CSV доступны при повторном запуске сборщика, они используются автоматически."
    else:
        curve_note = "Кривые построены по исходным `metrics.csv` из каталогов запусков. Для 35M/97M при наличии финального экспорта используются полные строки CSV, сохранённые в `final_evaluation.json`; округлённый консольный лог не подставляется вместо них."
    md("### Кривые validation\n\n" + curve_note + "\n\n35M обучена на 1.169 млрд токенов, 97M — на 2.337 млрд. Итоговое сравнение смешивает эффект размера и дополнительного обучения. Разница learning rate у 222M — ещё один фактор.")
    picture("01_pretrain_validation")
    md("### Скорость и память\n\n" + table_md(tables["batch_benchmark"], [("model", "Модель"), ("micro_batch", "Micro batch"), ("accumulation", "Накопление"), ("tokens_per_second", "Токенов/с"), ("allocated_GiB", "Пик allocated, ГиБ"), ("timed_steps", "Шагов замера")]) + "\n\nMicro batch 32 показал максимальную скорость среди протестированных вариантов для всех трёх конфигураций. Замеры короткие, без повторов и доверительных интервалов: для 35M 2 шага разогрева + 5 замеров, для 97M/222M — 3 + 10. Пик allocated отличается от зарезервированной памяти и от веса checkpoint.")
    picture("04_batch_benchmark")
    md("## 4. Влияние Wikipedia\n\nОбе ветки начинаются с одного checkpoint исходной 97M. В каждой заново создан AdamW, совпадают effective batch, warmup и расписание LR на 5000 шагов. Контролируемое сравнение проводится на **последнем состоянии шага 1000**, при 131 072 000 дополнительных целевых токенов в каждой ветке. Это сравнение использует сохранённые validation-метрики; checkpoint шага 1000 смеси не требуется для чтения графика.\n\n" + table_md([r for r in tables["controlled_continuation"] if r["step"] == 1000], [("model", "Ветка"), ("split", "Validation"), ("loss", "Loss на шаге 1000")]) + "\n\nНа Wikipedia смесь дала loss **2.7416** против **3.2001** у новостного контроля. На ArmWeb смесь в этой точке немного хуже: **2.5905** против **2.5792**. После 5000 шагов смеси loss составил **2.5650 / 2.6329** на ArmWeb / Wikipedia. Сравнения с новостной веткой на 5000 шагах нет. Это один запуск на seed 42; устойчивость эффекта на других seed не измерялась.")
    picture("02_data_mixture")
    has_reported_tests = bool(inference and inference.get("reported_test_rows"))
    if final_evaluation is not None or has_reported_tests:
        wiki = {row["model"]: row for row in tables["heldout"] if row["corpus"] == "wikipedia"}
        before, after = wiki["arm_gpt_97m"]["perplexity"], wiki["arm_gpt_97m_cpt_mix"]["perplexity"]
        test_note = (f"На отложенной Wikipedia perplexity снизилась с **{before:.2f} до {after:.2f}** "
                     f"({100 * (1 - after / before):.1f}% относительно исходной модели). "
                     "На обоих тестах ArmWeb есть небольшое улучшение. Checkpoint выбран по validation до этой оценки. "
                     "Обе 97M оценены на одинаковых test split одним скриптом. Это один запуск; вывод относится к данным корпусам, "
                     "а не к достоверности свободных фактических ответов.")
        if final_evaluation is None:
            test_note += ("\n\n**Точность записи:** шесть строк финального test прочитаны из сохранённой таблицы outputs "
                          "в `inference.ipynb`: loss округлён до 4 знаков, perplexity до 2. "
                          "Для ArmWeb исходных 35M/97M сохранены более точные значения из исходного архива, "
                          "совпадающие с округлённой таблицей. Прочерки в числе целевых токенов означают, "
                          "что это число не выведено в таблице ноутбука; оно не восстановлено предположением. "
                          "Полный JSON финальной оценки и полная кривая 97M не прочитаны, поэтому дополнительные знаки и точки не добавлены.")
    else:
        test_note = "**Финальный JSON тестовой оценки ещё не добавлен.** Validation финальной смеси не подставляется вместо test."
    md("## 5. Отложенные тесты и фактические вопросы\n\nLoss усреднён по всем целевым токенам соответствующего split, с последним коротким остатком; perplexity = exp(loss). Ниже именно test, а не validation.\n\n" + table_md(tables["heldout"], [("model", "Модель"), ("corpus", "Корпус"), ("split", "Test split"), ("scored_tokens", "Целевых токенов"), ("loss", "Loss"), ("perplexity", "Perplexity")]) + "\n\n" + test_note + "\n\n### Небольшая QA-диагностика\n\n8 фактов × 3 формулировки = 24 связанных вопроса. В MC четыре кандидата; 25% — вероятность случайного выбора среди них. Этот уровень не относится к свободным ответам. Считаются сумма и среднее log-probability токенов ответа; sum может предпочитать более короткие варианты. Свободный ответ оценивается exact match с нормализацией и заданными вариантами, поэтому правильные неучтённые формулировки могут быть пропущены. Два few-shot примера — часть фиксированного промпта, а не SFT.\n\n" + table_md(tables["qa"], [("model", "Модель"), ("shots", "Shots"), ("mc_sum", "MC sum /24"), ("mc_mean", "MC mean /24"), ("free_exact", "Свободный /24"), ("all_three_mc_sum", "Все 3 верны MC /8"), ("all_three_free", "Все 3 верны свободно /8")]) + "\n\nВ zero-shot после смеси MC sum вырос с 14 до 17, свободные ответы — с 3 до 6. Во всех вариантах **0/8 фактов отвечены свободно верно при всех трёх формулировках**. В two-shot улучшения свободных ответов нет. Факты могли встречаться в претрейне; этот многократно просмотренный маленький набор — диагностика, не независимый бенчмарк общего владения языком и не доказательство статистически значимого улучшения.")
    picture("05_qa_zero_shot")
    md("## 6. Эксперимент с SFT\n\nSFT выполнялся на исходной 97M, **до смеси с Wikipedia**. Train — 3136, validation — 391, test — 391 пары Aya/Dolly. Loss считался только по ответу и EOS; prompt и padding маскировались. Лучший checkpoint выбран на шаге 98, а не на последнем шаге 147.\n\n" + table_md(tables["sft_heldout"], [("model", "Модель"), ("step", "Шаг checkpoint"), ("examples", "Test пар"), ("answer_loss", "Answer loss"), ("answer_perplexity", "Answer perplexity")]) + "\n\nTeacher-forced test loss снизился с **3.4145 до 3.0038**, но свободные ответы в QA-диагностике ухудшились с **3/24 до 0/24**. Эти метрики оценивают разные способности. В данных замечены ошибки перевода и ответов; это возможный фактор, но его причинное влияние отдельным чистым экспериментом не проверялось. Эта SFT-модель не выбрана основной для демонстрации.")
    picture("06_sft")
    inference_note = "Выполненный ноутбук инференса ещё не добавлен."
    if inference is not None:
        inference_note = (f"В `inference.ipynb` сохранены выводы серверного запуска: выполнено ячеек кода — {inference['code_cells']}, "
                          f"ошибок в outputs — {inference['errors']}, сохранено примеров — {inference['samples_saved']}. "
                          f"Свободный ответ: **{inference['free_exact_correct']}/24**, все три формулировки верны: "
                          f"**{inference['all_paraphrases_correct']}/8**. Это повтор того же набора, а не новый независимый тест.\n\n"
                          "Восемь продолжений показывают новостной и энциклопедический стиль, но также повторы и смысловые противоречия. "
                          "Например, в спортивном продолжении Уэльс побеждает Уэльс; в ответах модель может назвать верный город, "
                          "а затем повторять вопрос и служебные заголовки. Все примеры оставлены в ноутбуке, без отбора только удачных. "
                          "Автоматическая оценка грамматики армянского языка не проводилась.")
    md("## 7. Инференс и комплект сдачи\n\n" + inference_note + "\n\nКод data/tokenizer/train, зависимости и инструкции воспроизведения находятся в корне репозитория; токенизатор — в tokenizer/. "
       "Для завершения комплекта сдачи остаётся опубликовать checkpoint модели. В отчётах весов нет: manifest исходного экспорта подтверждает только наличие/размеры файлов на сервере. "
       + ("Итоговая оценка test добавлена из `final_evaluation.json`. " if final_evaluation is not None
          else "Итоговая оценка test добавлена из таблицы выполненного `inference.ipynb`, с явным указанием округления. " if has_reported_tests
          else "Для добавления итоговой оценки нужен серверный `final_97m_comparison/results.json`. ")
       + "\n\nВ конфигурациях обучения указан PyTorch **2.9.1+cu128**, tokenizers **0.22.1**, NumPy **2.4.2**, PyArrow **20.0.0**. "
       "В manifest экспорта часть пакетов помечена null — это означает, что их версии не найдены в окружении экспорта. "
       "Выполненный notebook подтверждает работу инференса в серверном kernel на момент сохранения.\n\n"
       "Полные числовые таблицы, источники и SHA256 прочитанных файлов находятся в `results.json`. Источники — сохранённые файлы и outputs; код из прикреплённых ноутбуков при сборке отчёта не выполняется.")
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    nbformat.write(nb, output / "results.ipynb")


def make_html(tables, names, output):
    """Простая локальная версия таблиц и рисунков для просмотра без Jupyter."""
    sections = []
    for key, rows in tables.items():
        if not rows: continue
        columns = list(rows[0])
        headers = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>" for row in rows)
        sections.append(f"<h2>{html.escape(key)}</h2><div class='table'><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>")
    for name in names:
        data = base64.b64encode((output / "figures" / (name + ".png")).read_bytes()).decode()
        sections.append(f"<figure><img src='data:image/png;base64,{data}' alt='{html.escape(name)}'></figure>")
    partial = any(row["last_curve_step"] < row["last_step"] for row in tables["runs"])
    curve_note = "Кривые с пропусками показаны без интерполяции. " if partial else "Кривые обучения восстановлены из сохранённых CSV. "
    text = """<!doctype html><html lang='ru'><meta charset='utf-8'><title>Armenian LLM — результаты</title><style>
body{font:15px system-ui,sans-serif;line-height:1.5;max-width:1250px;margin:40px auto;padding:0 24px;color:#17212c;background:#fafbfc}
h1,h2{font-weight:650}.table{overflow:auto;border:1px solid #dae0e5;border-radius:8px;background:white}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{padding:9px 12px;border-bottom:1px solid #e8ecf0;text-align:left}th{background:#edf2f7}img{width:100%;height:auto}figure{margin:32px 0}p{max-width:850px}
</style><h1>Армянская LLM: таблицы и графики</h1><p>Пояснения и ограничения приведены в results.ipynb. Здесь только числовая сводка. 222M — пилот; сравнение 35M/97M использует разные бюджеты. """ + curve_note + "Validation не заменяет test; метрики QA получены на 8 фактах, а не 24 независимых фактах.</p>" + "".join(sections) + "</html>"
    (output / "index.html").write_text(text, encoding="utf-8")


def render_saved_summary(summary_path, output_dir):
    """Пересобрать отчёт из опубликованных таблиц и доступных точек кривых."""
    document = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    output = Path(output_dir).resolve()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    names = figures(document["tables"], document["curves"], output / "figures")
    make_notebook(document["tables"], names, output, document.get("read_issues", []),
                  document.get("final_evaluation"), document.get("inference_execution"))
    make_html(document["tables"], names, output)
    (output / "results.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Отчёт пересобран из сохранённых данных:", output / "results.ipynb")


def main(reports_dir, output_dir, final_evaluation_path=None, inference_path=None):
    reports = Reports(reports_dir)
    output = Path(output_dir).resolve()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    final_evaluation = None
    if final_evaluation_path is not None:
        try:
            final_evaluation = reports.external_json(final_evaluation_path)
        except OSError as error:
            reports.issues.append({"kind": "final_evaluation_unavailable", "path": str(final_evaluation_path),
                                   "error": type(error).__name__, "fallback": "executed notebook outputs, when present"})
    inference = inspect_inference(reports, inference_path) if inference_path is not None else None
    reported_test_rows = inference.get("reported_test_rows", []) if inference else []
    tables, metrics, configs = make_tables(reports, final_evaluation, reported_test_rows)
    names = figures(tables, metrics, output / "figures")
    limitations = ["Different pretrain token budgets", "222M is a pilot", "QA: eight related fact groups", "Cross-corpus near-duplicates not audited"]
    if final_evaluation is None:
        limitations.append("Final test read from rounded notebook table; raw JSON unavailable" if reported_test_rows
                           else "Final CPT test export not imported")
    if any(row["curve_format"] == "notebook_console_rounded" for row in tables["runs"]):
        limitations.append("Notebook fallback for inaccessible CSV logs")
    document = {"tables": tables, "configs": configs, "curves": metrics, "source_root": str(reports.root),
                "sources": reports.sources, "read_issues": reports.issues,
                "final_evaluation": final_evaluation, "inference_execution": inference, "limitations": limitations}
    (output / "results.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    if final_evaluation is not None:
        if Path(final_evaluation_path).resolve() != output / "final_evaluation.json":
            shutil.copyfile(final_evaluation_path, output / "final_evaluation.json")
    if inference_path is not None:
        if Path(inference_path).resolve() != output / "inference.ipynb":
            shutil.copyfile(inference_path, output / "inference.ipynb")
        for name in ("model.py", "generate.py", "benchmark_facts.json"):
            source = Path(__file__).resolve().parent / name
            if source.resolve() != output / name:
                shutil.copyfile(source, output / name)
    make_notebook(tables, names, output, reports.issues, final_evaluation, inference)
    make_html(tables, names, output)
    archive_path = output.parent / "submission_results.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file(): archive.write(path, arcname="submission_results/" + path.relative_to(output).as_posix())
        archive.write(Path(__file__), arcname="submission_results/build_final_results.py")
        for script_name in ("evaluate_final.py", "export_submission_model.py"):
            script = Path(__file__).with_name(script_name)
            if script.exists():
                archive.write(script, arcname="submission_results/" + script_name)
    print(json.dumps({"notebook": str(output / "results.ipynb"), "html": str(output / "index.html"),
                      "figures": len(names), "tables": len(tables), "archive": str(archive_path),
                      "final_test_source": "full_json" if final_evaluation is not None else "rounded_notebook_table" if reported_test_rows else None,
                      "inference": inference,
                      "read_issues": reports.issues}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path(__file__).resolve().parent / "final_reports_20261026_115500")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "submission_results")
    parser.add_argument("--final-evaluation", type=Path, help="Серверный final_97m_comparison/results.json")
    parser.add_argument("--from-summary", type=Path, help="Пересобрать графики и отчёт из опубликованного results.json")
    parser.add_argument("--inference-notebook", type=Path, default=Path(__file__).resolve().parent / "inference.ipynb")
    args = parser.parse_args()
    if args.from_summary is not None:
        render_saved_summary(args.from_summary, args.output_dir)
    else:
        final_path = args.final_evaluation
        if final_path is None:
            final_path = next((p for p in (Path(__file__).resolve().parent / "final_evaluation.json",
                                          Path(__file__).resolve().parent / "results.json") if p.exists()), None)
        main(args.reports_dir, args.output_dir, final_path, args.inference_notebook)
