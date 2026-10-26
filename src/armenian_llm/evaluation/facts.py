"""24 вопроса на армянском: ранжирование ответов и greedy-генерация.

Запуск: %run -m armenian_llm.evaluation.facts
Для сравнения до/после SFT: --models arm_gpt_97m arm_gpt_97m_sft --zero-shot
Данные: benchmark_facts.json рядом со скриптом. По умолчанию два примера
задают формат QA; --zero-shot исключает их. Веса моделей не обновляются.
"""

import argparse
import gc
import json
from pathlib import Path
import re
import unicodedata

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tokenizers import Tokenizer

from armenian_llm.inference.generate import generate_text
from armenian_llm.modeling.gpt import GPT, GPTConfig
from armenian_llm.paths import ARTIFACTS_DIR, CHECKPOINTS_DIR, FACTS_CONFIG
from armenian_llm.training.utils import file_sha256


def normalize_answer(text):
    text = text.strip().split("\n", 1)[0].split("։", 1)[0]
    text = unicodedata.normalize("NFC", text).casefold().replace("եւ", "և")
    return re.sub(r"\s+", " ", text).strip(" \t\r.,;:!?՝՛՜՞«»\"'—–-")


def make_prompt(examples, question):
    prefix = "".join(
        f"Հարց՝ {example['question']}\nՊատասխան՝ {example['answer']}։\n\n"
        for example in examples
    )
    return prefix + f"Հարց՝ {question}\nՊատասխան՝"


@torch.inference_mode()
def score_answer(model, tokenizer, prompt, answer, device):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    full_ids = tokenizer.encode(prompt + " " + answer, add_special_tokens=False).ids
    # Считаем P(answer | prompt). Токены вопроса в оценку не входят.
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Токенизация пересекла границу prompt/answer")
    answer_length = len(full_ids) - len(prompt_ids)
    if answer_length < 1 or len(full_ids) > model.config.context_length:
        raise ValueError("Неверная длина примера для контекста модели")
    ids = torch.tensor(
        [[tokenizer.token_to_id("<bos>")] + full_ids], dtype=torch.long, device=device
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(ids[:, :-1])
    labels = ids[:, 1:]
    losses = F.cross_entropy(
        logits[0, -answer_length:].float(), labels[0, -answer_length:], reduction="none"
    )
    log_probability = -losses.sum().item()
    return {
        "answer": answer,
        "tokens": answer_length,
        "sum_log_probability": log_probability,
        "mean_log_probability": log_probability / answer_length,
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["fact_id"], []).append(row)
    metrics = {"questions": len(rows), "facts": len(groups)}
    for mode in ("sum", "mean"):
        correct_key = f"choice_correct_{mode}"
        prediction_key = f"choice_prediction_{mode}"
        correct = sum(row[correct_key] for row in rows)
        metrics[f"choice_{mode}"] = {
            "correct": correct,
            "accuracy": correct / len(rows),
            "all_paraphrases_correct": sum(all(row[correct_key] for row in group) for group in groups.values()),
            "consistent_facts": sum(len({row[prediction_key] for row in group}) == 1 for group in groups.values()),
        }
    correct = sum(row["free_exact_match"] for row in rows)
    metrics["free_exact"] = {
        "correct": correct,
        "accuracy": correct / len(rows),
        "all_paraphrases_correct": sum(all(row["free_exact_match"] for row in group) for group in groups.values()),
    }
    return metrics


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(dataset_path=None, models=None, zero_shot=False, output_name=None):
    dataset_path = Path(dataset_path) if dataset_path is not None else FACTS_CONFIG
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    models = list(models) if models is not None else ["arm_gpt_35m", "arm_gpt_97m"]
    if not models or len(models) != len(set(models)):
        raise ValueError("Нужен непустой список разных моделей")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in models):
        raise ValueError("Модели задаются именами каталогов внутри checkpoints")
    if zero_shot:
        dataset["few_shot_examples"] = []
        dataset["protocol"]["shots"] = 0
        if not dataset["name"].endswith("_zero_shot"):
            dataset["name"] += "_zero_shot"
        dataset["description_ru"] = "Восемь фактов, по три формулировки; без few-shot примеров."
    if dataset["protocol"]["shots"] != len(dataset["few_shot_examples"]):
        raise ValueError("Число few-shot примеров не совпало с протоколом")
    output_name = output_name or dataset["name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output_name):
        raise ValueError("output_name должен быть именем каталога без слешей")
    device = torch.device("cuda")
    output_dir = ARTIFACTS_DIR / "evaluation" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # Сохраняем фактические вопросы и примеры с учётом --zero-shot.
    save_json(output_dir / "dataset.json", dataset)
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.jsonl"
    total_questions = sum(len(fact["questions"]) for fact in dataset["facts"])
    chance = sum(
        len(fact["questions"]) / len(dataset["candidate_pools"][fact["category"]])
        for fact in dataset["facts"]
    ) / total_questions
    summary = {
        "status": "running", "dataset": dataset["name"], "run_name": output_name,
        "dataset_sha256": file_sha256(output_dir / "dataset.json"),
        "source_dataset_sha256": file_sha256(dataset_path), "models": models,
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device),
        "precision": "bf16_autocast_fp32_parameters",
        "protocol": dataset["protocol"], "chance_choice_accuracy": chance,
        "limitations_ru": dataset["limitations_ru"],
        "checkpoints": {}, "metrics": {},
    }
    save_json(summary_path, summary)
    print(f"Фактов: {len(dataset['facts'])}; вопросов: {total_questions}; few-shot примеров: {len(dataset['few_shot_examples'])}", flush=True)
    print(f"Случайный выбор: {chance:.0%}", flush=True)
    all_rows = []
    with predictions_path.open("w", encoding="utf-8") as output:
        for name in models:
            print(f"\nОцениваем {name}...", flush=True)
            run_dir = CHECKPOINTS_DIR / name
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True, mmap=True)
            tokenizer_path = run_dir / "tokenizer" / "tokenizer.json"
            if file_sha256(tokenizer_path) != checkpoint["fingerprints"]["tokenizer_sha256"]:
                raise ValueError(f"{name}: токенизатор не совпал с обучающим")
            model = GPT(GPTConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model"])
            model = model.to(device).eval()
            summary["checkpoints"][name] = {
                "path": str(run_dir / "best.pt"), "step": checkpoint["step"],
                "training_tokens_seen": checkpoint["tokens_seen"],
                "stage": checkpoint.get("stage", "pretrain"),
                "tokens_seen_definition": "SFT answer and EOS tokens" if checkpoint.get("stage") == "sft" else "pretrain next-token targets",
                "provenance": checkpoint.get("provenance"),
                "parameters": model.num_parameters(), "fingerprints": checkpoint["fingerprints"],
                "model_config": checkpoint["model_config"], "train_config": checkpoint["train_config"],
            }
            del checkpoint
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.no_padding()
            tokenizer.no_truncation()
            model_rows = []
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for fact in dataset["facts"]:
                    pool = dataset["candidate_pools"][fact["category"]]
                    gold = next(choice for choice in pool if choice["text"] == fact["answer"])
                    accepted = {normalize_answer(alias) for alias in gold["aliases"]}
                    for variant, question in enumerate(fact["questions"], start=1):
                        prompt = make_prompt(dataset["few_shot_examples"], question["text"])
                        scores = [score_answer(model, tokenizer, prompt, choice["text"], device) for choice in pool]
                        prediction_sum = max(scores, key=lambda score: score["sum_log_probability"])["answer"]
                        prediction_mean = max(scores, key=lambda score: score["mean_log_probability"])["answer"]
                        completion = generate_text(model, tokenizer, prompt, device, **dataset["protocol"]["free_generation"])
                        extracted = normalize_answer(completion)
                        row = {
                            "model": name, "fact_id": fact["id"], "category": fact["category"],
                            "variant": variant, "question": question["text"],
                            "translation_ru": question["translation_ru"], "prompt": prompt,
                            "gold": fact["answer"], "candidate_scores": scores,
                            "choice_prediction_sum": prediction_sum, "choice_correct_sum": prediction_sum == fact["answer"],
                            "choice_prediction_mean": prediction_mean, "choice_correct_mean": prediction_mean == fact["answer"],
                            "completion": completion, "extracted_answer": extracted,
                            "free_exact_match": extracted in accepted,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output.flush()
                        model_rows.append(row)
            all_rows.extend(model_rows)
            summary["metrics"][name] = summarize(model_rows)
            save_json(summary_path, summary)
            print(f"Сохранено ответов: {len(model_rows)}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    summary["status"] = "complete"
    save_json(summary_path, summary)
    width = max(14, max(map(len, models)))
    n_facts = len(dataset["facts"])
    print(f"\n{'Модель':<{width}} | MC sum /{total_questions} | MC mean /{total_questions} | Свободный /{total_questions} | Все 3 верны MC /{n_facts} | Одинаковый MC /{n_facts}")
    for name, metrics in summary["metrics"].items():
        print(
            f"{name:<{width}} | {metrics['choice_sum']['correct']:>10} | {metrics['choice_mean']['correct']:>11} |"
            f" {metrics['free_exact']['correct']:>13} | {metrics['choice_sum']['all_paraphrases_correct']:>17} |"
            f" {metrics['choice_sum']['consistent_facts']:>16}"
        )
        print(f"  Свободный ответ верен при всех 3 формулировках: {metrics['free_exact']['all_paraphrases_correct']}/{metrics['facts']}")

    print("\nПримеры: три формулировки первого факта, все модели")
    for variant, question in enumerate(dataset["facts"][0]["questions"], start=1):
        print(f"\n{question['text']} ({question['translation_ru']})")
        print(f"Эталон: {dataset['facts'][0]['answer']}")
        for row in all_rows:
            if row["fact_id"] == dataset["facts"][0]["id"] and row["variant"] == variant:
                print(f"{row['model']}: MC sum — {row['choice_prediction_sum']}; генерация: {row['completion']}")
    print(f"\nМетрики: {summary_path}")
    print(f"Все ответы: {predictions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--models", nargs="+", help="Имена каталогов моделей внутри checkpoints")
    parser.add_argument("--zero-shot", action="store_true")
    parser.add_argument("--output-name", help="Имя каталога результатов внутри artifacts/evaluation")
    args = parser.parse_args()
    main(dataset_path=args.dataset, models=args.models, zero_shot=args.zero_shot, output_name=args.output_name)
