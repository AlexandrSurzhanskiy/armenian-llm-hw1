"""Генерация продолжения текста нашей GPT без чат-шаблона."""

import torch


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=96, temperature=0.8, top_k=50, seed=2026):
    if temperature <= 0:
        raise ValueError("temperature должна быть положительной")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    prompt_ids = [bos_id] + tokenizer.encode(prompt, add_special_tokens=False).ids
    sequence = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids = []
    try:
        for _ in range(max_new_tokens):
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits, _ = model(sequence[:, -model.config.context_length:])
            scores = logits[0, -1].float() / temperature
            scores[pad_id] = -torch.inf
            scores[bos_id] = -torch.inf
            if top_k > 0:
                threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                scores[scores < threshold] = -torch.inf
            next_id = torch.multinomial(scores.softmax(dim=-1), 1, generator=generator)
            token_id = next_id.item()
            generated_ids.append(token_id)
            if token_id == eos_id:
                break
            sequence = torch.cat((sequence, next_id.reshape(1, 1)), dim=1)
            del logits
    finally:
        model.train(was_training)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
