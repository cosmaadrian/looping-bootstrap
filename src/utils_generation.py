import torch
import torch.nn.functional as F


def fast_categorical_sample(logits, temperature = 1.0):
    noise = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
    return (logits / temperature + noise).argmax(dim = -1)


def top_p_filtering(logits, top_p = 0.9, min_tokens_to_keep = 1):
    """
    Nucleus (top-p) filtering.
    logits: (B, V)
    Returns logits with tokens outside the nucleus set to -inf.
    """
    if top_p is None or top_p >= 1.0:
        return logits

    top_p = float(top_p)
    top_p = max(min(top_p, 1.0), 0.0)

    sorted_logits, sorted_indices = torch.sort(logits, descending = True, dim = -1)  # (B,V)
    sorted_probs = F.softmax(sorted_logits, dim = -1)
    cumulative_probs = sorted_probs.cumsum(dim = -1)

    # Remove tokens with cumulative prob above threshold (keep at least one token)
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift right to also keep the first token that exceeds top_p (HF-style)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    indices_to_remove = torch.zeros_like(logits, dtype = torch.bool).scatter(dim = 1, index = sorted_indices, src = sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, -torch.inf)


def batch_decode(tokenizer, input_ids):
    return [tokenizer.decode(ids.long().tolist(), skip_special_tokens = True) for ids in input_ids]


def generate(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    max_new_tokens = 32,
    temperature = 1.0,
    top_p = 0.95,
    stop_on_eos = True,
):
    pad_id = 2

    B = input_ids.shape[0]
    device = input_ids.device

    answer = torch.full((B, max_new_tokens), pad_id, dtype = torch.long, device = device)
    attention_mask_answer = torch.zeros((B, max_new_tokens), dtype = torch.long, device = device)

    input_ids = F.pad(input_ids.detach().clone(), (0, max_new_tokens), 'constant', pad_id)
    attention_mask = F.pad(attention_mask.detach().clone(), (0, max_new_tokens), 'constant', 0)

    is_eos = torch.zeros((B, 1), dtype = torch.bool, device = device)

    for i in range(max_new_tokens):
        idxs = attention_mask.sum(dim = 1) - 1
        batch_indices = torch.arange(B, device = device)

        logits = model({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        })  # (B, T, V)

        last_logits = logits[batch_indices, idxs, :]  # (B, V)

        if temperature == 0:
            dist_logits = last_logits
            token_next = dist_logits.argmax(dim = -1)
        else:
            dist_logits = last_logits / temperature
            dist_logits = top_p_filtering(dist_logits, top_p = top_p, min_tokens_to_keep = 1)
            token_next = fast_categorical_sample(dist_logits, temperature = 1.0)

        token_idx = token_next.unsqueeze(-1)  # (B, 1)

        positions = idxs + 1
        input_ids[batch_indices, positions] = token_next
        answer[:, i] = token_next

        cond = (token_next == 2) | (token_next == 3)
        new_token_mask = torch.where(cond, torch.tensor(0, device = device), torch.tensor(1, device = device))

        attention_mask_answer[:, i] = new_token_mask
        attention_mask[batch_indices, positions] = new_token_mask

        is_eos = is_eos | (token_idx == 3) | (token_idx == 2)
        if stop_on_eos and is_eos.all():
            break

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'answer_input_ids': answer,
        'answer_attention_mask': attention_mask_answer,
        'answer_str': batch_decode(tokenizer, answer),
    }
