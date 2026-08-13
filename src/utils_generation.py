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


def _generation_budgets(max_new_tokens, batch_size, device):
    if torch.is_tensor(max_new_tokens):
        budgets = max_new_tokens.to(device = device, dtype = torch.long).flatten()
    elif isinstance(max_new_tokens, (list, tuple)):
        budgets = torch.tensor(max_new_tokens, device = device, dtype = torch.long)
    else:
        budgets = torch.full((batch_size, ), int(max_new_tokens), device = device, dtype = torch.long)

    if budgets.numel() != batch_size:
        raise ValueError(f'Expected {batch_size} generation budgets, received {budgets.numel()}.')
    if torch.any(budgets < 0):
        raise ValueError('max_new_tokens must be non-negative.')
    return budgets


def generate(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    max_new_tokens = 32,
    temperature = 1.0,
    top_p = 0.95,
    stop_on_eos = True,
    model_kwargs = None,
    forward_kwargs = None,
):
    B = input_ids.shape[0]
    device = input_ids.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError('Generation requires a tokenizer pad or EOS token.')
    pad_id = int(pad_id)

    budgets = _generation_budgets(max_new_tokens, B, device)
    max_steps = int(budgets.max().item()) if B else 0
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    forward_kwargs = {} if forward_kwargs is None else dict(forward_kwargs)

    answer = torch.full((B, max_steps), pad_id, dtype = torch.long, device = device)
    attention_mask_answer = torch.zeros((B, max_steps), dtype = torch.long, device = device)

    input_ids = F.pad(input_ids.detach().clone(), (0, max_steps), 'constant', pad_id)
    attention_mask = F.pad(attention_mask.detach().clone(), (0, max_steps), 'constant', 0)

    if B:
        positions = torch.arange(attention_mask.shape[1], device = device).unsqueeze(0).expand(B, -1)
        last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim = 1).values
        if torch.any(last_positions < 0):
            raise ValueError('Every generation prompt must contain at least one unmasked token.')
        next_positions = last_positions + 1
    else:
        next_positions = torch.empty(0, dtype = torch.long, device = device)

    finished = torch.zeros(B, dtype = torch.bool, device = device)
    stop_token_ids = {int(token_id) for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token_id is not None}

    for i in range(max_steps):
        active = budgets.gt(i)
        if stop_on_eos:
            active &= ~finished
        if not active.any():
            break

        active_indices = active.nonzero(as_tuple = False).flatten()
        active_last_positions = next_positions[active_indices] - 1
        active_width = int(next_positions[active_indices].max().item())

        model_batch = {
            'input_ids': input_ids[active_indices, :active_width],
            'attention_mask': attention_mask[active_indices, :active_width],
            **model_kwargs,
        }
        logits = model(model_batch, **forward_kwargs)
        local_indices = torch.arange(active_indices.numel(), device = device)
        last_logits = logits[local_indices, active_last_positions, :]

        if temperature == 0:
            token_next = last_logits.argmax(dim = -1)
        else:
            dist_logits = top_p_filtering(last_logits / temperature, top_p = top_p, min_tokens_to_keep = 1)
            token_next = fast_categorical_sample(dist_logits, temperature = 1.0)

        token_positions = next_positions[active_indices]
        input_ids[active_indices, token_positions] = token_next
        answer[active_indices, i] = token_next

        is_stop = torch.zeros_like(token_next, dtype = torch.bool)
        if stop_on_eos:
            for stop_token_id in stop_token_ids:
                is_stop |= token_next.eq(stop_token_id)

        continuing_indices = active_indices[~is_stop]
        continuing_positions = token_positions[~is_stop]
        attention_mask_answer[continuing_indices, i] = 1
        attention_mask[continuing_indices, continuing_positions] = 1
        next_positions[continuing_indices] += 1

        if stop_on_eos:
            finished[active_indices[is_stop]] = True

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'answer_input_ids': answer,
        'answer_attention_mask': attention_mask_answer,
        'answer_str': batch_decode(tokenizer, answer),
    }
