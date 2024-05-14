from helix.main import stub
from helix.esm import image
import modal
import numpy as np
import pandas as pd
import torch
import random

np.random.seed(1)
random.seed(1)

AAs = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
]


def deep_mutational_scan(sequence, exclude_noop=True):
    for pos, wt in enumerate(sequence):
        for mt in AAs:
            if exclude_noop and wt == mt:
                continue
            yield (pos, wt, mt)


def label_row(row, sequence, token_probs, tokenizer, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    wt_encoded, mt_encoded = tokenizer.convert_tokens_to_ids(
        [wt]), tokenizer.convert_tokens_to_ids([mt])

    # add 1 for BOS
    score = token_probs[0, 1 + idx, mt_encoded[0]] - \
        token_probs[0, 1 + idx, wt_encoded[0]]
    return score.item()


def mask_sequence(sequence, tokenized_sequence, tokenizer):
    for i in range(len(sequence)):
        tokens_masked = tokenized_sequence.clone()
        tokens_masked[0, i] = tokenizer.mask_token_id
        yield tokens_masked


def compute_pppl(row, sequence, model, tokenizer, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1):]

    batch_size = 64
    token_probs_list = []
    for batch_idx, tokens_masked_batch in enumerate(create_batches(mask_sequence(sequence, tokenizer), batch_size)):
        with torch.inference_mode():
            logits = model(tokens_masked_batch).logits
            token_probs = torch.log_softmax(logits, dim=-1)
        for j in range(len(tokens_masked_batch)):
            token_idx = batch_idx*batch_size+j
            token_probs_list.append(
                token_probs[token_idx, batch_idx, tokenizer.convert_tokens_to_ids([sequence[batch_idx]])[0]])

    return token_probs_list.sum().item()


def compute_masked_log_probs(tokenized_sequence, model, tokenizer, batch_size=64):

    all_token_probs = []  # tokenized sequence length
    for batch_idx, tokens_masked_batch in enumerate(create_batches(mask_sequence(tokenized_sequence, tokenizer), batch_size)):
        with torch.inference_mode():
            logits = model(tokens_masked_batch.cuda()).logits
            token_probs = torch.log_softmax(logits, dim=-1)
        for j in range(len(tokens_masked_batch)):
            all_token_probs.append(token_probs[:, batch_idx])  # vocab size

    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)


def create_batches(tensor, batch_size):
    for i in range(0, tensor.size(0), batch_size):
        yield tensor[i:i + batch_size]


def compute_log_probs(tokenized_sequence, model, batch_size=64):
    token_probs_list = []
    for batch in create_batches(tokenized_sequence, batch_size):
        with torch.inference_mode():
            logits = model(batch.cuda()).logits
            token_probs = torch.log_softmax(logits, dim=-1)
        token_probs_list.append(token_probs)

    token_probs = torch.cat(token_probs_list, dim=0)
    return token_probs


@stub.function(gpu=modal.gpu.A100(size="80GB"), image=image, timeout=4000)
def dms(sequence):
    import transformers
    offset_idx = 1
    mutation_col = "mutant"
    data = [
        f'{wt}{pos + offset_idx}{mt}'
        for pos, wt, mt in deep_mutational_scan(sequence)
    ]
    df = pd.DataFrame(data, columns=[mutation_col])
    [sequence[:pos] + mt + sequence[pos+1:]
     for pos, wt, mt in deep_mutational_scan(sequence)]

    model_name = "facebook/esm2_t36_3B_UR50D"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name)
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()
    tokenized_sequence = tokenizer(
        sequence, return_tensors="pt", add_special_tokens=True)['input_ids']
    token_probs = compute_log_probs(tokenized_sequence, model, batch_size=512)
    df['esm'] = df.apply(
        lambda row: label_row(
            row[mutation_col],
            sequence,
            token_probs,
            tokenizer,
            offset_idx,
        ),
        axis=1,
    )
    return df
