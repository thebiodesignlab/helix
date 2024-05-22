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


def mask_sequence(tokenized_sequence, tokenizer):
    for i in range(len(tokenized_sequence)):
        tokens_masked = tokenized_sequence.clone()
        tokens_masked[i] = tokenizer.mask_token_id
        yield tokens_masked


def compute_pppl(row, sequence, model, tokenizer, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1):]
    tokenized_sequence = tokenizer(
        sequence, return_tensors="pt", add_special_tokens=True)['input_ids']

    batch_size = 64
    token_probs_list = []
    for batch_idx, tokens_masked_batch in enumerate(create_batches(tokenized_sequence, batch_size)):
        with torch.inference_mode():
            logits = model(tokens_masked_batch).logits
            token_probs = torch.log_softmax(logits, dim=-1)
        for j in range(len(tokens_masked_batch)):
            token_idx = batch_idx*batch_size+j
            token_probs_list.append(
                token_probs[token_idx, batch_idx, tokenizer.convert_tokens_to_ids([sequence[batch_idx]])[0]])

    return token_probs_list.sum().item()


def compute_masked_log_probs(batch_tokenized_sequences, model, tokenizer):
    all_token_probs = []  # tokenized sequence length
    for i in range(batch_tokenized_sequences.size(1)):
        with torch.inference_mode():
            # batch size, seq len, vocab size
            tokens_masked_batch = batch_tokenized_sequences.clone()
            tokens_masked_batch[0, i] = tokenizer.mask_token_id
            logits = model(tokens_masked_batch.cuda()).logits
            token_probs = torch.log_softmax(logits, dim=-1)
            all_token_probs.append(token_probs)
    token_probs = torch.cat(all_token_probs, dim=0)
    return token_probs


def create_batches(tensor, batch_size):
    return [tensor[i:i + batch_size] for i in range(0, len(tensor), batch_size)]


def compute_log_probs(batch_tokenized_sequences, model):
    token_probs_list = []
    with torch.inference_mode():
        logits = model(batch_tokenized_sequences.cuda()).logits
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

    for model_name in ["facebook/esm1b_t33_650M_UR50S", "facebook/esm1v_t33_650M_UR90S_1", "facebook/esm1v_t33_650M_UR90S_2", "facebook/esm1v_t33_650M_UR90S_3", "facebook/esm1v_t33_650M_UR90S_4", "facebook/esm1v_t33_650M_UR90S_5", "facebook/esm2_t36_3B_UR50D", "facebook/esm2_t48_15B_UR50D"]:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name)
        model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = model.cuda()
        batch_tokenized_sequences = tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=True)['input_ids']
        masked_log_probs = compute_masked_log_probs(
            batch_tokenized_sequences, model, tokenizer)
        token_log_probs = compute_log_probs(batch_tokenized_sequences, model)
        df[model_name+"_wt_marginal"] = df.apply(
            lambda row: label_row(
                row[mutation_col],
                sequence,
                token_log_probs,
                tokenizer,
                offset_idx,
            ),
            axis=1,
        )
        df[model_name+"_masked_marginal"] = df.apply(
            lambda row: label_row(
                row[mutation_col],
                sequence,
                masked_log_probs,
                tokenizer,
                offset_idx,
            ),
            axis=1,
        )
    return df
