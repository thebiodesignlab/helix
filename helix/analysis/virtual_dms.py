from helix.core import images
from helix.analysis.sequence.scorer import score_mutations
from helix.core import app
import modal
import pandas as pd

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


@app.function(gpu=modal.gpu.A100(size="80GB"), image=images.base, timeout=4000)
def dms(sequence, metrics=["wildtype_marginal", "masked_marginal"]):
    offset_idx = 1
    mutation_col = "mutant"
    data = [
        f'{wt}{pos + offset_idx}{mt}'
        for pos, wt, mt in deep_mutational_scan(sequence)
    ]
    df = pd.DataFrame(data, columns=[mutation_col])
    [sequence[:pos] + mt + sequence[pos+1:]
     for pos, wt, mt in deep_mutational_scan(sequence)]
    model_names = ["facebook/esm1b_t33_650M_UR50S", "facebook/esm1v_t33_650M_UR90S_1", "facebook/esm1v_t33_650M_UR90S_2", "facebook/esm1v_t33_650M_UR90S_3",
                   "facebook/esm1v_t33_650M_UR90S_4", "facebook/esm1v_t33_650M_UR90S_5", "facebook/esm2_t36_3B_UR50D", "facebook/esm2_t48_15B_UR50D"]

    for metric in metrics:
        results = score_mutations.starmap(
            [(model_name, sequence, df[mutation_col], metric)
             for model_name in model_names]
        )
        for model_name, result in zip(model_names, results):
            df[model_name + "_" + metric] = result

    return df
