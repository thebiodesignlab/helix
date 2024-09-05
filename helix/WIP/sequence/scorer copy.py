from typing import Union
from helix.core import volumes, images
from helix.core import app
from modal import enter, method, gpu

MODEL_DIR = "/mnt/models"

foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"


@app.cls(gpu=gpu.H100(), volumes={MODEL_DIR: volumes.model_weights}, image=images.base, timeout=4000)
class MLMScorer:
    def __init__(self, model_name):
        self.model_name = model_name

    @enter()
    def setup(self):
        import transformers
        import torch
        volumes.model_weights.reload()  # Fetch latest changes to the volume
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(
            self.model_name, cache_dir=MODEL_DIR)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=MODEL_DIR)
        volumes.model_weights.commit()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode_sequences(self, sequences: Union[str, list[str]]):
        """
        Encode the given sequences using the tokenizer. Supports both single and batch sequences.

        Args:
            sequences (Union[str, List[str]]): The sequence or list of sequences to encode.

        Returns:
            torch.Tensor: The encoded sequences tensor.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        return self.tokenizer(sequences, return_tensors="pt", add_special_tokens=True)['input_ids'].to(self.model.device)

    @method()
    def compute_token_log_probs(self, sequences: Union[str, list[str]]):
        """
        Compute the log probabilities of the given sequences using the model.

        Args:
            sequences (Union[str, List[str]]): The sequence or list of sequences to compute log probabilities for.

        Returns:
            torch.Tensor: The log probabilities tensor.
        """
        import torch
        token_probs_list = []
        batch_tokenized_sequences = self.encode_sequences(sequences)

        with torch.inference_mode():
            logits = self.model(batch_tokenized_sequences).logits
            token_probs = torch.log_softmax(logits, dim=-1)

        token_probs_list.append(token_probs)
        token_probs = torch.cat(token_probs_list, dim=0)

        return token_probs

    @method()
    def get_vocab(self):
        return self.tokenizer.get_vocab()

    @method()
    def compute_masked_token_log_probs(self, sequences: Union[str, list[str]]):
        import torch
        all_token_probs = []
        batch_tokenized_sequences = self.encode_sequences(sequences)
        for i in range(batch_tokenized_sequences.size(1)):
            with torch.inference_mode():
                # batch size, seq len, vocab size
                batch_masked_tokens = batch_tokenized_sequences.clone()
                batch_masked_tokens[0, i] = self.tokenizer.mask_token_id
                logits = self.model(batch_masked_tokens).logits
                token_probs = torch.log_softmax(logits, dim=-1)
                all_token_probs.append(token_probs)
        token_probs = torch.cat(all_token_probs, dim=0)
        return token_probs

    @method()
    def deep_mutational_scan(self, sequence, metric):
        if metric == 'wildtype_marginal':
            log_probs = self.compute_token_log_probs.remote(sequence)

        elif metric == 'masked_marginal':
            log_probs = self.compute_masked_token_log_probs.remote(sequence)
        tokenized_sequence = self.tokenizer.tokenize(sequence)
        mutation_scores = {}

        # Iterate over the length of the tokenized sequence
        for idx in range(len(tokenized_sequence)):
            wt_token = tokenized_sequence[idx][0]
            self.tokenizer.convert_tokens_to_ids(wt_token)
            wt_token_vocab_start = self.tokenizer.get_vocab(
            )[wt_token + foldseek_struc_vocab[0]]

            for mt_token_id in range(log_probs.size(2)):
                # Check that mt_token_id doesn't correspond to a special token
                if self.tokenizer.convert_ids_to_tokens(mt_token_id) in self.tokenizer.all_special_tokens:
                    continue

                mt_token = self.tokenizer.convert_ids_to_tokens(mt_token_id)[0]
                mt_token_vocab_start = self.tokenizer.get_vocab(
                )[mt_token + foldseek_struc_vocab[0]]

                wt_prob = log_probs[0, idx + 1, wt_token_vocab_start: wt_token_vocab_start + len(
                    foldseek_struc_vocab)].sum()
                mt_prob = log_probs[0, idx + 1, mt_token_vocab_start: mt_token_vocab_start + len(
                    foldseek_struc_vocab)].sum()

                score = mt_prob - wt_prob
                mutation_scores[f'{wt_token}{idx + 1}{mt_token}'] = score.item()

        return mutation_scores


@app.function(gpu="any", image=images.base, timeout=4000, volumes={MODEL_DIR: volumes.model_weights})
def score_mutations(sequence: str, model_name: str, metric: str, offset_idx: int = 1, mutations: list[str] | None = None,) -> float:
    """
    Calculate the score of mutations based on the specified metric.

    Args:
        sequence (str): The original protein sequence.
        mutations (list[str]): The mutations to score. Format is ['W123A', 'Y456F', ...].
        metric (str): The scoring metric to use. Options include 'wildtype_marginal', 'masked_marginal', or 'pppl'.
        offset_idx (int, optional): The offset index to adjust the mutation index. Defaults to 1.
    Returns:
        float: The score of the mutations according to the specified metric.
    """
    import pandas as pd
    if metric not in ['wildtype_marginal', 'masked_marginal', 'pppl']:
        raise ValueError(
            "The metric must be one of 'wildtype_marginal', 'masked_marginal', or 'pppl'")

    scorer = MLMScorer(model_name=model_name)
    mutation_scores = scorer.deep_mutational_scan.remote(sequence, metric)

    return pd.DataFrame(list(mutation_scores.items()), columns=["mutation", "score"])
    scores = []
    for mutation in mutations:
        wt, idx, mt = mutation[0], int(
            mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        mutation_key = f'{wt}{idx + 1}{mt}'
        if mutation_key in mutation_scores:
            scores.append(mutation_scores[mutation_key])
        else:
            raise ValueError(
                f"Mutation {mutation_key} not found in the computed mutation scores")

    return scores
