from helix.core import volumes, images
from helix.core import app
from modal import enter, method, gpu

MODEL_DIR = "/mnt/models"


def compute_score(log_probs, wt, idx, mt, vocab):
    """
    Compute the score of a mutation based on log probabilities.

    Args:
        log_probs (torch.Tensor): The log probabilities tensor.
        wt (str): The wildtype amino acid.
        idx (int): The index of the mutation.
        mt (str): The mutant amino acid.
        vocab (dict): The vocabulary mapping tokens to indices.
    Returns:
        float: The score of the mutation.
    """

    wt_encoded, mt_encoded = vocab[wt], vocab[mt]

    score = log_probs[0, 1 + idx, mt_encoded] - \
        log_probs[0, 1 + idx, wt_encoded]
    return score.item()


def compute_saprot_score(log_probs, wt, idx, mt, vocab):
    """
    Compute the score of a mutation based on log probabilities.

    Args:
        log_probs (torch.Tensor): The log probabilities tensor.
        wt (str): The wildtype amino acid.
        idx (int): The index of the mutation.
        mt (str): The mutant amino acid.
        vocab (dict): The vocabulary mapping tokens to indices.

    Returns:
        float: The score of the mutation.
    """
    foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
    # assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    wt_token_vocab_start = vocab[wt + foldseek_struc_vocab[0]]
    mt_token_vocab_start = vocab[mt + foldseek_struc_vocab[0]]
    wt_prob = log_probs[0, idx + 1, wt_token_vocab_start: wt_token_vocab_start + len(
        foldseek_struc_vocab)].sum()
    mt_prob = log_probs[0, idx + 1, mt_token_vocab_start: mt_token_vocab_start + len(
        foldseek_struc_vocab)].sum()
    score = mt_prob - wt_prob
    return score.item()


@app.cls(gpu=gpu.A100(), volumes={MODEL_DIR: volumes.model_weights}, image=images.base, timeout=4000)
class MLMScorer:
    def __init__(self, model_name, sequence):
        self.model_name = model_name
        self.sequence = sequence
        self.token_log_probs = None
        self.masked_token_log_probs = None

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
        self.encoded_sequence = self.tokenizer(
            self.sequence, return_tensors="pt", add_special_tokens=True)['input_ids'].to(self.model.device)
        self.vocab = self.tokenizer.get_vocab()

    def compute_token_log_probs(self):
        """
        Compute the log probabilities of the given sequences using the model.

        Args:
            sequences (Union[str, List[str]]): The sequence or list of sequences to compute log probabilities for.

        Returns:
            torch.Tensor: The log probabilities tensor.
        """
        import torch
        if self.token_log_probs is not None:
            return self.token_log_probs
        token_probs_list = []

        with torch.inference_mode():
            logits = self.model(self.encoded_sequence).logits
            token_probs = torch.log_softmax(logits, dim=-1)

        token_probs_list.append(token_probs)
        token_probs = torch.cat(token_probs_list, dim=0)
        self.token_log_probs = token_probs
        return token_probs

    def compute_masked_token_log_probs(self):
        import torch
        if self.masked_token_log_probs is not None:
            return self.masked_token_log_probs
        all_token_probs = []

        for i in range(self.encoded_sequence.size(1)):
            with torch.inference_mode():
                # batch size, seq len, vocab size
                batch_masked_tokens = self.encoded_sequence.clone()
                batch_masked_tokens[0, i] = self.tokenizer.mask_token_id
                logits = self.model(batch_masked_tokens).logits
                token_probs = torch.log_softmax(logits, dim=-1)
                all_token_probs.append(token_probs)
        token_probs = torch.cat(all_token_probs, dim=0)
        self.masked_token_log_probs = token_probs
        return token_probs

    @method()
    def score_mutations(self, mutations: list[str], metric: str, offset_idx: int = 1):
        """
        Calculate the score of a mutation based on the specified metric.

        Args:
            mutation (str): The mutation to score. Format is 'W123A'.
            metric (str): The scoring metric to use. Options include 'wildtype_marginal', 'masked_marginal', or 'pppl'.
            offset_idx (int, optional): The offset index to adjust the mutation index. Defaults to 1.
        Returns:
            float: The score of the mutation according to the specified metric.
        """
        scores = []
        for mutation in mutations:
            if metric not in ['wildtype_marginal', 'masked_marginal', 'pppl']:
                raise ValueError(
                    "The metric must be one of 'wildtype_marginal', 'masked_marginal', or 'pppl'")

            if metric == 'wildtype_marginal':
                log_probs = self.compute_token_log_probs()

            elif metric == 'masked_marginal':
                log_probs = self.compute_masked_token_log_probs()

            wt, idx, mt = mutation[0], int(
                mutation[1:-1]) - offset_idx, mutation[-1]
            # assert self.sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
            if "saprot" in self.model_name.lower():
                tokenized_sequence = self.tokenizer.tokenize(self.sequence)
                assert tokenized_sequence[idx][
                    0] == wt, f'The listed wildtype {wt} does not match the provided sequence {tokenized_sequence[idx][0]} at position {idx}'
                scores.append(compute_saprot_score(
                    log_probs, wt, idx, mt, self.vocab))
            else:
                assert self.sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
                scores.append(compute_score(
                    log_probs, wt, idx, mt, self.vocab))
        return scores


@app.function(gpu="any", image=images.base, timeout=5000)
def score_mutations(model_name: str, sequence: str, mutations: list[str], metric: str, offset_idx: int = 1) -> list[float]:
    """
    Calculate the score of mutations based on the specified metric.

    Args:
        model_name (str): The name of the model to use for scoring.
        sequence (str): The original protein sequence.
        mutations (list[str]): The mutations to score. Format is ['W123A', 'Y456F', ...].
        metric (str): The scoring metric to use. Options include 'wildtype_marginal', 'masked_marginal', or 'pppl'.
        offset_idx (int, optional): The offset index to adjust the mutation index. Defaults to 1.
    Returns:
        list[float]: The scores of the mutations according to the specified metric.
    """

    if metric not in ['wildtype_marginal', 'masked_marginal', 'pppl']:
        raise ValueError(
            "The metric must be one of 'wildtype_marginal', 'masked_marginal', or 'pppl'")

    scorer = MLMScorer(model_name=model_name, sequence=sequence)
    scores = scorer.score_mutations.remote(mutations, metric, offset_idx)
    return scores
