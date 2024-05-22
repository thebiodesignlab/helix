from typing import Union
from helix.core import volumes, images
from helix.core import app
from modal import enter, method
import transformers
import torch

MODEL_DIR = "/vol/models"


@app.cls(gpu="any", volumes={MODEL_DIR: volumes.model_weights}, image=images.base)
class MLMScorer:
    def __init__(self, model_name):
        self.model_name = model_name

    @enter()
    def setup(self):
        volumes.model_weights.reload()  # Fetch latest changes to the volume
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(
            self.model_name, cache_dir=MODEL_DIR)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=MODEL_DIR)
        volumes.model_weights.commit()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def encode_sequences(self, sequences: Union[str, list[str]]) -> torch.Tensor:
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
    def compute_token_log_probs(self, sequences: Union[str, list[str]]) -> torch.Tensor:
        """
        Compute the log probabilities of the given sequences using the model.

        Args:
            sequences (Union[str, List[str]]): The sequence or list of sequences to compute log probabilities for.

        Returns:
            torch.Tensor: The log probabilities tensor.
        """
        token_probs_list = []
        batch_tokenized_sequences = self.encode_sequences(sequences)

        with torch.inference_mode():
            logits = self.model(batch_tokenized_sequences).logits
            token_probs = torch.log_softmax(logits, dim=-1)

        token_probs_list.append(token_probs)
        token_probs = torch.cat(token_probs_list, dim=0)

        return token_probs

    @method()
    def compute_masked_token_log_probs(self, sequences: Union[str, list[str]]) -> torch.Tensor:
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
    def score_mutations(self, sequence: str, mutations: list[str], metric: str, offset_idx: int = 1) -> float:
        """
        Calculate the score of mutations based on the specified metric.

        Args:
            sequence (str): The original protein sequence.
            mutation (str): The mutation to score. Format is 'W123A'.
            metric (str): The scoring metric to use. Options include 'wildtype_marginal', 'masked_marginal', or 'pppl'.
            offset_idx (int, optional): The offset index to adjust the mutation index. Defaults to 1.
        Returns:
            float: The score of the mutation according to the specified metric.
        """

        if metric not in ['wildtype_marginal', 'masked_marginal', 'pppl']:
            raise ValueError(
                "The metric must be one of 'wildtype_marginal', 'masked_marginal', or 'pppl'")

        scores = []

        if metric == 'wildtype_marginal':
            log_probs = self.compute_token_log_probs.remote(sequence)

        elif metric == 'masked_marginal':
            log_probs = self.compute_masked_token_log_probs.remote(sequence)

        for mutation in mutations:
            wt, idx, mt = mutation[0], int(
                mutation[1:-1]) - offset_idx, mutation[-1]
            assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

            if metric == 'pppl':
                sequence = sequence[:idx] + mt + sequence[(idx + 1):]
                score = 0
                log_probs = self.compute_token_log_probs.remote(sequence)
                # Add 1 to the index to account for the [BOS] token
                for i in range(1, len(sequence)):
                    position_encoded = self.tokenizer.convert_tokens_to_ids(
                        [sequence[i]])[0]
                    score += log_probs[0, i, position_encoded]
                scores.append(score.item())
            else:
                wt_encoded, mt_encoded = self.tokenizer.convert_tokens_to_ids(
                    [wt]), self.tokenizer.convert_tokens_to_ids([mt])

                score = log_probs[0, 1 + idx, mt_encoded[0]] - \
                    log_probs[0, 1 + idx, wt_encoded[0]]
                scores.append(score.item())
        print(scores)
        return scores


@app.local_entrypoint()
def score_mutation(sequence: str, mutation: str, model_name: str = "facebook/esm1b_t33_650M_UR50S") -> float:
    """
    Entry point to score a single mutation in a given protein sequence.

    Args:
        sequence (str): The original protein sequence.
        mutation (str): The mutation to score, formatted as 'W123A'.

    Returns:
        float: The score of the mutation using the default metric.
    """
    scorer = MLMScorer(model_name=model_name)
    scores = scorer.score_mutations.remote(
        sequence, [mutation], 'pppl')
    print(f"Score for mutation {mutation}: {scores[0]}")
