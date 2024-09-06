from typing import List, Dict, Any, Literal
from loguru import logger
import numpy as np
from helix.core import app, images, volumes
from helix.utils.filecache import file_cache
from modal import gpu, batched

EmbeddingStrategy = Literal["cls", "mean", "max"]
MODEL_DIR = "/mnt/models"
CACHE_DIR = "/mnt/cache"


@app.function(image=images.base, gpu=gpu.A100(size='40GB'), volumes={MODEL_DIR: volumes.model_weights, CACHE_DIR: volumes.cache}, timeout=4000)
@batched(max_batch_size=30, wait_ms=1000)
@file_cache(verbose=True, cache_dir=CACHE_DIR)
def compute_batch_embeddings(
    batch: List[str],
    model_names: List[str],
    embedding_strategies: List[EmbeddingStrategy],
    max_lengths: List[int]
) -> List[np.ndarray]:
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = model_names[0]
    embedding_strategy = embedding_strategies[0]
    max_length = max_lengths[0]

    logger.info(
        f"Processing batch of size {len(batch)} with max_length {max_length}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=MODEL_DIR)
    model = AutoModel.from_pretrained(model_name, cache_dir=MODEL_DIR)
    model.eval()

    # Tokenize with error handling
    try:
        inputs = tokenizer(batch, return_tensors="pt",
                           padding=True, truncation=True, max_length=max_length)
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        logger.error(f"Problematic batch: {batch}")
        raise

    logger.info(f"Tokenized input shape: {inputs['input_ids'].shape}")

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if embedding_strategy == "cls":
        embeddings = outputs.last_hidden_state[:, 0, :]
    elif embedding_strategy == "mean":
        embeddings = outputs.last_hidden_state.mean(dim=1)
    elif embedding_strategy == "max":
        embeddings = outputs.last_hidden_state.max(dim=1).values
    else:
        raise ValueError(
            f"Invalid embedding strategy: {embedding_strategy}")

    return [emb.cpu().numpy() for emb in embeddings]


@app.function(image=images.base, gpu="any", volumes={MODEL_DIR: volumes.model_weights, CACHE_DIR: volumes.cache}, timeout=10000)
def get_protein_embeddings(
    sequences: List[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    embedding_strategy: EmbeddingStrategy = "cls",
    max_length: int = 1024,
) -> Dict[str, Any]:
    """
    Get protein embeddings using a specified Hugging Face model.

    Args:
        sequences (List[str]): List of protein sequences.
        model_name (str): Name of the Hugging Face model to use.
        embedding_strategy (EmbeddingStrategy): Strategy for embedding extraction.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        Dict[str, Any]: Dictionary containing embeddings and model info.
    """

    # Process all sequences at once
    args_list = [(sequence, model_name, embedding_strategy, max_length)
                 for sequence in sequences]
    all_embeddings = list(compute_batch_embeddings.starmap(
        args_list, return_exceptions=True))

    # Replace exceptions with None
    all_embeddings = [None if isinstance(
        emb, Exception) else emb for emb in all_embeddings]

    logger.info(
        f"Number of valid embeddings (not None): {sum(1 for emb in all_embeddings if emb is not None)}")
    logger.info(f"Total number of embeddings: {len(all_embeddings)}")

    return {
        "embeddings": all_embeddings,
        "model_name": model_name,
        "embedding_strategy": embedding_strategy,
    }


@app.local_entrypoint()
def get_protein_embedding_from_seq(
    sequences: str,  # Comma-separated list of sequences
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    embedding_strategy: str = "cls",
    max_length: int = 1024,
    batch_size: int = 32,
):
    try:
        sequences_list = sequences.split(",")
        result = get_protein_embeddings.remote(
            sequences_list, model_name, embedding_strategy, max_length, batch_size
        )
        logger.info(f"Generated embeddings using {result['model_name']}")
        logger.info(f"Embedding dimension: {result['embedding_dim']}")
        logger.info(f"Embedding strategy: {result['embedding_strategy']}")
        logger.info(
            f"Number of sequences processed: {len(result['embeddings'])}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
