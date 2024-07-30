from typing import List, Dict, Any, Literal

from loguru import logger
from helix.core import app, images, volumes

EmbeddingStrategy = Literal["cls", "mean", "max"]
MODEL_DIR = "/mnt/models"


@app.function(image=images.base, gpu="A100", volumes={MODEL_DIR: volumes.model_weights}, timeout=10000)
def get_protein_embeddings(
    sequences: List[str],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    embedding_strategy: EmbeddingStrategy = "cls",
    max_length: int = 1024,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Get protein embeddings using a specified Hugging Face model.

    Args:
        sequences (List[str]): List of protein sequences.
        model_name (str): Name of the Hugging Face model to use.
        embedding_strategy (EmbeddingStrategy): Strategy for embedding extraction.
        max_length (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for processing sequences.

    Returns:
        Dict[str, Any]: Dictionary containing embeddings and model info.
    """
    from transformers import AutoTokenizer, AutoModel
    import torch
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=MODEL_DIR)
        model = AutoModel.from_pretrained(model_name, cache_dir=MODEL_DIR)
        model.eval()

        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt",
                               padding=True, truncation=True, max_length=max_length)

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

            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()

        return {
            "embeddings": all_embeddings.tolist(),
            "model_name": model_name,
            "embedding_dim": all_embeddings.shape[1],
            "embedding_strategy": embedding_strategy,
        }

    except Exception as e:
        logger.error(f"Error in get_protein_embeddings: {str(e)}")
        raise


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
