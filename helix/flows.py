from itertools import repeat
import os

import numpy as np
from .esm import EsmModel, EsmForMaskedLM, image as esm_image
from .main import CACHE_DIR, stub, volume
from Bio import SeqIO

from .utils import create_batches


@stub.function(network_file_systems={CACHE_DIR: volume}, image=esm_image)
def perform_pca_on_embeddings(embeddings_dict, n_components: int = 2):
    import io
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Extract embeddings and sequence IDs from the dictionary
    embeddings = list(embeddings_dict.values())
    sequence_ids = list(embeddings_dict.keys())

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)

    # Create a scatter plot of the first two principal components
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1])

    # Add labels to the points
    for i, sequence_id in enumerate(sequence_ids):
        ax.text(pca_embeddings[i, 0], pca_embeddings[i, 1], sequence_id)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA of Protein Embeddings')
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return img


@stub.function(gpu='any', network_file_systems={CACHE_DIR: volume}, image=esm_image)
def get_embeddings(sequences, model_name: str = "facebook/esm2_t36_3B_UR50D", batch_size: int = 32):
    model = EsmModel(model_name=model_name)
    embeddings = {}

    batched_sequences = create_batches(sequences, batch_size)
    batched_results = model.infer.map(
        batched_sequences, return_exceptions=True)
    for result_batch, sequence_batch in zip(batched_results, batched_sequences):
        if isinstance(result_batch, Exception):
            print(f"Error: {result_batch}")
        else:
            embeddings_batch = result_batch.last_hidden_state.cpu().detach().numpy().mean(axis=1)
            for i, sequence in enumerate(sequence_batch):
                embeddings[sequence.id] = embeddings_batch[i]

    return embeddings  # Return a dictionary of embeddings


@stub.function(gpu='any', network_file_systems={CACHE_DIR: volume}, image=esm_image)
def get_scores(sequences, model_name: str = "facebook/esm2_t33_650M_UR50D", batch_size: int = 32):
    model = EsmForMaskedLM(model_name=model_name)
    perplexities = {}

    results = model.score.map(
        [str(sequence.seq) for sequence in sequences], return_exceptions=True)
    for result, sequence in zip(results, sequences):
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            perplexities[sequence.id] = result

    return perplexities  # Return a dictionary of perplexities


@stub.function(gpu='any', network_file_systems={CACHE_DIR: volume}, image=esm_image)
def get_attentions(sequences, model_name: str = "facebook/esm2_t36_3B_UR50D", batch_size: int = 32):

    model = EsmModel(model_name=model_name)
    attentions = {}

    batched_sequences = create_batches(sequences, batch_size)
    batched_results = model.infer.starmap(
        zip(batched_sequences, repeat(False), repeat(True)), return_exceptions=True)
    for result_batch, sequence_batch in zip(batched_results, batched_sequences):
        if isinstance(result_batch, Exception):
            print(f"Error: {result_batch}")
        else:
            # Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
            attentions_batch = result_batch.attentions
            # Take the last layer and average over the heads TODO: Make this configurable
            attentions_batch = attentions_batch[-1].cpu(
            ).detach().numpy().mean(axis=1)

            for i, sequence in enumerate(sequence_batch):
                attentions[sequence.id] = attentions_batch[i]

    return attentions


@stub.local_entrypoint()
def pca_from_fasta(fasta_file: str, model_name: str = "facebook/esm2_t36_3B_UR50D", batch_size: int = 32, n_components: int = 2, output_dir: str = None):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    embeddings = get_embeddings.remote(sequences, model_name, batch_size)
    fig = perform_pca_on_embeddings.remote(embeddings, n_components)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/pca.png", "wb") as f:
            f.write(fig.read())
        # Save the embeddings as a NumPy array
        np.save(f"{output_dir}/embeddings.npy", embeddings)


@stub.local_entrypoint()
# Batch size should be low for attention maps, otherwise it will run out of memory
def get_attentions_from_fasta(fasta_file: str, model_name: str = "facebook/esm2_t36_3B_UR50D", batch_size: int = 1):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    attentions = get_attentions.remote(sequences, model_name, batch_size)
    return attentions


@stub.local_entrypoint()
# Batch size should be low for attention maps, otherwise it will run out of memory
def get_embeddings_from_fasta(fasta_file: str, model_name: str = "facebook/esm2_t36_3B_UR50D", batch_size: int = 32):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    embeddings = get_embeddings.remote(sequences, model_name, batch_size)
    print(embeddings)
    return embeddings
