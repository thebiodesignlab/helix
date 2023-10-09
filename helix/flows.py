import os

import numpy as np
from helix.esm import ESMFold, EsmModel, dockerhub_image
from helix.main import CACHE_DIR, stub, volume
from Bio import SeqIO
from Bio.PDB.PDBIO import PDBIO

# Define type or data strcuture for possible models for structure prediction
PROTEIN_STRUCTURE_MODELS = {
    "esmfold": ESMFold
}


@stub.function(network_file_systems={CACHE_DIR: volume}, image=dockerhub_image)
def predict_structures(sequences, model_name: str = "esmfold"):
    if model_name not in PROTEIN_STRUCTURE_MODELS:
        raise ValueError(
            f"Model {model_name} is not supported. Supported models are: {list(PROTEIN_STRUCTURE_MODELS.keys())}")
    print(f"Using model {model_name}")
    print(f"Predicting structures for {len(sequences)} sequences")
    model = PROTEIN_STRUCTURE_MODELS[model_name]()

    result = []
    for struct in model.infer.map(sequences, return_exceptions=True):
        if isinstance(struct, Exception):
            print(f"Error: {struct}")
        else:
            print(f"Successfully predicted structure for {struct.id}")
            result.append(struct)
    return result


@stub.function(network_file_systems={CACHE_DIR: volume}, image=dockerhub_image)
def perform_pca_on_embeddings(embeddings, n_components: int = 2):
    import io
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)

    # Create a scatter plot of the first two principal components
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1])
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA of Protein Embeddings')
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return img


@stub.function(gpu='any', network_file_systems={CACHE_DIR: volume}, image=dockerhub_image)
def get_embeddings(sequences, model_name: str = "facebook/esm2_t36_3B_UR50D", batch_size: int = 32):
    import numpy as np
    model = EsmModel(model_name=model_name)
    embeddings = []

    # Split sequences into batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i:i + batch_size])

    for results in model.infer.map(batches, return_exceptions=True):
        if isinstance(results, Exception):
            print(f"Error: {results}")
        else:
            print(f"Successfully predicted embeddings for {results}")
            # Get the mean of the embeddings to get a single embedding for each sequence
            embedding = results.last_hidden_state.cpu().detach().numpy().mean(axis=1)
            # Reshape to 2D
            embedding = embedding.reshape(-1, embedding.shape[-1])
            embeddings.append(embedding)

    return np.concatenate(embeddings, axis=0)  # Return a 2D NumPy array


@stub.local_entrypoint()
def predict_structures_from_fasta(fasta_file: str, output_dir: str):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    result = predict_structures.remote(sequences)
    os.makedirs(output_dir, exist_ok=True)
    for struct in result:
        io = PDBIO()
        io.set_structure(struct)
        io.save(f"{output_dir}/{struct.id}.pdb")


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
def get_embeddings_from_fasta(fasta_file: str, model_name: str = "facebook/esm2_t36_3B_UR50D", batch_size: int = 32):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    embeddings = get_embeddings.remote(sequences, model_name, batch_size)
    return embeddings
