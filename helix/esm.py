import uuid
from modal import Image, method
import os
from helix.main import CACHE_DIR, RESULTS_DIR, volume, stub


def download_model():
    import esm
    esm.pretrained.esm2_t36_3B_UR50D()
    esm.pretrained.esmfold_v1()


dockerhub_image = Image.from_registry(
    "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel"
).apt_install("git"
              ).pip_install("fair-esm[esmfold]",
                            "dllogger @ git+https://github.com/NVIDIA/dllogger.git",
                            "openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307"
                            ).run_function(download_model
                                           ).pip_install("gradio",
                                                         "biopython",
                                                         "pandas")


@stub.cls(gpu='A10G', timeout=2000, network_file_systems={CACHE_DIR: volume}, image=dockerhub_image)
class ESMFold():
    def __enter__(self):
        import esm
        import torch
        self.model = esm.pretrained.esmfold_v1()
        self.model.set_chunk_size(64)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @method()
    def predict(self, job_id, sequence, label):
        import torch
        with torch.inference_mode():
            output = self.model.infer_pdb(sequence)
        output_path = f"{RESULTS_DIR}/{job_id}"
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/{label}.pdb", "w") as f:
            f.write(output)
        return output, label

    def __exit__(self, exc_type, exc_value, traceback):
        import torch
        torch.cuda.empty_cache()


@stub.local_entrypoint()
def predict_structures(fasta_file: str, output_dir: str):
    from Bio import SeqIO
    job_id = uuid.uuid4()

    print("Structure prediction job started...")
    print(
        f"Retrieved results can also be downloaded from the cloud volume using job id {job_id}")
    model = ESMFold()
    for result in model.predict.starmap(((job_id, str(record.seq), record.id) for record in SeqIO.parse(fasta_file, "fasta")), return_exceptions=True):
        if isinstance(result, Exception):
            print(f"Error: {result}")
        elif output_dir:
            print(f"Writing structure for {result[1]} to {output_dir}")
            with open(f"{output_dir}/{result[1]}.pdb", "w") as f:
                f.write(result[0])
    print("Structure prediction job finished successfully.")
