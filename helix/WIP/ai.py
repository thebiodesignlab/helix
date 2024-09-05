import modal
from .main import stub


@stub.local_entrypoint()
def main():
    sb = stub.spawn_sandbox(
        "python",
        "-c",
        """
import helix
from helix.esm import EsmModel
with helix.stub.run():
    model = EsmModel(model_name="facebook/esm1b_t33_650M_UR50S")
    print(model.infer.remote(['MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG']))
        """,
        image=modal.Image.debian_slim().pip_install("helixbio"),
        mounts=[modal.Mount.from_local_python_packages("helix")],
        secrets=[modal.secret.Secret.from_name("MODAL_SECRETS")],
        timeout=1200,  # 10 minutes
        gpu="any",
    )

    sb.wait()
    print(sb.stdout.read())
    if sb.returncode != 0:
        print(f"Tests failed with code {sb.returncode}")
        print(sb.stderr.read())
