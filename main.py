import pathlib
from fastapi import FastAPI
from modal import Image, Stub, NetworkFileSystem, asgi_app
from fastapi.responses import FileResponse
import os

VOLUME_NAME = "job-storage-vol"

CACHE_DIR = "/cache"

RESULTS_DIR = pathlib.Path(CACHE_DIR, "results")

volume = NetworkFileSystem.persisted(VOLUME_NAME)


web_app = FastAPI()
stub = Stub(name="proteinator")
image = Image.debian_slim()

@web_app.get("/results/{job_id}")
def download_results(job_id: str):
    import tarfile
    job_results_path = f"{RESULTS_DIR}/{job_id}"
    # Compress the results into a gzip file
    with tarfile.open(f"/tmp/{job_id}.tar.gz", "w:gz") as tar:
        tar.add(job_results_path, arcname=os.path.basename(job_results_path))
    
    # Return the gzip file as a response
    return FileResponse(f"/tmp/{job_id}.tar.gz", media_type="application/gzip", filename=f"{job_id}.tar.gz")


@stub.function(image=image,network_file_systems={CACHE_DIR: volume},name="proteinator_api")
@asgi_app()
def proteinator_api():
    return web_app
