import pathlib
from fastapi import FastAPI
from modal import Image, Stub, Volume, Mount

MOUNT_PATH = pathlib.Path("/mnt")
PROTEIN_DBS_PATH = MOUNT_PATH / "protein_dbs"

VOLUME_CONFIG = {
    PROTEIN_DBS_PATH: Volume.from_name(
        "protein-dbs", create_if_missing=True)}


web_app = FastAPI()
stub = Stub(name="helix", mounts=[Mount.from_local_python_packages("helix")])
image = Image.debian_slim()


# @stub.function()
# @web_app.get("/results/{job_id}")
# def download_results(job_id: str):
#     import tarfile
#     job_results_path = f"{RESULTS_DIR}/{job_id}"
#     # Compress the results into a gzip file
#     with tarfile.open(f"/tmp/{job_id}.tar.gz", "w:gz") as tar:
#         tar.add(job_results_path, arcname=os.path.basename(job_results_path))

#     # Return the gzip file as a response
#     return FileResponse(f"/tmp/{job_id}.tar.gz", media_type="application/gzip", filename=f"{job_id}.tar.gz")


# @stub.function()
# @web_app.get("/results")
# def list_results():
#     results = os.listdir(RESULTS_DIR)
#     return HTMLResponse(
#         content="<br>".join(
#             f'<a href="/results/{result}">{result}</a>' for result in results
#         )
#     )


# @stub.function(image=image, network_file_systems={CACHE_DIR: volume}, name="helix_api")
# @asgi_app()
# def helix_api():
#     return web_app
