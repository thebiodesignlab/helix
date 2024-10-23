from modal import Image

base = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "biopython",
        "matplotlib",
        "transformers",
        "loguru",
        "pandas")
)

mmseqs = Image.micromamba().apt_install("wget", "git", "tar").micromamba_install(
    "mmseqs2",
    "foldseek",
    channels=[
        "bioconda",
        "conda-forge"
    ],
).pip_install("pandas", "numpy", "biopython")
