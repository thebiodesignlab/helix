from modal import Image

base = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch",
        "biopython",
        "matplotlib",
        "transformers",
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
