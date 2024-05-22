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
