from modal import Volume, CloudBucketMount, Secret


model_weights = Volume.from_name(
    "model-weights", create_if_missing=True)

rosetta = CloudBucketMount(
    "helix", secret=Secret.from_name("cloudflare-r2-secrets"), bucket_endpoint_url="https://b6e2e34967f8dc08d00fe1b17a2fd681.r2.cloudflarestorage.com")

mmseqs_databases = Volume.from_name("mmseqs-databases", create_if_missing=True)

results = Volume.from_name("results", create_if_missing=True)

cache = Volume.from_name("function-cache", create_if_missing=True)
