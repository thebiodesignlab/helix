from modal import Volume


model_weights = Volume.from_name(
    "model-weights", create_if_missing=True)
databases = Volume.from_name("databases", create_if_missing=True)

results = Volume.from_name("results", create_if_missing=True)
