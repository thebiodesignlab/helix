def create_batches(sequences, batch_size: int = 32):
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i:i + batch_size])
    return batches
