import hashlib
from modal import Image, method
import pandas as pd
from helix.core import app


def download_esm_models(slugs: list[str] = ["facebook/esm1b_t33_650M_UR50S", "facebook/esm2_t33_650M_UR50D", "facebook/esm2_t36_3B_UR50D", "facebook/esm2_t48_15B_UR50D"]):
    from transformers import EsmForMaskedLM, AutoTokenizer
    for slug in slugs:
        EsmForMaskedLM.from_pretrained(slug)
        AutoTokenizer.from_pretrained(slug)


image = Image.debian_slim().pip_install(
    "transformers[torch]==4.30.0",
    "torch",
    "evo_prot_grad",
    "pandas").run_function(download_esm_models)


@app.cls(gpu='A100', timeout=2000, image=image, allow_cross_region_volumes=True, concurrency_limit=9)
class EvoProtGrad:
    def __init__(self, experts: list[str] = ["esm"], device: str = "cuda"):
        from evo_prot_grad import get_expert
        from transformers import EsmForMaskedLM, AutoTokenizer
        self.experts = []
        for expert in experts:
            model = None
            tokenizer = None
            if "/esm" in expert:
                model = EsmForMaskedLM.from_pretrained(expert)
                tokenizer = AutoTokenizer.from_pretrained(expert)
                expert = "esm"
            self.experts.append(get_expert(
                expert_name=expert, temperature=1.0, device=device, model=model, tokenizer=tokenizer))

    @method()
    def evolve(self, sequence: str, n_steps: int = 100, parallel_chains: int = 10, max_mutations: int = -1, random_seed: int = None):
        from evo_prot_grad import DirectedEvolution

        try:
            variants, scores = DirectedEvolution(wt_protein=sequence, experts=self.experts, n_steps=n_steps,
                                                 parallel_chains=parallel_chains, max_mutations=max_mutations, random_seed=random_seed, output="best")()
            variants = [variant.replace(' ', '') for variant in variants]
        except Exception as e:
            print(e)
            return e, None
        return variants, scores


@app.local_entrypoint()
def get_evoprotgrad_variants(sequence: str, output_csv_file: str = None, output_fasta_file: str = None, experts: str = "esm", n_steps: int = 100, num_chains: int = 20, max_mutations: int = -1, random_seed: int = None, batch_size: int = 9):
    from .evoprotgrad import EvoProtGrad
    from helix.utils.sequence import dataframe_to_fasta

    experts = experts.split(",")
    evoprotgrad = EvoProtGrad(experts=experts)

    if output_csv_file is None and output_fasta_file is None:
        raise Exception(
            "Must specify either output_csv_file or output_fasta_file")

    num_calls = num_chains // batch_size
    remaining_chains = num_chains % batch_size
    print(
        f"Running {num_chains} parallel chains in {num_calls+1} containers")

    results = []
    args = [(sequence, n_steps, batch_size, max_mutations, random_seed)
            for _ in range(num_calls)]
    if remaining_chains > 0:
        args.append((sequence, n_steps, remaining_chains,
                    max_mutations, random_seed))
    for variants, scores in evoprotgrad.evolve.starmap(args, return_exceptions=True):
        if isinstance(variants, Exception):
            print(f"Error: {variants}")
        else:
            for variant, score in zip(variants, scores):
                num_mutations = sum(1 for wt, mut in zip(
                    sequence, variant) if wt != mut)
                results.append({
                    'variant': variant.replace(' ', ''),
                    'score': score,
                    'num_mutations': num_mutations,
                    'experts': ' '.join(experts),
                    'n_steps': n_steps,
                    'max_mutations': max_mutations,
                    'random_seed': random_seed
                })

    print(f"Successfully generated {len(results)} variants")

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # hash the sequence to get an ID
    df_results['id'] = df_results['variant'].apply(
        lambda x: hashlib.sha1(x.encode()).hexdigest())

    # Remove duplicates
    df_results.drop_duplicates(subset=['id'], inplace=True)

    # Remove variants with no mutations
    df_results = df_results[df_results['num_mutations'] > 0]

    df_results['mutations'] = df_results['variant'].apply(
        lambda x: ', '.join([f"{wt}{pos+1}{mut}" for pos, (wt, mut) in enumerate(zip(sequence, x)) if wt != mut]))

    # Sort the DataFrame by score in descending order
    df_results.sort_values(by='score', ascending=False, inplace=True)

    if output_csv_file is not None:
        df_results.to_csv(output_csv_file, index=False)
    if output_fasta_file is not None:
        with open(output_fasta_file, "w") as f:
            fasta_content = dataframe_to_fasta(
                df_results, id_col='id', seq_col='variant')
            f.write(fasta_content)
