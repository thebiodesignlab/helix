<div align="center">
logo
<hr>

### **✨ Run large protein models in less than 30 seconds with Modal. Open an issue if it takes longer! ✨**
[![PyPI version](https://badge.fury.io/py/helixbio.svg)](https://badge.fury.io/py/helixbio)
</div>

---
Running large models and code that scales for big datasets in this repository is enabled by [Modal](https://modal.com) (no affiliation). It allows us to run code in the cloud on thousands of containers and GPUs without having to think for a second about infrastructure.

#### Currently implemented
- [ESMFold](https://github.com/facebookresearch/esm#esmfold)
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- Parallelized codon optimization with [DNAChisel](https://github.com/Edinburgh-Genome-Foundry/DnaChisel)
#### Up Next
- RFDiffusion
- EvoProtGrad
- Evodiff

## ⚙️ Getting started

1. Create an account at [modal.com](https://modal.com).
3. Open a terminal and run (requires Python 3.10+): `pip install helixbio`
4. Run `modal token new`

## 🧬 Run your first model

Let's predict a protein structure using ESMFold. This also works in parallel for multiple sequences.

```bash
modal run helix.esm::predict_structures_from_fasta --fasta-file "my_lovely_proteins.fasta" --output-dir "my_lovely_structures"
```
## Examples

Generate protein variants using [EvoProtGrad](https://github.com/NREL/EvoProtGrad/).

```bash
modal run helix.evoprotgrad::get_evoprotgrad_variants --sequence "MSGKIDKILIVGGGTAGWMAASYLGKALQGTADITLLQAPDIP"  --max-mutations 4 --num-chains 20 --n-steps 200 --output-csv-file "my-variants.csv" --output-fasta-file "my-variants.fasta"
```

## Contributing

We welcome contributions of any size! Below are some good ways to get started.

-   **GitHub Discussions**: A great way to talk about features you want added or things that are confusing/need clarification.
-   **GitHub Issues**: These are an excellent way to report bugs. Additionally, you can try and solve an existing issue and submit a PR.

We are actively looking for contributors, no matter your skill level or experience.

## License

Helix is open-source and licensed under the [Apache License 2.0](LICENSE).
