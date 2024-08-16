# Mixture of Experts

## Getting Started

```bash
conda create -n moe python=3.11 -y
conda activate moe

pip install -e .
pip install -r requirements.txt
```

If you want to extract tiles, also install openslide via conda.

```bash
conda install -c conda-forge openslide openslide-python
```

## Architecture Definitions

To run experiments, we use the definitions in the `configs` directory. Each config file can be used to instantiate various model architectures using `hydra`.
