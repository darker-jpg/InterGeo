# InterGeo

**Interface-Centric Geometric Representation Learning for Predicting Mutation
Effects Protein–Protein Interactions**

InterGeo provides local-geometry vector-quantized pretraining and paired
wild-type/mutant prediction of mutation-induced protein–protein binding affinity
changes (DDG).

## Installation

```bash
conda env create -f environment.yml
conda activate intergeo
python -m pip install -e .
```

## Data

- [SKEMPI v2.0](https://life.bsc.es/pid/skempi2/database/index): mutation and
  protein–protein binding data, including the downloadable CSV and cleaned PDB
  structures.
- [PDBbind+](https://www.pdbbind-plus.org.cn/): protein-complex structures and
  binding-affinity data used to prepare pretraining inputs. Registration and
  licensing conditions are defined by the data provider.

## Usage

Pretraining requires a user-supplied YAML configuration:

```bash
python -m ppi_ddg.train.pretrain_local_geometry \
  --config /path/to/pretrain.yaml \
  --device cuda:0
```

Downstream training also requires a user-supplied YAML configuration:

```bash
python -m ppi_ddg.train.train_ddg \
  --config /path/to/downstream.yaml \
  --device cuda:0
```

## Repository

```text
ppi_ddg/data/         Dataset and paired-graph loading
ppi_ddg/local_geom/   Geometric encoder, vector quantization, and paired model
ppi_ddg/train/        Pretraining and downstream entry points
environment.yml       Conda environment
pyproject.toml        Python package metadata
```
