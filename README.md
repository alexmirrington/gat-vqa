# Graphgen

<p align="center">
  <a href="https://github.com/alexmirrington/graphgen/actions?query=branch%3Adevelop+workflow%3Atests">
    <img src="https://github.com/alexmirrington/graphgen/workflows/tests/badge.svg" alt="Tests status" />
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black" />
  </a>
</p>

----------------------

Generate graph representations of images and text for downstream multimodal
tasks like VQA.

## Getting started

### Preprequisites

Python: >=3.8
CUDA: >=10.2

Previous Python and CUDA versions may be compatible, but have not been tested.

### Install requirements

It is strongly recommended to install all requirements in a virtual environment,
as some required pytorch extensions are sensitive to both `torch` and system CUDA versions.

Make a clean virtual environment:

```Bash
python -m venv .virtualenvs/graphgen
source .virtualenvs/graphgen/bin/activate
```

Check your current system CUDA install version with `nvcc`:

```Bash
nvcc --version
```

If your system cuda version is the same as the latest CUDA version supported by `torch`,
you can simply run `make install` to install all dependencies.

If not, you should install `torch` and `torchvision` binaries that were built on the same CUDA version as your system CUDA version, either [prebuilt](https://pytorch.org/get-started/locally/) or from source. You can then run `make install` to install remaining dependencies.

For CUDA 11.0 support, you need to build `torch`, `torchvision`, `torchtext` and all [torch-geometric dependencies](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) from source until CUDA 11.0 is officially supported. Remaining requirements can be installed using the command `make requirements`.

### Set up Weights and Biases

Weights and Biases (`wandb`) is a superb tool for tracking machine learning projects, providing a myriad of time-saving features such as live metric logging, dataset/model artifact versioning and distributed hyperparameter tuning. Whilst the Python package is required for this code to run, you do not need to make an account if you prefer not to sign up. The first time you run `main.py`, simply choose whether to log in to an existing account, sign up for a `wandb` account or disable experiment tracking.

If you would like to track experiments and use dataset and model versioning capabilities, you can also log in anonymously like so:

```Bash
wandb login --anonymously
```

### Running a model

There are three types of jobs that `main.py` supports: `preprocess`, `train` and `test`. To run the code, simply provide a job type, a config file and optionally a `--sync` flag to tell `wandb` to sync your run to the cloud, _e.g._

```Bash
python main.py --config config/gqa.json --job train --sync
```

You can either download the GQA dataset yourself and run a preprocessing job to generate dependency parser outputs and other required preprocessed data, or use an existing preprocessed dataset specified by the `config.data.artifact` string.

## Testing

To run unit tests, simply run the `test` target from the `Makefile` using `make`:

```Bash
make test
```

This target wraps `pytest`, and logs code coverage results.
