"""Main entrypoint for running, loading, saving and evaluating models."""


import argparse
import sys

import torch
from termcolor import colored


def main(config):
    """Run a model according to the config parameters."""
    # Print environment info
    print(colored("Environment:", attrs=["bold"]))
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(device) if cuda else 'CPU'}")

    print(config)


def parse_args():
    """Parse `sys.argv` and place parsed arguments in a config object."""
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--dataset", type=str, required=True, help="The dataset to use."
    )
    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    main(parse_args())
