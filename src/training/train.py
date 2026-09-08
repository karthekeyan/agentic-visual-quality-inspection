"""Training entry point for the casting defect detection model."""

import argparse

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    raise NotImplementedError("Training loop not yet implemented")


if __name__ == "__main__":
    main()
