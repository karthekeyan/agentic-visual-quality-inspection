"""Inference entry point for running the trained model on new casting images."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    raise NotImplementedError("Inference pipeline not yet implemented")


if __name__ == "__main__":
    main()
