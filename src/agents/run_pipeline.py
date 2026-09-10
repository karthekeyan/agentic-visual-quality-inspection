"""CLI: run the agent graph on a single image and print the resulting state.

    python -m src.agents.run_pipeline --image path/to/image.jpeg

For manually verifying the graph executes end-to-end as agents are added
in later phases.
"""

import argparse

import numpy as np

from src.agents.graph import get_graph


def main():
    parser = argparse.ArgumentParser(description="Run the agent pipeline on a single image.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--out", default=None, help="Optional path to save the Grad-CAM overlay PNG.")
    args = parser.parse_args()

    graph = get_graph()
    final_state = graph.invoke({"image_path": args.image})

    print(f"image_path:    {final_state['image_path']}")
    print(f"label:         {final_state['label']}")
    print(f"confidence:    {final_state['confidence']:.4f}")
    print(f"raw_logits:    {np.array2string(final_state['raw_logits'], precision=4)}")
    print(f"heatmap:       shape={final_state['heatmap'].shape}")
    print(f"overlay_image: {final_state['overlay_image'].size} {final_state['overlay_image'].mode}")
    print(f"agent_outputs: {sorted(final_state['agent_outputs'].keys())}")

    if args.out:
        from pathlib import Path

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_state["overlay_image"].save(out_path)
        print(f"overlay saved to: {out_path}")


if __name__ == "__main__":
    main()
