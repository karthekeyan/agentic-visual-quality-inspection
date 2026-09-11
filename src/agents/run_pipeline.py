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

    characterization = final_state["defect_characterization"]
    print(f"characterization.applicable:          {characterization.applicable}")
    print(f"characterization.category:            {characterization.category}")
    print(f"characterization.description:         {characterization.description}")
    print(f"characterization.heuristic_approximation: {characterization.heuristic_approximation}")
    if characterization.applicable:
        print(f"characterization.region_size:          {characterization.region_size}")
        print(f"characterization.position:             {characterization.position}")
        print(f"characterization.confidence_tier:      {characterization.confidence_tier}")

    root_cause = final_state["root_cause"]
    print(f"root_cause.applicable: {root_cause.applicable}")
    if root_cause.applicable:
        print(f"root_cause.model:      {root_cause.model}")
        print("root_cause.retrieved_entries:")
        for entry in root_cause.retrieved_entries:
            print(f"  - {entry['defect_type']} (distance={entry['distance']:.4f})")
        print(f"root_cause.explanation:\n{root_cause.explanation}")

    disposition = final_state["disposition"]
    print(f"disposition.decision:                 {disposition.decision}")
    print(f"disposition.reasoning:                {disposition.reasoning}")
    print(f"disposition.confidence_threshold_used: {disposition.confidence_threshold_used}")

    trend = final_state["trend"]
    print(f"trend.batch_id (SIMULATED):    {trend.batch_id}")
    print(f"trend.defect_rate:              {trend.defect_rate:.4f}")
    print(f"trend.scrap_rate:               {trend.scrap_rate:.4f}")
    print(f"trend.drift_flag:               {trend.drift_flag}")
    print(f"trend.sample_size:              {trend.sample_size}")
    print(f"trend.note:                     {trend.note}")

    print(f"agent_outputs: {sorted(final_state['agent_outputs'].keys())}")

    if args.out:
        from pathlib import Path

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_state["overlay_image"].save(out_path)
        print(f"overlay saved to: {out_path}")


if __name__ == "__main__":
    main()
