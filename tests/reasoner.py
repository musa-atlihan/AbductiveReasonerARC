import argparse
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from utils.reason import AbductiveReasoner
from utils.kernel import KernelD1K9, KernelD2K25
from utils.memory import MemoryLTVHR, MemoryLTVR, MemoryLTVHR2N


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="File name from the dataset.", type=str)
    args = parser.parse_args()

    config = {
        "dataset": "data/dataset",
        "observation_walk_directions": ["btlr"],
        "prediction_walk_directions": ["btlr"],
        "color_frequency_threshold": 2,
        "frequency_threshold": 2,
        "combination_r_min": 1,
        "combination_r_max": 3,
        "num_observation_loops": 1,
        "num_inference_loops": 1,
        "visualize_prediction": False
    }

    #memory = None
    memory = MemoryLTVHR(max_len=6)

    kernel = KernelD1K9(memory)
    #kernel = KernelD2K25(memory)
    reasoner = AbductiveReasoner(kernel, config)
    reasoner.init_task(args.file)

    reasoner.observe()
    observations = reasoner.create_observation_df(is_sorted=True)
    observations.to_csv("data/outputs/observations.csv", index=False)

    reasoner.reason()
    explanations = reasoner.create_explanation_df(is_sorted=True)
    explanations.to_csv("data/outputs/explanations.csv", index=False)

    reasoner.explain_color()
    color_explanations = reasoner.create_explanation_df(is_sorted=True, is_color=True)
    color_explanations.to_csv("data/outputs/color_explanations.csv", index=False)

    prediction, scores = reasoner.predict(is_train=False, visualize=True)
    print("execution finished.")
