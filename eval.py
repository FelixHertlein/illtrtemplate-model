import cv2

cv2.setNumThreads(0)

import argparse
import sys
from pathlib import Path

project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir / "src"))

import argparse

import gdown
import yaml

from inv3d_illuminator.model_zoo import ModelZoo

model_sources = yaml.safe_load((project_dir / "models.yaml").read_text())
data_source = "https://drive.google.com/drive/folders/1Zoj9ydSp5TSztha1VULNFl_b9-GT6rmk?usp=sharing"


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluation script")

    parser.add_argument(
        "--trained_model",
        type=str,
        choices=list(zoo.list_trained_models(verbose=False)),
        required=True,
        help="Select the model for evaluation.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["inv3d_real_unwarp"],
        required=True,
        help="Select the dataset to evaluate on.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="The index of the GPU to use for training.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="The number of workers as an integer.",
    )

    return parser


# Usage:
if __name__ == "__main__":
    zoo = ModelZoo(
        root_dir=project_dir / "models", sources_file=project_dir / "sources.yaml"
    )

    parser = create_arg_parser()
    args = parser.parse_args()

    if "pad" in args.trained_model:
        template_patch_padding = int(args.trained_model.split("=")[-1])
    else:
        template_patch_padding = None

    # prepare model
    model_url = model_sources[args.trained_model]
    model_dir = project_dir / "models" / args.trained_model

    if not model_dir.is_dir():
        gdown.download_folder(model_url, output=model_dir.as_posix())

    # prepare data
    inv3d_real_unwarp_dir = Path("input/inv3d_real_unwarp")
    if args.dataset == "inv3d_real_unwarp" and not inv3d_real_unwarp_dir.is_dir():
        gdown.download_folder(data_source, output=inv3d_real_unwarp_dir.as_posix())

    # inference samples
    zoo.inference(
        trained_model=args.trained_model,
        dataset=args.dataset,
        gpu=args.gpu,
        num_workers=args.num_workers,
        template_patch_padding=template_patch_padding,
    )

    # evaluate samples
    zoo.evaluate(
        trained_model=args.trained_model,
        dataset=args.dataset,
        gpu=args.gpu,
        num_workers=args.num_workers,
        ground_truth_dir=project_dir / f"input/inv3d_real_unwarp",
    )
