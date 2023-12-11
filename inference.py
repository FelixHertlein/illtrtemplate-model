import sys
from pathlib import Path

project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir / "src"))

import argparse
import os

import gdown
import torch
import tqdm
import yaml
from einops import rearrange

import inv3d_illuminator.models.model_factory as model_factory
from inv3d_util.load import load_image, save_image
from inv3d_util.path import list_dirs

model_sources = yaml.safe_load((project_dir / "models.yaml").read_text())
data_source = "https://drive.google.com/drive/folders/1Zoj9ydSp5TSztha1VULNFl_b9-GT6rmk?usp=sharing"


def inference(model_name: str, dataset: str, gpu: int):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # prepare model
    model_url = model_sources[model_name]
    model_dir = project_dir / "models" / model_name

    if not model_dir.is_dir():
        gdown.download_folder(model_url, output=model_dir.as_posix())

    # prepare data
    inv3d_real_unwarp_dir = Path("input/inv3d_real_unwarp")
    if dataset == "inv3d_real_unwarp" and not inv3d_real_unwarp_dir.is_dir():
        gdown.download_folder(data_source, output=inv3d_real_unwarp_dir.as_posix())

    # search checkpoint
    checkpoints = list(model_dir.rglob("checkpoint-epoch=*.ckpt"))
    if len(checkpoints) > 0:
        checkpoint = max(
            checkpoints, key=lambda x: int(x.stem.replace("=", "-").split("-")[2])
        )
    else:
        checkpoint = model_dir / "checkpoints" / "last.ckpt"

    if not checkpoint.is_file():
        print(f"ERROR! Could not find a checkpoint in directory '{model_dir}'")

    # load model
    model = model_factory.load_from_checkpoint(model_name.split("@")[0], checkpoint)
    model.to("cuda")
    model.eval()

    input_dir = project_dir / "input" / dataset
    output_dir = project_dir / "output" / f"{dataset} - {model_name}"
    output_dir.mkdir(exist_ok=True)

    image_paths = list(input_dir.rglob("norm_image.png"))

    for image_path in tqdm.tqdm(image_paths, "Remove illumination"):
        # prepare image
        image = load_image(image_path)
        image = rearrange(image, "h w c -> () c h w")
        image = image.astype("float32") / 255
        image = torch.from_numpy(image)
        image = image.to("cuda")

        model_kwargs = {"data": {"input": {"image": image}}}

        # prepare template
        if "template" in model_name:
            template_path = image_path.parent / "template.png"

            template = load_image(template_path)
            template = rearrange(template, "h w c -> () c h w")
            template = template.astype("float32") / 255
            template = torch.from_numpy(template)
            template = template.to("cuda")

            model_kwargs["data"]["input"]["template"] = template

        if "pad" in model_name:
            model_kwargs["template_patch_padding"] = int(model_name.split("=")[-1])

        if "full" in model_name:
            model_kwargs["template_patch_padding"] = None

        # inference model
        out_image = model(**model_kwargs)

        # post process image
        out_image = out_image.detach().cpu()
        out_image = out_image.numpy()
        out_image = (out_image * 255).astype("uint8")
        out_image = rearrange(out_image, "() c h w -> h w c")

        # export results
        sample_name = "_".join(image_path.relative_to(input_dir).parent.parts)
        save_image(
            output_dir / f"corrected_{sample_name}.png",
            out_image,
            override=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=list(model_sources.keys()),
        required=True,
        help="Select the model and the dataset used for training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(
            sorted(
                list(
                    set(
                        list(map(lambda x: x.name, list_dirs(project_dir / "input")))
                        + ["inv3d_real_unwarp"]
                    )
                )
            )
        ),
        required=True,
        help="Selects the inference dataset. All folders in the input directory can be selected.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        required=True,
        help="The index of the GPU to use for inference.",
    )
    args = parser.parse_args()

    inference(
        model_name=args.model,
        dataset=args.dataset,
        gpu=args.gpu,
    )
