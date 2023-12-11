from pathlib import Path
from typing import *

import lpips
import numpy as np
import pandas as pd
import torch
from cachetools import cached
from einops import rearrange
from Levenshtein import distance
from PIL import Image
from pytesseract import pytesseract
from pytorch_msssim import ms_ssim
from tqdm import tqdm

from inv3d_util.misc import check_tensor, to_numpy_image
from inv3d_util.parallel import process_tasks


@cached({})
def get_lpips_metric(gpu: int):
    return lpips.LPIPS(net="alex").cuda(gpu)


def score_all(
    images_out: np.ndarray,
    images_true: np.ndarray,
    samples: list[Path],
    num_workers: int,
    gpu: int,
) -> pd.DataFrame:
    check_tensor(images_out, "n h w c", c=3)
    check_tensor(images_true, "n h w c", c=3)

    results: dict[str, list[Any]] = {"sample": samples}
    results.update(**_score_msssim(images_out, images_true, gpu))
    results.update(**_score_lpips(images_out, images_true, gpu))
    results.update(**_score_textual(images_out, images_true, num_workers=num_workers))

    return pd.DataFrame.from_dict(results)


@torch.no_grad()
def _score_msssim(
    images_out: np.ndarray, images_true: np.ndarray, gpu: int, batch_size: int = 64
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {"ms_ssim": []}
    for images_batch_out, images_batch_true in tqdm(
        list(
            zip(
                _split_given_size(images_out, batch_size),
                _split_given_size(images_true, batch_size),
            )
        ),
        desc="Scoring MS-SSIM",
    ):
        tensor_out = (
            rearrange(torch.from_numpy(images_batch_out), "n h w c -> n c h w")
            .cuda(gpu)
            .float()
            / 255
        )

        tensor_true = (
            rearrange(torch.from_numpy(images_batch_true), "n h w c -> n c h w")
            .cuda(gpu)
            .float()
            / 255
        )

        scores = ms_ssim(tensor_true, tensor_out, data_range=1, size_average=False)
        results["ms_ssim"].extend(scores.tolist())

    return results


@torch.no_grad()
def _score_lpips(
    images_out: np.ndarray, images_true: np.ndarray, gpu: int, batch_size: int = 16
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {"lpips": []}
    for images_batch_out, images_batch_true in tqdm(
        list(
            zip(
                _split_given_size(images_out, batch_size),
                _split_given_size(images_true, batch_size),
            )
        ),
        desc="Scoring LPIPS",
    ):
        tensor_out = (
            rearrange(torch.from_numpy(images_batch_out), "n h w c -> n c h w")
            .cuda(gpu)
            .float()
            / 255
        )

        tensor_true = (
            rearrange(torch.from_numpy(images_batch_true), "n h w c -> n c h w")
            .cuda(gpu)
            .float()
            / 255
        )

        scores = get_lpips_metric(gpu)(tensor_true, tensor_out).squeeze()
        results["lpips"].extend(scores.tolist())

    return results


def _score_textual(
    images_out: np.ndarray, images_true: np.ndarray, num_workers: int
) -> dict[str, list[float]]:
    print("Calculating textual scores")

    tasks = [
        {
            "image_true": image_true,
            "image_out": image_out,
        }
        for image_true, image_out in zip(images_true, images_out)
    ]

    all_results = process_tasks(
        _score_textual_task, tasks, num_workers=num_workers, use_indexes=True
    )

    return {
        metric_name: [all_results[k][metric_name] for k in sorted(all_results.keys())]
        for metric_name in all_results[0].keys()
    }


def _score_textual_task(task: Dict):
    true_text = pytesseract.image_to_string(
        Image.fromarray(task["image_true"]), lang="eng"
    )
    norm_text = pytesseract.image_to_string(
        Image.fromarray(task["image_out"]), lang="eng"
    )

    text_distance = distance(true_text, norm_text)

    return {"ed": text_distance, "cer": text_distance / len(true_text)}


def _split_given_size(a: np.ndarray, size: int) -> list[np.ndarray]:
    return np.split(a, np.arange(size, len(a), size))
