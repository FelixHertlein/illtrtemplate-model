from copy import deepcopy
from random import random, seed, shuffle
from typing import *

import torch
import torch.nn.functional as FN
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from unflatten import unflatten

from inv3d_util.load import load_json
from inv3d_util.path import check_dir, list_dirs

from .loaders import *


class Inv3DDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        template: str = "full",
        patch_size: int = 128,
        extra_features: Optional[List[str]] = None,
        limit_samples: Optional[int] = None,
        repeat_samples: Optional[int] = None,
        template_patch_padding: Optional[int] = None,
    ):
        self.source_dir = check_dir(source_dir)
        self.template = template
        self.extra_features = [] if extra_features is None else extra_features
        self.patch_size = patch_size
        self.template_patch_padding = template_patch_padding
        self.color_transform = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        )

        self.samples = list(sorted([s.name for s in list_dirs(self.source_dir)]))

        seed(42)
        self.unwarp_factors = [random() for _ in self.samples]

        if limit_samples:
            self.samples = self.samples[:limit_samples]

        if repeat_samples:
            self.samples = self.samples * repeat_samples

        if self.template_patch_padding is not None:
            print(
                f"Inv3D: Using template patches with a padding of {self.template_patch_padding}"
            )
        else:
            print("Inv3D: Using full template.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = self.source_dir / sample

        unwarp_factor = self.unwarp_factors[idx]

        template_scale_kwargs = (
            {}
            if self.template_patch_padding is not None
            else {"resolution": self.patch_size}
        )

        template = prepare_template(
            sample_dir / "flat_template.png",
            sample_dir / "flat_information_delta.png",
            sample_dir / "flat_text_mask.png",
            self.template,
            **template_scale_kwargs,
        )

        image = prepare_masked_image(
            sample_dir / "warped_document.png",
            sample_dir / "warped_UV.npz",
            sample_dir / "warped_BM.npz",
            unwarp_factor=unwarp_factor,
        )

        albedo = prepare_masked_image(
            sample_dir / "warped_albedo.png",
            sample_dir / "warped_UV.npz",
            sample_dir / "warped_BM.npz",
            unwarp_factor=unwarp_factor,
            bg_color=(255, 255, 255),
        )
        assert image.shape == albedo.shape

        # randomly crop single patch from images
        top, left, height, width = transforms.RandomCrop.get_params(
            image[0], (self.patch_size, self.patch_size)
        )

        if self.template_patch_padding is not None:
            _, h, w = image.shape

            template = torch.from_numpy(template).unsqueeze(0)
            template = FN.interpolate(template, (h, w), mode="area")
            template = FN.pad(template, [self.template_patch_padding] * 4, value=1)
            template = F.crop(
                img=template,
                top=top,
                left=left,
                height=height + 2 * self.template_patch_padding,
                width=width + 2 * self.template_patch_padding,
            )
            template = FN.interpolate(template, self.patch_size, mode="area")
            template = template.squeeze()

        image = F.crop(torch.from_numpy(image), top, left, height, width)
        albedo = F.crop(torch.from_numpy(albedo), top, left, height, width)

        image = (image.numpy() * 255).astype("uint8")
        image = rearrange(image, "c h w -> h w c")
        image = self.color_transform(image)
        image = rearrange(image, "h w c -> c h w")
        image = torch.from_numpy(image.astype("float32") / 255)

        # add jitter to image
        data = {}
        data["sample"] = str(sample)
        data["index"] = idx
        data["input.template"] = template
        data["input.image_patch"] = image
        data["train.albedo_patch"] = albedo

        return unflatten(data)


class Inv3DRealUnwarpDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        num_text_evals: int = 64,
        **kwargs,
    ):
        self.source_dir = check_dir(source_dir)
        self.num_text_evals = num_text_evals
        self.samples = list(
            sorted(
                [
                    sample.absolute()
                    for sample in list_dirs(self.source_dir, recursive=True)
                    if sample.name.startswith("warped")
                ]
            )
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        data = {}
        data["sample"] = str(sample_dir)
        data["index"] = idx

        template = load_image(sample_dir / "template.png")
        template = rearrange(template, "h w c -> c h w")
        template = template.astype("float32") / 255
        data["input.template"] = template

        image = load_image(sample_dir / "norm_image.png")
        image = rearrange(image, "h w c -> c h w")
        image = image.astype("float32") / 255
        data["input.image"] = image

        return unflatten(data)


class Inv3DTestDataset(Dataset):
    def __init__(
        self,
        source_dir: Path,
        unwarp_factor: float,
        limit_samples: Optional[int] = None,
    ):
        self.source_dir = check_dir(source_dir)
        self.unwarp_factor = unwarp_factor

        self.samples = list(sorted([s.name for s in list_dirs(self.source_dir)]))

        if limit_samples is not None:
            self.samples = self.samples[:limit_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_dir = self.source_dir / sample

        template = prepare_template(
            sample_dir / "flat_template.png",
            sample_dir / "flat_information_delta.png",
            sample_dir / "flat_text_mask.png",
            variation="full",
        )

        image = prepare_masked_image(
            sample_dir / "warped_document.png",
            sample_dir / "warped_UV.npz",
            sample_dir / "warped_BM.npz",
            unwarp_factor=self.unwarp_factor,
            resolution=template.shape[1:],
        )

        albedo = prepare_masked_image(
            sample_dir / "warped_albedo.png",
            sample_dir / "warped_UV.npz",
            sample_dir / "warped_BM.npz",
            unwarp_factor=self.unwarp_factor,
            bg_color=(255, 255, 255),
            resolution=template.shape[1:],
        )
        assert image.shape == albedo.shape

        data = {}
        data["sample"] = str(sample)
        data["index"] = idx
        data["input.template"] = template
        data["input.image"] = image
        data["eval.albedo"] = albedo

        return unflatten(data)
