import numpy as np
import torch
import torch.nn.functional as FN
import torch.nn.functional as F
from einops import rearrange

from inv3d_util.misc import check_tensor

from .core.inference_ill import composePatch, illCorrection, padCropImg


def load_model(model, path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path)

    pretrained_dict = {
        k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict
    }

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def image2patches(image: torch.Tensor, patch_res: int, overlap: int):
    check_tensor(image, "n c h w")

    shift = patch_res - overlap

    _, _, H, W = image.shape

    padH = (int((H - patch_res) / (shift) + 1) * (shift) + patch_res) - H
    padW = (int((W - patch_res) / (shift) + 1) * (shift) + patch_res) - W

    image_pad = F.pad(image, (0, padW, 0, padH))

    patches = image_pad.unfold(-2, patch_res, step=shift)
    patches = patches.unfold(-2, patch_res, step=shift)
    patches = rearrange(patches, "n c y x h w -> n y x h w c")

    patches = patches.clone()  # important before assigning

    # overwrite last row
    row = image[:, :, -patch_res:, :].unfold(-1, patch_res, step=shift)
    row = rearrange(row, "n c h a w -> n a h w c")
    patches[:, -1, : row.shape[1], ...] = row

    # overwrite last column
    col = image[:, :, :, -patch_res:].unfold(-2, patch_res, step=shift)
    col = rearrange(col, "n c a h w -> n a w h c")
    patches[:, : col.shape[1], -1, :, :, :] = col

    # overwrite corner case
    corner = image[:, :, -patch_res:, -patch_res:]
    corner = rearrange(corner, "n c h w -> n h w c")
    patches[:, -1, -1, ...] = corner

    return rearrange(patches, "n y x h w c -> n y x c h w")


def template2patches(
    template: torch.Tensor, patch_padding: int, patch_res: int, overlap: int
) -> torch.Tensor:
    check_tensor(template, "n c h w")

    template = FN.pad(template, [patch_padding] * 4, value=1)

    patches = image2patches(
        template,
        patch_res=patch_res + 2 * patch_padding,
        overlap=overlap + 2 * patch_padding,
    )

    if patch_padding != 0:
        n, y, x, c, h, w = patches.shape
        patches = rearrange(patches, "n y x c h w -> (n y x) c h w")
        patches = FN.interpolate(patches, patch_res, mode="area")
        patches = rearrange(patches, "(n y x) c h w -> n y x c h w", n=n, y=y, x=x)

    return patches


def patches2images_average(
    patches: torch.Tensor,
    height: int,
    width: int,
    patch_res: int = 128,
    overlap: int = 16,
) -> torch.Tensor:
    dims = check_tensor(patches, "n y x c h w", c=3, h=patch_res, w=patch_res)
    n = dims["n"]

    patches = rearrange(patches, "n y x c h w -> n y x h w c")

    # coordinate grid with height x width
    image_coords = torch.stack(
        torch.meshgrid(torch.arange(height), torch.arange(width))
    )

    # cut out patches from the coordinate grid and flatten coordinates
    patches_coords = image2patches(
        image_coords.unsqueeze(0), patch_res=patch_res, overlap=overlap
    )
    patches_coords = rearrange(patches_coords, "n y x c h w -> n (y x h w) c")
    patches_coords = patches_coords[:, :, 0] * width + patches_coords[:, :, 1]
    patches_coords = patches_coords + (torch.arange(n) * height * width).unsqueeze(-1)
    patches_coords = patches_coords.reshape(-1)
    patches_coords = patches_coords.unsqueeze(-1).repeat(1, 3)

    # flatten patches
    patches = rearrange(patches, "n y x h w c -> (n y x h w) c")

    # calculate sum and count image
    image_sums = torch.zeros((n * height * width, 3), dtype=torch.float).scatter_add(
        0, patches_coords, patches.float()
    )
    image_counts = torch.zeros((n * height * width, 3), dtype=torch.float).scatter_add(
        0, patches_coords, torch.ones_like(patches).float()
    )

    # finalize image
    image_sums = image_sums.reshape(n, height, width, 3)
    image_counts = image_counts.reshape(n, height, width, 3)

    result = (image_sums / image_counts).type(patches.dtype)

    return rearrange(result, "n h w c -> n c h w")


def patches2images(
    patches: torch.Tensor,
    height: int,
    width: int,
    patch_res: int = 128,
    overlap: int = 16,
) -> torch.Tensor:
    dims = check_tensor(patches, "n y x c h w", c=3, h=patch_res, w=patch_res)

    assert overlap % 2 == 0, "overlap needs to be divisible by two"

    image = torch.zeros((dims["n"], dims["c"], height, width))
    patch_inner = patch_res - overlap

    margin = int(overlap / 2)

    for y in range(dims["y"]):
        for x in range(dims["x"]):
            offset_top = -margin if y == 0 else 0
            offset_left = -margin if x == 0 else 0
            offset_bottom = margin if y == dims["y"] - 1 else 0
            offset_right = margin if x == dims["x"] - 1 else 0

            base_y = margin + y * patch_inner
            base_x = margin + x * patch_inner

            y_min = base_y + offset_top
            y_max = min(base_y + patch_inner + offset_bottom, height)

            x_min = base_x + offset_left
            x_max = min(base_x + patch_inner + offset_right, width)

            patch_height = y_max - y_min
            patch_width = x_max - x_min

            image[
                :,
                :,
                y_min:y_max,
                x_min:x_max,
            ] = patches[
                :,
                y,
                x,
                :,
                margin + offset_top : margin + offset_top + patch_height,
                margin + offset_left : margin + offset_left + patch_width,
            ]

    return image


def process_patches(
    model: torch.nn.Module, patches: torch.Tensor, batch_size: int
) -> torch.Tensor:
    check_tensor(patches, "n y x c h h", c=3)

    n, y, x, c, h, w = patches.shape

    patches = patches.reshape(-1, c, h, w)

    out = torch.concat(
        [model(batch).detach().cpu() for batch in torch.split(patches, batch_size)]
    )

    return out.reshape(n, y, x, c, h, w)


def process_patches_template(
    model: torch.nn.Module,
    patches: torch.Tensor,
    template_patches: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    check_tensor(patches, "n y x c h h", c=3)
    check_tensor(template_patches, "n y x c h h", c=3)

    n, y, x, c, h, w = patches.shape

    patches = patches.reshape(-1, c, h, w)
    template_patches = template_patches.reshape(-1, c, h, w)

    out = torch.concat(
        [
            model(image_batch, template_batch).detach().cpu()
            for image_batch, template_batch in zip(
                torch.split(patches, batch_size),
                torch.split(template_patches, batch_size),
            )
        ]
    )

    return out.reshape(n, y, x, c, h, w)


def illuminate(
    net: torch.nn.Module, img: np.ndarray, device: torch.device
) -> np.ndarray:
    totalPatch, padH, padW = padCropImg(img)

    totalResults = illCorrection(net, totalPatch, device)

    resImg = composePatch(totalResults, padH, padW, img)

    return resImg
