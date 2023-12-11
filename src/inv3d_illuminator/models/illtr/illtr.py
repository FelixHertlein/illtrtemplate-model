import itertools
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid

from inv3d_util.misc import check_tensor

from ..vgg_pretrained_loss import VGGPerceptualLoss
from .core.IllTr import IllTr
from .core.inference_ill import rec_ill
from .util import image2patches, load_model, patches2images, process_patches


class LitIllTr(pl.LightningModule):
    dataset_options: dict[str, Any] = {}
    train_options: dict[str, Any] = {
        "batch_size": 24,
        "max_epochs": 300,
        "early_stopping_patience": 25,
    }

    def __init__(self):  # , pretrained: bool = False
        super().__init__()
        self.save_hyperparameters()

        self.model = IllTr()
        self.vgg = VGGPerceptualLoss()
        self.alpha = 1e-5
        self.patch_batch_size = 32
        self.patch_size = 128
        self.overlap = 16

        # if pretrained:
        #    load_model(self.model, "/workspaces/inv3d-illuminator/models/illtr.pth")

    def forward(self, data, **kwargs):
        image = data["input"]["image"]

        patches_in = image2patches(
            image, patch_res=self.patch_size, overlap=self.overlap
        )

        patches_out = process_patches(
            self.model, patches_in, batch_size=self.patch_batch_size
        )

        image_out = patches2images(
            patches_out, height=image.shape[2], width=image.shape[3]
        )

        return image_out

    def training_step(self, batch, batch_idx, split: str = "train"):
        image_patches = batch["input"]["image_patch"]
        albedo_patches = batch["train"]["albedo_patch"]

        out = self.model(image_patches)

        l1_loss = F.l1_loss(out, albedo_patches)
        vgg_loss = self.vgg(out, albedo_patches)
        total_loss = l1_loss + self.alpha * vgg_loss

        self.log(f"{split}/l1_loss", l1_loss)
        self.log(f"{split}/vgg_loss", vgg_loss)
        self.log(f"{split}/total_loss", total_loss)

        if split == "val" and batch_idx == 0:
            self.log_images(image_patches, out, albedo_patches, tag="val/patches")

        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.training_step(batch=batch, batch_idx=batch_idx, split="val")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/total_loss",
        }

    def log_images(
        self,
        images_input: torch.Tensor,
        images_output: torch.Tensor,
        albedo: torch.Tensor,
        tag: str,
        limit_samples: Optional[int] = 8,
    ):
        check_tensor(images_input, "N 3 128 128")
        check_tensor(images_output, "N 3 128 128")
        check_tensor(albedo, "N 3 128 128")

        if limit_samples:
            images_input = images_input[:limit_samples]
            images_output = images_output[:limit_samples]
            albedo = albedo[:limit_samples]

        data = torch.stack(
            list(
                itertools.chain.from_iterable(zip(images_input, images_output, albedo))
            )
        )

        grid = make_grid(data, nrow=3)

        self.logger.experiment.add_image(tag, grid, self.current_epoch)  # type: ignore
