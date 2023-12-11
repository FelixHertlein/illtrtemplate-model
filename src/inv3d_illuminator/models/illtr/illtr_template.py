import itertools
from functools import partial
from typing import Any, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as FN
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid

from inv3d_util.misc import check_tensor

from ..vgg_pretrained_loss import VGGPerceptualLoss
from .core.IllTr import (
    DecoderLayer,
    DePatchEmbed,
    EncoderLayer,
    Head,
    PatchEmbed,
    Tail,
    trunc_normal_,
)
from .util import (
    image2patches,
    patches2images,
    process_patches_template,
    template2patches,
)


class LitIllTrTemplate(pl.LightningModule):
    dataset_options: dict[str, Any] = {}
    train_options: dict[str, Any] = {
        "batch_size": 24,
        "max_epochs": 300,
        "early_stopping_patience": 25,
    }

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = IllTrTemplate()
        self.vgg = VGGPerceptualLoss()
        self.alpha = 1e-5
        self.patch_batch_size = 16
        self.patch_size = 128
        self.overlap = 16

    def forward(self, data, template_patch_padding: Optional[int], **kwargs):
        image = data["input"]["image"]
        template = data["input"]["template"]

        patches_in = image2patches(
            image, patch_res=self.patch_size, overlap=self.overlap
        )
        n, y, x, c, h, w = patches_in.shape

        if template_patch_padding is None:
            template_small = FN.interpolate(template, self.patch_size, mode="area")
            patches_template = repeat(
                template_small, "n c h w -> n y x c h w", y=y, x=x
            )
        else:
            patches_template = template2patches(
                template,
                patch_padding=template_patch_padding,
                patch_res=self.patch_size,
                overlap=self.overlap,
            )

        patches_out = process_patches_template(
            self.model, patches_in, patches_template, batch_size=self.patch_batch_size
        )

        image_out = patches2images(
            patches_out, height=image.shape[2], width=image.shape[3]
        )

        return image_out

    def training_step(self, batch, batch_idx, split: str = "train"):
        image_patches = batch["input"]["image_patch"]
        albedo_patches = batch["train"]["albedo_patch"]
        templates = batch["input"]["template"]

        out = self.model(image_patches, templates)

        l1_loss = F.l1_loss(out, albedo_patches)
        vgg_loss = self.vgg(out, albedo_patches)
        total_loss = l1_loss + self.alpha * vgg_loss

        self.log(f"{split}/l1_loss", l1_loss)
        self.log(f"{split}/vgg_loss", vgg_loss)
        self.log(f"{split}/total_loss", total_loss)

        if split == "val" and batch_idx == 0:
            self.log_images(
                templates, image_patches, out, albedo_patches, tag="val/patches"
            )

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
        image_small: torch.Tensor,
        image_patch: torch.Tensor,
        image_output: torch.Tensor,
        albedo: torch.Tensor,
        tag: str,
        limit_samples: Optional[int] = 8,
    ):
        check_tensor(image_small, "N 3 128 128")
        check_tensor(image_patch, "N 3 128 128")
        check_tensor(image_output, "N 3 128 128")
        check_tensor(albedo, "N 3 128 128")

        if limit_samples:
            image_small = image_small[:limit_samples]
            image_patch = image_patch[:limit_samples]
            image_output = image_output[:limit_samples]
            albedo = albedo[:limit_samples]

        data = torch.stack(
            list(
                itertools.chain.from_iterable(
                    zip(image_small, image_patch, image_output, albedo)
                )
            )
        )

        grid = make_grid(data, nrow=4)

        self.logger.experiment.add_image(tag, grid, self.current_epoch)  # type: ignore


class IllTrTemplate_Net(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=1,
        in_channels=3,
        mid_channels=16,
        num_classes=1000,
        depth=12,
        num_heads=8,
        ffn_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super(IllTrTemplate_Net, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = patch_size * patch_size * mid_channels
        self.head_patch = Head(in_channels, mid_channels)
        self.head_template = Head(in_channels, mid_channels)
        self.patch_embedding = PatchEmbed(
            patch_size=patch_size, in_channels=mid_channels
        )
        self.embed_dim = self.patch_embedding.dim
        if self.embed_dim % num_heads != 0:
            raise RuntimeError("Embedding dim must be devided by numbers of heads")

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 2 * ((128 // patch_size) ** 2), self.embed_dim)
        )
        self.task_embed = nn.Parameter(
            torch.zeros(6, 1, (128 // patch_size) ** 2, self.embed_dim)
        )

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.de_patch_embedding = DePatchEmbed(
            patch_size=patch_size, in_channels=mid_channels
        )
        # tail
        self.tail = Tail(int(mid_channels), in_channels)

        self.acf = nn.Hardtanh(0, 1)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, patches, templates):
        # patches: [24, 3, 128, 128]
        # templates: [24, 3, 128, 128]

        x = self.head_patch(patches)
        t = self.head_template(templates)

        # x: [24, 16, 128, 128]
        # t: [24, 16, 128, 128]

        x, ori_shape = self.patch_embedding(x)
        t, ori_shape = self.patch_embedding(t)

        # x: [24, 1024, 256]
        # t: [24, 1024, 256]

        x = torch.concat([x, t], dim=1)

        # x: [24, 2048, 256]

        x = x + self.pos_embed[:, : x.shape[1]]

        # x: [24, 1024, 256]

        for blk in self.encoder:
            x = blk(x)

        # x: [24, 2048, 256]

        x = x[:, :1024, :]  # cut off template part

        # x: [24, 1024, 256]

        for blk in self.decoder:
            x = blk(x, self.task_embed[0, :, : x.shape[1]])

        x = self.de_patch_embedding(x, ori_shape)

        x = self.tail(x)

        x = self.acf(x)
        return x


def IllTrTemplate(**kwargs):
    model = IllTrTemplate_Net(
        patch_size=4,
        depth=6,
        num_heads=8,
        ffn_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    return model
