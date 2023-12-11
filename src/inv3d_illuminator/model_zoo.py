import cv2
import numpy as np

cv2.setNumThreads(0)

import inspect
import os
import re
import resource
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from inv3d_util.load import load_image, save_image
from inv3d_util.mapping import apply_map_torch
from inv3d_util.misc import to_numpy_image, to_numpy_map
from inv3d_util.path import *
from inv3d_util.path import check_dir
from inv3d_util.visualization import visualize_bm, visualize_image

from .datasets.dataset_factory import DatasetFactory, DatasetSplit
from .models import model_factory
from .models.illtr.illtr import LitIllTr
from .score.score import *

seed_everything(seed=42, workers=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class ModelZoo:
    def __init__(self, root_dir: Path, sources_file: Path):
        self.root_dir = Path(root_dir)
        self.dataset_factory = DatasetFactory(sources_file)

        # torch.multiprocessing.set_sharing_strategy('file_system')
        resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))

    def list_models(self, verbose: bool = True) -> list[str]:
        models = list(model_factory.get_all_models())

        if verbose:
            print("\n".join(models))

        return models

    def list_datasets(self, verbose: bool = True) -> list[str]:
        datasets = list(self.dataset_factory.get_all_datasets())

        if verbose:
            print("\n".join(datasets))

        return datasets

    def list_trained_models(self, verbose: bool = True) -> list[str]:
        trained_models = [d.stem for d in list_dirs(self.root_dir)]

        if verbose:
            print("\n".join(trained_models))

        return trained_models

    def load_model(self, name: str) -> LightningModule:
        if name == "illtr@dirc@original":
            return LitIllTr(pretrained=True)

        run_config = RunConfig.from_str(name)

        checkpoints = list((self.root_dir / name).rglob("checkpoint-epoch=*.ckpt"))
        if len(checkpoints) > 0:
            checkpoint = max(
                checkpoints, key=lambda x: int(x.stem.replace("=", "-").split("-")[2])
            )
        else:
            checkpoint = self.root_dir / name / "checkpoints" / "last.ckpt"

        print(f"Loading checkpoint: {str(checkpoint.resolve())}")

        return model_factory.load_from_checkpoint(run_config.model, checkpoint)

    def delete_model(self, name: str):
        model_dir = self.root_dir / name
        if model_dir.is_dir():
            shutil.rmtree(str(model_dir))

    def train_model(
        self,
        name: str,
        gpu: int,
        num_workers: int,
        fast_dev_run: bool = False,
        model_kwargs: Optional[Dict] = None,
        resume: bool = False,
        dataset_kwargs: Optional[dict[str, Any]] = None,
    ):
        run_config = RunConfig.from_str(name)

        assert run_config.dataset is not None

        output_dir = check_dir(self.root_dir / name, exist=resume)
        output_dir.mkdir(exist_ok=True)

        # create model
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model = model_factory.create_new(run_config.model, **model_kwargs)

        if not resume:
            shutil.copyfile(
                str(inspect.getsourcefile(model.__class__)), output_dir / "model.py"
            )

        # gather options
        train_options: dict[str, Any] = deepcopy(model.train_options)  # type: ignore
        batch_size = train_options.pop("batch_size")
        max_epochs = train_options.pop("max_epochs")
        patience = train_options.pop("early_stopping_patience", None)

        dataset_options: dict[str, Any] = model.dataset_options  # type: ignore
        dataset_options.update(dataset_kwargs if dataset_kwargs else {})

        # prepare gpu configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # create datasets
        train_dataset = self.dataset_factory.create(
            name=run_config.dataset,
            split=DatasetSplit.TRAIN,
            limit_samples=run_config.limit_samples,
            repeat_samples=run_config.repeat_samples,
            **dataset_options,
        )

        val_dataset = self.dataset_factory.create(
            name=run_config.dataset,
            split=DatasetSplit.VALIDATE,
            limit_samples=run_config.limit_samples,
            **dataset_options,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=False,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

        # update properties required for training scheduler
        model.epochs = max_epochs
        model.steps_per_epoch = len(train_loader)  # type: ignore

        # create callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                monitor="val/total_loss",
                save_top_k=1,
                mode="min",
                filename="checkpoint-epoch={epoch:002d}-val_total_loss={val/total_loss:.4f}",
                auto_insert_metric_name=False,
                save_last=True,
                every_n_epochs=35,
            ),
            LearningRateMonitor(),
        ]

        if patience:
            callbacks.append(
                EarlyStopping(
                    monitor="val/total_loss",
                    patience=patience,
                    mode="min",
                    verbose=True,
                )
            )

        logger = TensorBoardLogger(output_dir, name="logs", version="")

        # create trainer
        trainer = Trainer(
            accelerator="gpu",
            devices=[0],
            logger=logger,
            fast_dev_run=fast_dev_run,
            callbacks=callbacks,
            max_epochs=max_epochs,
            **train_options,
        )

        resume_kwargs = (
            {"ckpt_path": output_dir / "checkpoints/last.ckpt"} if resume else {}
        )

        trainer.fit(model, train_loader, val_loader, **resume_kwargs)

    def evaluate(
        self,
        trained_model: str,
        dataset: str,
        ground_truth_dir: str,
        num_workers: int,
        gpu: int,
    ):
        ground_truth_dir = check_dir(ground_truth_dir)
        input_dir = check_dir(self.root_dir / trained_model / "inference" / dataset)

        eval_dir = check_dir(self.root_dir / trained_model) / "eval" / dataset
        eval_dir.mkdir(parents=True)

        samples = list(input_dir.rglob("ill_image.png"))
        gt_files = [
            ground_truth_dir / sample.with_name("true_image.png").relative_to(input_dir)
            for sample in samples
        ]

        assert len(samples) == len(list(Path(ground_truth_dir).rglob("true_image.png")))

        print("Loading images")
        images_out = np.stack([load_image(sample) for sample in samples])
        images_true = np.stack([load_image(gt_file) for gt_file in gt_files])

        df = score_all(images_out, images_true, samples, num_workers, gpu)
        df.to_csv(eval_dir / "results.csv", index=False)

        self.show_results(trained_model, dataset)

    def inference(
        self,
        trained_model: str,
        dataset: str,
        gpu: int,
        num_workers: int,
        template_patch_padding: Optional[int],
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        inference_dir = check_dir(self.root_dir / trained_model) / "inference" / dataset
        inference_dir.mkdir(parents=True, exist_ok=True)

        model = self.load_model(trained_model)
        model.cuda()
        model.eval()

        dataset_options: dict[str, Any] = model.dataset_options  # type: ignore

        inference_dataset = self.dataset_factory.create(
            name=dataset, split=DatasetSplit.TEST, **dataset_options
        )

        inference_loader = DataLoader(
            inference_dataset,
            batch_size=1,  # batch size 1 is required due to changing aspect ratios!
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )

        with tqdm(total=len(inference_loader), desc="Inference samples") as progress:
            for batch in inference_loader:
                input_data = {
                    key: value.cuda() for key, value in batch["input"].items()
                }

                input_data = {"input": input_data}
                out = (
                    model(input_data, template_patch_padding=template_patch_padding)
                    .detach()
                    .cpu()
                )

                [sample] = batch["sample"]
                sample = Path(sample)

                output_dir = inference_dir / sample.parent.name / sample.stem
                output_dir.mkdir(parents=True, exist_ok=True)
                save_image(
                    output_dir / "ill_image.png", to_numpy_image(out), override=True
                )
                save_image(
                    output_dir / "orig_image.png",
                    to_numpy_image(batch["input"]["image"]),
                    override=True,
                )

                if "eval" in batch:
                    if "albedo" in batch["eval"]:
                        save_image(
                            output_dir / "true_image.png",
                            to_numpy_image(batch["eval"]["albedo"]),
                            override=True,
                        )

                progress.update(1)

    def show_results(self, name: str, dataset_name: str):
        eval_dir = self.root_dir / name / "eval" / dataset_name

        df = pd.read_csv(eval_dir / "results.csv")

        print(f"Results for model '{name}' on '{dataset_name}':")
        print(df.drop(columns=["sample"]).mean())

        return df

    def show_all_results(self, evaluation: Optional[str] = None, group: bool = True):
        dfs = []
        for file in self.root_dir.rglob("results.csv"):
            df = pd.read_csv(file)
            df["model"] = file.parts[-4]
            df["evaluation"] = file.parts[-2]
            dfs.append(df)

        if len(dfs) == 0:
            return pd.DataFrame()

        df = pd.concat(dfs)

        if evaluation is not None:
            df = df[df.evaluation == evaluation]

        if group:
            df = df.drop(columns=["sample"])
            df = df.groupby(["model", "evaluation"]).agg(["mean", "std"])

        return df


@dataclass
class RunConfig:
    model: str
    dataset: Optional[str]
    limit_samples: Optional[int]
    repeat_samples: Optional[int]
    version: Optional[str]

    @staticmethod
    def from_str(name: str) -> "RunConfig":
        pattern = r"^(?P<model>[^\s@]+)(@(?P<dataset>[^\s@\[\]]+)(\[(?P<limit_samples>\d+)(x(?P<repeat_samples>\d+))?\])?(@(?P<version>[^\s@]+))?)?$"
        match = re.match(pattern, name)

        if match is None:
            raise ValueError(f"Invalid model name! Could not parse '{name}'!")

        limit_samples = match.groupdict().get("limit_samples", None)
        limit_samples = None if limit_samples is None else int(limit_samples)

        repeat_samples = match.groupdict().get("repeat_samples", None)
        repeat_samples = None if repeat_samples is None else int(repeat_samples)

        return RunConfig(
            model=match.groupdict()["model"],
            dataset=match.groupdict().get("dataset", None),
            limit_samples=limit_samples,
            repeat_samples=repeat_samples,
            version=match.groupdict().get("version", None),
        )
