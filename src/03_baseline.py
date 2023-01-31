from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyrallis
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.data.block import DataBlock
from fastai.data.transforms import ColReader, ColSplitter
from fastai.torch_core import set_seed
from fastai.vision.augment import Resize, aug_transforms
from fastai.vision.data import ImageBlock, MaskBlock
from fastai.vision.learner import unet_learner
from torchvision.models.resnet import resnet18

import wandb
from src.config import MainConfig
from src.utils import (
    MIOU,
    BackgroundIOU,
    BicycleIOU,
    PersonIOU,
    RoadIOU,
    TrafficLightIOU,
    TrafficSignIOU,
    VehicleIOU,
    create_iou_table,
    get_predictions,
    label_func,
)


@pyrallis.wrap()
def main(cfg: MainConfig) -> None:
    set_seed(cfg.train.seed, reproducible=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        job_type="training",
        config=asdict(cfg.train),
    )

    processed_dataset_dir = download_data(cfg.wandb.processed_data_at)

    df = get_df(processed_dataset_dir, cfg.dataset.data_split_file, is_test=False)

    dls = get_data(
        df,
        cfg.dataset.classes,
        bs=cfg.train.batch_size,
        img_size=cfg.train.img_size,
        augment=cfg.train.augment,
    )

    metrics = [
        MIOU(),
        BackgroundIOU(),
        RoadIOU(),
        TrafficLightIOU(),
        TrafficSignIOU(),
        PersonIOU(),
        VehicleIOU(),
        BicycleIOU(),
    ]

    learn = unet_learner(dls, arch=resnet18, pretrained=cfg.train.pretrained, metrics=metrics)

    callbacks = [SaveModelCallback(monitor="miou"), WandbCallback(log_preds=False, log_model=True)]

    learn.fit_one_cycle(cfg.train.epochs, cfg.train.lr, cbs=callbacks)

    if cfg.train.log_preds:
        log_predictions(run, learn, cfg.dataset.classes)

    log_final_metrics(run, learn)

    run.finish()


def download_data(dataset_name):
    processed_data_at = wandb.use_artifact(f"{dataset_name}:latest")
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir


def get_df(processed_dataset_dir, data_split_file, is_test=False):
    df = pd.read_csv(processed_dataset_dir / data_split_file)

    if not is_test:
        df = df[df.Stage != "test"].reset_index(drop=True)
        df["is_valid"] = df.Stage == "valid"
    else:
        df = df[df.Stage == "test"].reset_index(drop=True)

    # assign paths
    df["image_fname"] = [processed_dataset_dir / f"images/{f}.jpg" for f in df.File_Name.values]
    df["label_fname"] = [label_func(f) for f in df.image_fname.values]
    return df


def get_data(df, classes, bs=4, img_size=(180, 320), augment=True):
    block = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=classes)),
        get_x=ColReader("image_fname"),
        get_y=ColReader("label_fname"),
        splitter=ColSplitter(),
        item_tfms=Resize(img_size),
        batch_tfms=aug_transforms() if augment else None,
    )
    return block.dataloaders(df, bs=bs)


def log_predictions(run, learn, classes):
    "Log a Table with model predictions"
    samples, outputs, predictions = get_predictions(learn)
    table = create_iou_table(samples, outputs, predictions, classes)
    run.log({"pred_table": table})


def log_final_metrics(run, learn):
    scores = learn.validate()
    metric_names = ["final_loss"] + [f"final_{x.name}" for x in learn.metrics]
    final_results = {metric_names[i]: scores[i] for i in range(len(scores))}
    for k, v in final_results.items():
        run.summary[k] = v


if __name__ == "__main__":
    main()
