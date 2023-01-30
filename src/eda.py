import numpy as np
import pyrallis
from fastai.data.transforms import get_image_files
from PIL import Image
from tqdm import tqdm

import wandb
from src.config import MainConfig


@pyrallis.wrap()
def main(cfg: MainConfig) -> None:

    if cfg.wandb.connect:
        run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="upload")

    raw_data_at = wandb.Artifact(cfg.wandb.raw_data_at, type="raw_data")
    raw_data_at.add_file(cfg.dataset.license, name=cfg.dataset.license_file.name)
    raw_data_at.add_dir(cfg.dataset.images, name=cfg.dataset.images_dir.name)
    raw_data_at.add_dir(cfg.dataset.labels, name=cfg.dataset.labels_dir.name)

    image_files = get_image_files(cfg.dataset.images, recurse=False)

    if cfg.debug:
        image_files = image_files[:10]  # Sample a subset if debug

    table = create_table(image_files, cfg.dataset.classes)
    raw_data_at.add(table, "eda_table")

    if cfg.wandb.connect:
        run.log_artifact(raw_data_at)
        run.finish()


def create_table(image_files, class_labels):
    "Create a table with the dataset"
    labels = [str(class_labels[_lab]) for _lab in list(class_labels)]
    table = wandb.Table(columns=["File_Name", "P1", "P2", "Images", "Dataset"] + labels)

    for image_file in tqdm(image_files):
        image = Image.open(image_file)
        mask_data = np.array(Image.open(label_func(image_file)))
        class_in_image = get_classes_per_image(mask_data, class_labels)
        table.add_data(
            image_file.stem,
            image_file.stem.split("-")[0],
            image_file.stem.split("-")[1],
            wandb.Image(
                image,
                masks={
                    "predictions": {
                        "mask_data": mask_data,
                        "class_labels": class_labels,
                    }
                },
            ),
            "bdd1k",
            *[class_in_image[_lab] for _lab in labels],
        )

    return table


def label_func(fname):
    return (fname.parent.parent / "labels") / f"{fname.stem}_mask.png"


def get_classes_per_image(mask_data, class_labels):
    unique = list(np.unique(mask_data))
    result_dict = {}
    for _class in class_labels.keys():
        result_dict[class_labels[_class]] = int(_class in unique)
    return result_dict


if __name__ == "__main__":
    main()
