from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class WandBConfig:
    entity: str = "aliberts"
    project: str = "mlops-course-001"
    run_name: Optional[str] = None
    tags: Optional[list] = None
    note: Optional[str] = None
    eda_table: str = "eda_table"


@dataclass
class DatasetConfig:
    name: str = "bdd1k"
    raw_data_at: str = "bdd_simple_1k"
    processed_data_at: str = "bdd_simple_1k_split"
    dir: Path = field(default=Path("artifacts/bdd_simple_1k:v0"))
    classes: dict = field(
        default_factory=lambda: {
            i: c
            for i, c in enumerate(
                [
                    "background",
                    "road",
                    "traffic light",
                    "traffic sign",
                    "person",
                    "vehicle",
                    "bicycle",
                ]
            )
        }
    )
    images_dir: Path = field(default=Path("images"))
    labels_dir: Path = field(default=Path("labels"))
    license_file: Path = field(default=Path("LICENSE.txt"))
    data_split_file: Path = field(default=Path("data_split.csv"))

    @property
    def images(self) -> Path:
        return self.dir / self.images_dir

    @property
    def labels(self) -> Path:
        return self.dir / self.labels_dir

    @property
    def license(self) -> Path:
        return self.dir / self.license_file

    @property
    def data_split(self) -> Path:
        return self.dir / self.data_split_file


@dataclass
class TrainConfig:
    framework: str = "fastai"
    img_size: tuple[int] = (180, 320)
    batch_size: int = 8
    augment: bool = True  # use data augmentation
    epochs: int = 10
    lr: float = 2e-3
    arch: str = "resnet18"
    pretrained: bool = True  # whether to use pretrained encoder
    seed: int = 42
    log_preds: bool = True


@dataclass
class MainConfig:
    wandb: WandBConfig = field(default_factory=WandBConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    debug: bool = False
