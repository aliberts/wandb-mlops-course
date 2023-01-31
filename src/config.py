from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class WandBConfig:
    connect: bool = False
    entity: str = "aliberts"
    project: str = "mlops-course-001"
    run_name: Optional[str] = None
    tags: Optional[list] = None
    note: Optional[str] = None
    raw_data_at: str = "bdd_simple_1k"
    processed_data_at: str = "bdd_simple_1k_split"
    eda_table: str = "eda_table"


@dataclass
class DatasetConfig:
    name: str = "bdd1k"
    dir: Path = field(default=Path("dataset/BDD_SIMPLE_1k"))
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

    @property
    def images(self) -> Path:
        return self.dir / self.images_dir

    @property
    def labels(self) -> Path:
        return self.dir / self.labels_dir

    @property
    def license(self) -> Path:
        return self.dir / self.license_file


@dataclass
class MainConfig:
    wandb: WandBConfig = field(default_factory=WandBConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    debug: bool = False
