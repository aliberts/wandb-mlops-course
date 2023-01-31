from pathlib import Path

import pandas as pd
import pyrallis
from sklearn.model_selection import StratifiedGroupKFold

import wandb
from src.config import MainConfig


@pyrallis.wrap()
def main(cfg: MainConfig) -> None:

    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="data_split")
    raw_data_at = run.use_artifact(f"{cfg.dataset.raw_data_at}:latest")
    path = Path(raw_data_at.download())

    orig_eda_table = raw_data_at.get(cfg.wandb.eda_table)
    fnames = orig_eda_table.get_column("File_Name")
    groups = [s.split("-")[0] for s in fnames]
    y = orig_eda_table.get_column("bicycle")

    df = split_data(fnames, y, groups)

    print("data split:")
    print(df.Stage.value_counts())
    df.to_csv(cfg.dataset.data_split, index=False)

    processed_data_at = wandb.Artifact(cfg.dataset.processed_data_at, type="split_data")
    processed_data_at.add_file(cfg.dataset.data_split)
    processed_data_at.add_dir(path)
    data_split_table = wandb.Table(dataframe=df[["File_Name", "Stage"]])
    join_table = wandb.JoinedTable(orig_eda_table, data_split_table, "File_Name")
    processed_data_at.add(join_table, "eda_table_data_split")

    if cfg.wandb.connect:
        run.log_artifact(processed_data_at)

    run.finish()


def split_data(fnames, y, groups) -> pd.DataFrame:
    """
    Splits the file names and the target labels (fnames, y) into
    train (80%) / validation (10%) / test (10%) with no overlap by groups
    of similar images (to avoid data leakage) thanks to StratifiedGroupKFold.
    """
    df = pd.DataFrame()
    df["File_Name"] = fnames
    df["fold"] = -1

    cv = StratifiedGroupKFold(n_splits=10)
    for i, (_, test_idxs) in enumerate(cv.split(fnames, y, groups)):
        df.loc[test_idxs, ["fold"]] = i

    df["Stage"] = "train"  # 80%
    df.loc[df.fold == 0, ["Stage"]] = "test"  # 10%
    df.loc[df.fold == 1, ["Stage"]] = "valid"  # 10%
    del df["fold"]

    return df


if __name__ == "__main__":
    main()
