from cnnClassifier import logger
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    unzip_dir: Path
    source_url: str
    local_data_file: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    L_model_path: Path
    R_model_path: Path
    F_model_path: Path

    updated_L_model_path: Path
    updated_R_model_path: Path
    updated_F_model_path: Path

    params_model_lr_image_size: list
    params_model_f_image_size: list
    params_learning_rate: float
    params_classes: int
