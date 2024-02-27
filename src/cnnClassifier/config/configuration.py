import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig,  PrepareBaseModelConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            L_model_path=Path(config.L_model_path),
            R_model_path=Path(config.R_model_path),
            F_model_path=Path(config.F_model_path),

            updated_L_model_path=Path(config.updated_L_model_path),
            updated_R_model_path=Path(config.updated_R_model_path),
            updated_F_model_path=Path(config.updated_F_model_path),

            params_model_lr_image_size=self.params.MODEL_LR_IMAGE_SIZE,
            params_model_f_image_size=self.params.MODEL_F_IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
