from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        model_left = PrepareBaseModel(
            config=prepare_base_model_config, type='l')
        model_right = PrepareBaseModel(
            config=prepare_base_model_config, type='r')
        model_full = PrepareBaseModel(
            config=prepare_base_model_config, type='f')

        model_left.get_base_model()
        model_right.get_base_model()
        model_full.get_base_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e
