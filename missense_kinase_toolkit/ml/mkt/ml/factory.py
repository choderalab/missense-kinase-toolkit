import logging
from dataclasses import dataclass

import yaml
from mkt.ml.constants import DICT_DATASET, DICT_MODELS
from mkt.ml.log_config import configure_logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentFactory:
    """Factory class for creating experiments based on a configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    """

    config_path: str

    def __post_init__(self):
        """Post-initialization method to load the configuration file."""
        try:
            # load the configuration file
            if not path.exists(self.config_path):
                raise FileNotFoundError(
                    f"Configuration file {self.config_path} not found."
                )
            if not self.config_path.endswith(".yaml"):
                raise ValueError("Configuration file must be a YAML file.")
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f.read())

            if not isinstance(self.config, dict):
                raise ValueError("Configuration file must be a dictionary.")
            if not all(key in self.config for key in ["data", "model", "trainer"]):
                raise ValueError(
                    "Configuration file must contain 'data', 'model' and 'trainer' keys."
                )

            self.dataset, self.model = self.create_experiment()

        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")

    def create_experiment(self) -> tuple[object | None, object | None]:
        """Create an experiment based on the configuration file.

        Returns
        -------
        dataset : object | None
            The dataset object.
        model : object | None
            The model object.

        """
        try:

            dataset_class = DICT_DATASET.get(self.config["data"]["source"])
            model_class = DICT_MODELS.get(self.config["model"]["model_type"])

            if dataset_class is None:
                raise ValueError(f"Dataset {self.config["data"]["source"]} not found.")
            if model_class is None:
                raise ValueError(
                    f"Model {self.config["model"]["model_type"]} not found."
                )

            # instantiate the dataset class
            dataset = dataset_class(**self.config["data"]["configs"])

            # instantiate the model class
            dict_model_configs = self.config["model"]["configs"]
            dict_model_configs["model_name_drug"] = dataset.model_drug
            dict_model_configs["model_name_kinase"] = dataset.model_kinase
            model = model_class(**dict_model_configs)

            return dataset, model

        except Exception as e:

            logger.error(f"Error creating experiment: {e}")

            return None, None
