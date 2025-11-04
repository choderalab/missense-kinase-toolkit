import logging
from dataclasses import dataclass
from os import path

import yaml
from mkt.ml.constants import (
    DataSet,
    DrugModel,
    KinaseInputType,
    KinaseModel,
    ModelType,
)

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

            self.seed = self.config["seed"]

        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")

    def build(self) -> tuple[object | None, object | None, dict | None]:
        """Create an experiment based on the configuration file.

        Returns
        -------
        dataset : object | None
            The dataset object.
        model : object | None
            The model object.
        dict_trainer_configs : dict | None
            The trainer configurations.

        """
        try:
            # validate the configuration file models
            model_drug_short = self.config["data"]["configs"]["model_drug"]
            self.config["data"]["configs"]["model_drug"] = getattr(
                DrugModel, self.config["data"]["configs"]["model_drug"]
            ).value
            model_kinase_short = self.config["data"]["configs"]["model_kinase"]
            self.config["data"]["configs"]["model_kinase"] = getattr(
                KinaseModel, self.config["data"]["configs"]["model_kinase"]
            ).value
            self.config["data"]["configs"]["col_kinase"] = getattr(
                KinaseInputType, self.config["data"]["configs"]["col_kinase"]
            ).value

            dataset_class = getattr(DataSet, self.config["data"]["type"])
            model_class = getattr(ModelType, self.config["model"]["type"])

            # instantiate the dataset class
            logger.info(f"Dataset configs:\n{self.config['data']['configs']}\n")
            dataset = dataset_class.value(**self.config["data"]["configs"])

            # instantiate the model class
            dict_model_configs = self.config["model"]["configs"]
            # make sure tokenizer matches model so get model names from dataset
            dict_model_configs["model_name_drug"] = dataset.model_drug
            dict_model_configs["model_name_kinase"] = dataset.model_kinase
            logger.info(f"Model configs:\n{dict_model_configs}\n")
            model = model_class.value(**dict_model_configs)

            dict_trainer_configs = self.config["trainer"]["configs"]
            # model_name
            split_type = ""
            if "col_kinase_split" in self.config["data"]["configs"]:
                split_type += (
                    self.config["data"]["configs"]["col_kinase_split"].upper()
                    + "_"
                    + "_".join(self.config["data"]["configs"]["list_kinase_split"])
                )
            if split_type == "":
                split_type = "CV"
            model_name = (
                self.config["data"]["type"].upper()
                + "-"
                + self.config["data"]["configs"]["col_kinase"].upper()
                + "-"
                + self.config["data"]["configs"]["col_drug"].upper()
                + "-"
                + model_drug_short.upper()
                + "-"
                + model_kinase_short.upper()
                + "-"
                + self.config["model"]["type"].upper()
                + "-"
                + split_type
            )
            dict_trainer_configs["model_name"] = model_name
            # otherwise treats as string if contains e
            dict_trainer_configs["learning_rate"] = float(
                dict_trainer_configs["learning_rate"]
            )
            logger.info(f"Trainer configs:\n{dict_trainer_configs}\n")

            return dataset, model, dict_trainer_configs

        except Exception as e:

            logger.error(f"Error creating experiment: {e}")

            return None, None, None
