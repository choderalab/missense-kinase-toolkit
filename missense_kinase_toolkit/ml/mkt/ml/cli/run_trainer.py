import argparse

# from mkt.ml.datasets.pkis2 import PKIS2Dataset
from mkt.ml.log_config import add_logging_flags, configure_logging
from mkt.ml.utils import set_seed

# from mkt.ml.models.pooling import CombinedPoolingModel
# from mkt.ml.trainer import create_dataloaders, train_model
from mkt.ml.factory import ExperimentFactory
from mkt.ml.trainer import run_pipeline_with_wandb

# from mkt.ml.utils import return_device

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Arguments to run the trainer.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )

    parser = add_logging_flags(parser)

    return parser.parse_args()


def main():
    """Main function to run the trainer."""

    args = parse_args()

    configure_logging()
    set_seed()

    dataset, model, dict_trainer_configs = ExperimentFactory(args.config)

    #TODO: if dataset dict, batch otherwise run
    # if isinstance(dataset.dataset_test, dict):
    run_pipeline_with_wandb(
        model = model,
        dataset_train = dataset.dataset_train,
        dataset_test = dataset.dataset_test,
        **dict_trainer_configs,
    )


if __name__ == "__main__":
    main()
