import argparse

# from mkt.ml.models.pooling import CombinedPoolingModel
# from mkt.ml.trainer import create_dataloaders, train_model
from mkt.ml.factory import ExperimentFactory

# from mkt.ml.datasets.pkis2 import PKIS2Dataset
from mkt.ml.log_config import add_logging_flags, configure_logging
from mkt.ml.trainer import run_pipeline_with_wandb
from mkt.ml.utils import set_seed

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

    experiment = ExperimentFactory(args.config)
    set_seed(experiment.seed)

    dataset, model = experiment.build()

    # device = return_device()

    # dataset_pkis2 = PKIS2Dataset()
    # dataloader generates a batch of data that is a list of dictionaries - recreate here
    # list_test = [i for idx, i in enumerate(dataset_pkis2.dataset_train) if idx in range(0, 5)]

    # dataloader_train, dataloader_test = create_dataloaders(
    #     dataset_pkis2.dataset_train, dataset_pkis2.dataset_test
    # )

    # for batch in dataloader_train:
    #     print(batch)

    # model = CombinedPoolingModel(
    #     model_name_drug=dataset_pkis2.model_drug,
    #     model_name_kinase=dataset_pkis2.model_kinase,
    # )

    # model.to(device)

    # train_model(
    #     model,
    #     dataloader_train,
    #     dataloader_test,
    # )

    run_pipeline_with_wandb()


if __name__ == "__main__":
    main()
