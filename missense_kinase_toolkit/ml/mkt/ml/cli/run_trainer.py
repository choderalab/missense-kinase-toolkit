from mkt.ml.utils import return_device
from mkt.ml.datasets.pkis2 import PKIS2Dataset
from mkt.ml.models.pooling import CombinedPoolingModel
from mkt.ml.trainer import create_dataloaders, train_model
from mkt.ml.log_config import configure_logging
from mkt.ml.trainer import run_pipeline_with_wandb


def main():
    """Main function to run the trainer."""

    configure_logging()

    # run_pipeline_with_wandb()

    device = return_device()

    dataset_pkis2 = PKIS2Dataset()

    train_dataloader, test_dataloader = create_dataloaders(
        dataset_pkis2.dataset_train,
        dataset_pkis2.dataset_test
    )

    train_model(
        CombinedPoolingModel,
        train_dataloader,
        test_dataloader,
        device,
    )

if __name__ == "__main__":
    main()