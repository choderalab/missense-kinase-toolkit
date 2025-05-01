import argparse
import logging

from mkt.ml.factory import ExperimentFactory
from mkt.ml.log_config import add_logging_flags, configure_logging
from mkt.ml.trainer import run_pipeline_with_wandb
from mkt.ml.utils import set_seed
from mkt.ml.utils_trainer import batch_submit_folds, run_fold_locally

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Arguments to run the trainer.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    
    parser.add_argument(
        "--fold",
        action=int,
        nargs="?", # 0 or 1 - if not provided for CV, batch all folds
        help="Run a single fold of cross-validation.",
    )

    parser.add_argument(
        "--script_dir",
        type=str,
        default="slurm_scripts",
        help="Directory for SLURM output files.",
    )

    parser.add_argument(
        "--job_name",
        type=str,
        default="run_trainer",
        help="Name of the SLURM job.",
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="componc_gpu",
        help="SLURM partition to use (default: componc_gpu).",
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes to use for SLURM job (default: 1).",
    )

    parser.add_argument(
        "--ntasks",
        type=int,
        default=1,
        help="Number of tasks to use for SLURM job (default: 1).",
    )

    parser.add_argument(
        "--mem_per_cpu",
        type=str,
        default="64G",
        help="Memory per CPU for SLURM job (default: 64G).",
    )

    parser.add_argument(
        "--gpus_per_task",
        type=int,
        default=1,
        help="Number of GPUs per task for SLURM job (default: 1).",
    )
            
    parser.add_argument(
        "--time",
        type=str,
        default="24:00:00",
        help="Time limit for SLURM job (HH:MM:SS).",
    )
        
    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help="SLURM account to use.",
    )
        
    parser.add_argument(
        "--env_name",
        type=str,
        default="mkt_ml_plus",
        help="Conda environment name to activate in SLURM script.",
    )

    parser = add_logging_flags(parser)

    return parser.parse_args()


def main():
    """Main function to run the trainer."""

    args = parse_args()

    configure_logging()
    set_seed()

    experiment = ExperimentFactory(args.config)
    
    # cross-validation setup
    if "CV" in experiment.config["data"]["type"]:
        logger.info("Cross-validation setup detected...")

        # SLURM parameters from args
        slurm_params = {
            "script_dir": args.script_dir,
            "job_name": args.job_name,
            "partition": args.partition,
            "nodes": args.nodes,
            "ntasks": args.ntasks,
            "mem_per_cpu": args.mem_per_cpu,
            "gpus_per_task": args.gpus_per_task,
            "time": args.time,
            "account": args.account,
            "env_name": args.env_name,
        }

        if args.fold is None:
            logger.info("Submitting all folds as separate SLURM jobs...") 

            k_folds = experiment.config["data"]["configs"]["k_folds"]
        
            # submit jobs for all folds
            job_ids = batch_submit_folds(
                config_path=args.config,
                folds=k_folds,
                slurm_params=slurm_params,
            )
            
            logger.info(f"Submitted {len(job_ids)} jobs to SLURM")
            for fold, job_id in job_ids.items():
                logger.info(f"Fold {fold}: Job ID {job_id}")
            
        else:
            fold = args.fold
            logger.info(f"Running CV job fold-{fold+1}...")

            experiment.config["data"]["configs"]["fold_idx"] = fold
            dataset, model, dict_trainer_configs = experiment.build()

            # run a single fold locally
            run_pipeline_with_wandb(
                model=model,
                dataset_train=dataset.dataset_train[f"fold_{fold+1}"],
                dataset_test=dataset.dataset_test[f"fold_{fold+1}"],
                **dict_trainer_configs,
            )

    # train/test split
    else:
        logger.info("Train/test split detected...")

        dataset, model, dict_trainer_configs = experiment.build()

        run_pipeline_with_wandb(
            model=model,
            dataset_train=dataset.dataset_train,
            dataset_test=dataset.dataset_test,
            **dict_trainer_configs,
        )


if __name__ == "__main__":
    main()
