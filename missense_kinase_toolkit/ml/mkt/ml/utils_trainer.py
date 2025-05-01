import logging
import os
import subprocess
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def create_slurm_script(
    fold_number: int,
    config_path: str,
    script_dir: str,
    job_name: str = "cv_trainer",
    partition: str = "componc_gpu",
    nodes: int = 1,
    ntasks: int = 1,
    mem_per_cpu: str = "64G",
    gpus_per_task: int = 1,
    time: str = "24:00:00",
    env_name: str | None = "mkt_ml_plus",
    account: str | None = None,
) -> str:
    """
    Creates a SLURM job script for training a specific fold.

    Parameters
    ----------
    fold_number : int
        Number of the fold to train
    config_path : str
        Path to the configuration file
    script_dir : str
        Directory to store slurm script files;
            will be saved in a subdirectory with the current date and time
    job_name : str
        Name of the job
    partition : str
        SLURM partition to use
    nodes : int
        Number of nodes to use (e.g., -N)
    ntasks : int
        Number of tasks to run (e.g., -n)
    mem_per_cpu : str
        Memory per CPU to request
    gpus_per_task : int
        Number of GPUs to request per task
    time : str
        Time limit for the job (format: HH:MM:SS)
    env_name : str, optional
        Name of the conda environment to activate
    account : str, optional
        SLURM account to charge the job to (if applicable)

    Returns
    -------
    str
        Path to the created SLURM script
    """
    fold_name = f"fold_{fold_number + 1}"

    # create a script filename
    script_path = os.path.join(script_dir, f"train_{fold_name}.sh")

    # build the SLURM script content
    content = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}_{fold_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --mem-per-cpu={mem_per_cpu}",
        f"#SBATCH --gpus-per-task={gpus_per_task}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={script_dir}/%x_{fold_name}_%j.out",
        f"#SBATCH --error={script_dir}/%x_{fold_name}_%j.err",
    ]

    # add account if specified
    if account:
        content.append(f"#SBATCH --account={account}")

    # add environment activation if needed
    if env_name:
        content.append("\nsource ~/.bashrc")
        content.append(f"mamba activate {env_name}")

    # change to the script fold directory for the purposes of checkpointing/plotting
    content.append(f"\ncd {script_dir}")

    # add the training command with fold argument
    content.append("\n# Run the training script")
    content.append(f"run_trainer --config {config_path} --fold {fold_number}")

    # Write the script to file
    with open(script_path, "w") as f:
        f.write("\n".join(content))

    # Make the script executable
    os.chmod(script_path, 0o755)

    return script_path


def submit_job(script_path: str) -> str:
    """
    Submit a job to SLURM and return the job ID.

    Parameters
    ----------
    script_path : str
        Path to the SLURM script to submit

    Returns
    -------
    str
        Job ID of the submitted job
    """
    try:
        # Submit the job and capture the output
        output = subprocess.check_output(
            ["sbatch", script_path], stderr=subprocess.STDOUT, text=True
        )

        # Extract the job ID (output is usually "Submitted batch job 12345")
        job_id = output.strip().split()[-1]
        logger.info(f"Job submitted successfully with ID: {job_id}")

        return job_id

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to submit job: {e.output}")
        raise


def batch_submit_folds(
    config_path: str,
    folds: int,
    slurm_params: dict[str, Any],
    outer_dir: str = "cv_trainer",
) -> dict[str, str]:
    """
    Batch submit multiple fold training jobs to SLURM.

    Parameters
    ----------
    config_path : str
        Path to the configuration file
    folds : int
        Number of folds to train
    slurm_params : dict[str, Any]
        Dictionary of parameters for SLURM job submission

    Returns
    -------
    Dict[str, str]
        Dictionary mapping fold names to their job IDs
    """
    # create datetime string for the script directory
    now = datetime.now()
    inner_dir = os.path.join(outer_dir, now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(inner_dir)

    job_ids = {}
    for fold in range(folds):
        fold_name = f"fold_{fold + 1}"

        fold_dir = os.path.join(inner_dir, fold_name)
        os.makedirs(fold_dir)
        # just store stdout/stderr in the fold_dir for now
        # for subdir in ["stdout", "stderr"]:
        #     os.makedirs(os.path.join(fold_dir, subdir))

        logger.info(f"Creating and submitting job for {fold_name}")

        # create a SLURM script for this fold
        script_path = create_slurm_script(
            fold_number=fold,
            config_path=config_path,
            script_dir=fold_dir,
            **slurm_params,
        )

        # submit the script to SLURM
        job_id = submit_job(script_path)
        job_ids[fold_name] = job_id

    return job_ids
