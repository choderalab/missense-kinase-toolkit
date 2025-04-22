import os
from datetime import datetime

import torch
import wandb


def setup_wandb(
    project_name: str, 
    config: dict,
) -> wandb.run:
    """Set up Weights & Biases for experiment tracking.

    Parameters:
    -----------
    project_name: str
        Name of the wandb project
    config: dict
        Configuration parameters for the experiment

    Returns:
    wandb.run
        Initialized wandb run object

    """
    run = wandb.init(
        project=project_name,
        config=config,
        name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        save_code=True,
    )

    return run


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    train_loss: float, 
    val_loss: float, 
    metrics: dict, 
    checkpoint_dir: str,
) -> str:
    """Save model checkpoint.

    Parameters:
    -----------
    model: torch.nn.Module
        PyTorch model to save
    optimizer: torch.optim.Optimizer
        PyTorch optimizer to save
    epoch: int
        Current epoch number
    train_loss: float
        Training loss
    val_loss: float
        Validation loss
    metrics: dict
        Dictionary of validation metrics
    checkpoint_dir: str
        Directory to save checkpoints

    Returns:
    --------
    str
        Path to saved checkpoint
    
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics,
        },
        checkpoint_path,
    )

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load model from checkpoint.

    Parameters:
    -----------
    checkpoint_path: str
        Path to the checkpoint file
    model: torch.nn.Module
        PyTorch model to load the weights into
    optimizer: torch.optim.Optimizer, optional
        PyTorch optimizer to load the state (if available)

    Returns:
    --------
    dict
        Dictionary containing checkpoint data
    
    """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def log_metrics_to_wandb(
    metrics: dict, 
    step: int, 
    prefix="",
):
    """Log metrics to wandb.

    Parameters:
    -----------
    metrics: dict
        Dictionary of metrics to log
    step: int
        Current step (epoch or batch)
    prefix: str, optional
        Optional prefix for metric names (e.g., "train/" or "val/")
    
    Returns:
    --------
    None
        Logs metrics to wandb

    """
    log_dict = {f"{prefix}{k}": v for k, v in metrics.items()}
    wandb.log(log_dict, step=step)


def log_model_to_wandb(model_path, metadata=None):
    """Log model artifact to wandb.

    Args:
        model_path: Path to the model file
        metadata: Optional dictionary of metadata for the model

    Returns:
        wandb Artifact object
    """
    # Create an artifact
    artifact = wandb.Artifact(
        name=f"model-{wandb.run.id}", type="model", metadata=metadata
    )

    # Add the model file to the artifact
    artifact.add_file(model_path)

    # Log the artifact
    wandb.log_artifact(artifact)

    return artifact


def log_plots_to_wandb(
    plot_paths: str,
    step: int | None = None,
) -> None:
    """Log plots to wandb.

    Parameters:
    -----------
    plot_paths: str
        Dictionary mapping plot names to file paths
    step: int, optional
        Optional step for logging (e.g., epoch or batch number)

    Returns:
    --------
    None
        Logs plots to wandb

    """
    for name, path in plot_paths.items():
        wandb.log({name: wandb.Image(path)}, step=step)


def finish_wandb_run():
    """Finish the current wandb run."""
    wandb.finish()
