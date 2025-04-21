import wandb
import os
import torch
from datetime import datetime

def setup_wandb(project_name, config):
    """
    Set up Weights & Biases for experiment tracking.
    
    Args:
        project_name: Name of the project in wandb
        config: Dictionary containing configuration parameters
    
    Returns:
        wandb run object
    """
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        config=config,
        name=f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        save_code=True
    )
    
    return run

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, metrics, checkpoint_dir):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Dictionary of validation metrics
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics
    }, checkpoint_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: PyTorch model to load the weights into
        optimizer: Optional PyTorch optimizer to load the state
    
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

def log_metrics_to_wandb(metrics, step, prefix=""):
    """
    Log metrics to wandb.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current step (epoch or batch)
        prefix: Optional prefix for metric names (e.g., "train/" or "val/")
    """
    log_dict = {f"{prefix}{k}": v for k, v in metrics.items()}
    wandb.log(log_dict, step=step)

def log_model_to_wandb(model_path, metadata=None):
    """
    Log model artifact to wandb.
    
    Args:
        model_path: Path to the model file
        metadata: Optional dictionary of metadata for the model
    
    Returns:
        wandb Artifact object
    """
    # Create an artifact
    artifact = wandb.Artifact(
        name=f"model-{wandb.run.id}",
        type="model",
        metadata=metadata
    )
    
    # Add the model file to the artifact
    artifact.add_file(model_path)
    
    # Log the artifact
    wandb.log_artifact(artifact)
    
    return artifact

def log_plots_to_wandb(plot_paths, step=None):
    """
    Log plots to wandb.
    
    Args:
        plot_paths: Dictionary mapping plot names to file paths
        step: Optional step for logging
    """
    for name, path in plot_paths.items():
        wandb.log({name: wandb.Image(path)}, step=step)

def finish_wandb_run():
    """Finish the current wandb run."""
    wandb.finish()