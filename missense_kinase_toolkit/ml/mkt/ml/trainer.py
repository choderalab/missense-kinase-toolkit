import heapq
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datasets import Dataset
from mkt.ml.utils import return_device
from mkt.ml.utils_wandb import (  # log_metrics_to_wandb,; log_model_to_wandb,; log_plots_to_wandb,; save_checkpoint,
    setup_wandb,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    bool_train_shuffle: bool = True,
):
    """Create PyTorch DataLoaders from HuggingFace datasets.

    Parameters
    ----------
    train_dataset: Dataset
        Training dataset
    test_dataset: Dataset
        Testing dataset
    batch_size: int
        Batch size for DataLoader
    bool_train_shuffle: bool
        Whether to shuffle the training dataset

    Returns
    -------
    train_dataloader: DataLoader
        DataLoader for training dataset
    test_dataloader: DataLoader
        DataLoader for testing dataset

    """

    def collate_fn(batch):
        smiles_input_ids = torch.tensor([item["smiles_input_ids"] for item in batch])
        smiles_attention_mask = torch.tensor(
            [item["smiles_attention_mask"] for item in batch]
        )
        klifs_input_ids = torch.tensor([item["klifs_input_ids"] for item in batch])
        klifs_attention_mask = torch.tensor(
            [item["klifs_attention_mask"] for item in batch]
        )
        labels = torch.tensor(
            [item["labels"] for item in batch], dtype=torch.float
        ).view(-1, 1)

        return {
            "smiles_input_ids": smiles_input_ids,
            "smiles_attention_mask": smiles_attention_mask,
            "klifs_input_ids": klifs_input_ids,
            "klifs_attention_mask": klifs_attention_mask,
            "labels": labels,
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool_train_shuffle,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_dataloader, test_dataloader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
):
    """Evaluate the model on the given dataloader.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate
    dataloader : DataLoader
        DataLoader for evaluation
    criterion : nn.Module
        Loss function
    device : str
        Device to run evaluation on

    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            smiles_input_ids = batch["smiles_input_ids"].to(device)
            smiles_attention_mask = batch["smiles_attention_mask"].to(device)
            klifs_input_ids = batch["klifs_input_ids"].to(device)
            klifs_attention_mask = batch["klifs_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_drug=smiles_input_ids,
                mask_drug=smiles_attention_mask,
                input_kinase=klifs_input_ids,
                mask_kinase=klifs_attention_mask,
            )

            # Calculate loss
            loss = criterion(outputs, labels)

            # Accumulate loss
            total_val_loss += loss.item()

            # Store predictions and labels for metrics
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(dataloader)

    # Calculate metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    return {
        "loss": avg_val_loss,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": all_preds,
        "labels": all_labels,
    }


def create_prediction_plot(
    labels: list,
    preds: list,
    title: str,
    plot_dir: str,
    epoch: int | None = None,
    step: int | None = None,
) -> str:
    """Create a scatter plot of predictions vs actual values.

    Parameters
    ----------
    labels : list
        True values
    preds : list
        Predicted values
    title : str
        Plot title
    plot_dir : str
        Directory to save the plot
    epoch : int, optional
        Current epoch
    step : int, optional
        Current step

    Returns
    -------
    str
        Path to saved plot
    """
    subplot_title = title
    if epoch is not None:
        subplot_title += f" (Epoch {epoch+1}"
        if step is not None:
            subplot_title += f", Step {step}"
        subplot_title += ")"

    plt.figure(figsize=(10, 6))
    plt.scatter(labels, preds, alpha=0.5)
    plt.plot(
        [min(labels), max(labels)],
        [min(labels), max(labels)],
        "r--",
    )
    plt.xlabel("True standardized percent_displacement")
    plt.ylabel("Predicted standardized percent_displacement")
    plt.title(subplot_title)

    # Generate a different filename based on context
    if step is not None:
        plot_path = f"val_predictions_step_{step}.png"
    elif epoch is not None:
        plot_path = f"val_predictions_epoch_{epoch+1}.png"
    else:
        plot_path = "val_predictions.png"

    plt.savefig(os.path.join(plot_dir, plot_path))
    plt.close()

    return plot_path


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    checkpoint_dir: str,
    plot_dir: str,
    bool_wandb: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    percent_warmup: float,
    bool_clip_grad: bool,
    save_every: int,
    moving_avg_window: int,
    log_interval: int,
    validation_step_interval: int,
    best_models_to_keep: int,
):
    """Train the combined model with optional wandb logging.

    Parameters
    ----------
    model : nn.Module
        The model to train
    train_dataloader : DataLoader
        DataLoader containing training data
    test_dataloader : DataLoader
        DataLoader containing test data
    checkpoint_dir : str
        Directory to save model checkpoints
    plot_dir : str
        Directory to save plots
    bool_wandb : bool
        Whether to use wandb logging
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay for optimizer
    bool_clip_grad : bool
        Whether to clip gradients
    save_every : int
        Save checkpoint every N epochs
    moving_avg_window : int
        Window size for moving average of loss
    log_interval : int
        Interval for logging metrics to wandb
    validation_step_interval : int
        Run validation every N steps
    best_models_to_keep : int
        Number of best models to keep based on validation loss

    Returns
    -------
    model : nn.Module
        Trained model
    training_stats : list
        List containing training statistics

    """
    device = return_device()
    model = model.to(device)

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # loss function
    criterion = nn.MSELoss()

    # learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=percent_warmup * total_steps,
        num_training_steps=total_steps,
    )

    # checkpoint directory if using wandb
    if bool_wandb:
        os.makedirs(checkpoint_dir, exist_ok=True)
        wandb.watch(model, criterion, log="all", log_freq=10)

        # define custom metrics to avoid monotonically increasing step warnings
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("val_step/*", step_metric="val_step")
        wandb.define_metric("val/*", step_metric="epoch")

    best_val_loss = float("inf")
    training_stats = []
    global_step = 0

    # moving average for loss
    recent_losses = []
    recent_lr = []

    # tracking best models
    best_model_heap = []
    saved_models = {}

    # tracking loss over time
    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    step_list = []
    step_val_loss_list = []

    # training loop
    for epoch in tqdm(range(epochs), desc="Training epochs..."):
        model.train()
        total_train_loss = 0
        batch_count = 0

        for batch in tqdm(train_dataloader, desc="Training batches..."):
            batch_count += 1
            global_step += 1

            smiles_input_ids = batch["smiles_input_ids"].to(device)
            smiles_attention_mask = batch["smiles_attention_mask"].to(device)
            klifs_input_ids = batch["klifs_input_ids"].to(device)
            klifs_attention_mask = batch["klifs_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_drug=smiles_input_ids,
                mask_drug=smiles_attention_mask,
                input_kinase=klifs_input_ids,
                mask_kinase=klifs_attention_mask,
            )

            loss = criterion(outputs, labels)

            loss.backward()

            # optional: clip gradients to prevent exploding gradients
            if bool_clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if bool_wandb:
                recent_losses.append(loss.item())
                recent_lr.append(scheduler.get_last_lr()[0])

                # keep loss and lr lists at window size
                if len(recent_losses) > moving_avg_window:
                    recent_losses.pop(0)
                if len(recent_lr) > moving_avg_window:
                    recent_lr.pop(0)

                # log smoothed values every log_interval batches
                if batch_count % log_interval == 0:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    avg_lr = sum(recent_lr) / len(recent_lr)

                    # use the custom step metric
                    wandb.log(
                        {
                            "train_step": global_step,
                            "train/loss": avg_loss,
                            "train/loss_raw": loss.item(),
                            "train/learning_rate": avg_lr,
                        }
                    )

            # run validation at regular intervals during training
            if (global_step % validation_step_interval == 0) and bool_wandb:
                print(f"Running validation at step {global_step}...")
                # evaluate model
                val_metrics = evaluate_model(model, test_dataloader, criterion, device)

                # log metrics to wandb with custom step
                step_metrics = {
                    "val_step": global_step,
                    "val_step/loss": val_metrics["loss"],
                    "val_step/mse": val_metrics["mse"],
                    "val_step/rmse": val_metrics["rmse"],
                    "val_step/mae": val_metrics["mae"],
                    "val_step/r2": val_metrics["r2"],
                }
                wandb.log(step_metrics)

                # plot and log predictions
                plot_path = create_prediction_plot(
                    val_metrics["labels"],
                    val_metrics["predictions"],
                    "Predictions vs. Actual Values",
                    plot_dir,
                    step=global_step,
                )

                # log plot to wandb with custom step
                wandb.log(
                    {
                        "val_step": global_step,
                        "val_step/predictions": wandb.Image(plot_path),
                    }
                )

                # track for loss over time plot
                step_list.append(global_step)
                step_val_loss_list.append(val_metrics["loss"])

                # plot loss over time
                plt.figure(figsize=(12, 6))
                plt.plot(
                    step_list,
                    step_val_loss_list,
                    "b-",
                    label="Validation Loss (by step)",
                )
                if epoch_list and val_loss_list:
                    # convert epoch numbers to equivalent step numbers for plotting on same axis
                    epoch_steps = [(e + 1) * len(train_dataloader) for e in epoch_list]
                    plt.plot(
                        epoch_steps,
                        val_loss_list,
                        "r-",
                        label="Validation Loss (by epoch)",
                    )
                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                plt.title("Validation Loss Over Time")
                plt.legend()
                plt.grid(True)

                loss_plot_path = "validation_loss_over_time.png"
                plt.savefig(os.path.join(plot_dir, loss_plot_path))
                plt.close()

                # log plot to wandb with custom step
                wandb.log(
                    {
                        "val_step": global_step,
                        "val_step/loss_over_time": wandb.Image(loss_plot_path),
                    }
                )

                # return to training mode
                model.train()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # full validation at end of epoch
        model.eval()
        val_metrics = evaluate_model(model, test_dataloader, criterion, device)
        avg_val_loss = val_metrics["loss"]
        mse = val_metrics["mse"]
        rmse = val_metrics["rmse"]
        mae = val_metrics["mae"]
        r2 = val_metrics["r2"]
        all_preds = val_metrics["predictions"]
        all_labels = val_metrics["labels"]

        # save training stats
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mse": mse,
            "val_rmse": rmse,
            "val_mae": mae,
            "val_r2": r2,
        }
        training_stats.append(epoch_stats)

        # update loss tracking lists for plotting
        epoch_list.append(epoch)
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

        print(f"Epoch: {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

        # handle wandb-specific operations
        if bool_wandb:
            # create metrics dictionary for wandb with custom epoch step
            epoch_metrics = {
                "epoch": epoch,
                "val/loss": avg_val_loss,
                "val/mse": mse,
                "val/rmse": rmse,
                "val/mae": mae,
                "val/r2": r2,
            }

            # log metrics to wandb
            wandb.log(epoch_metrics)

            # create and log validation prediction plot
            plot_path = create_prediction_plot(
                all_labels,
                all_preds,
                "Predictions vs. Actual Values",
                plot_dir,
                epoch=epoch,
            )

            # log plot to wandb with custom epoch step
            wandb.log({"epoch": epoch, "val/predictions": wandb.Image(plot_path)})

            # plot loss over time (epoch-based)
            plt.figure(figsize=(12, 6))
            plt.plot(epoch_list, train_loss_list, "g-", label="Training Loss")
            plt.plot(epoch_list, val_loss_list, "r-", label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss Over Time")
            plt.legend()
            plt.grid(True)

            loss_plot_path = "loss_curves.png"
            plt.savefig(os.path.join(plot_dir, loss_plot_path))
            plt.close()

            wandb.log({"epoch": epoch, "val/loss_curves": wandb.Image(loss_plot_path)})

        # save model checkpoint and manage best models
        if bool_wandb and (epoch % save_every == 0 or epoch == epochs - 1):
            model_name = f"model_epoch_{epoch+1}.pt"
            model_path = os.path.join(checkpoint_dir, model_name)

            # save the model
            torch.save(model.state_dict(), model_path)

            # add to best models heap if needed
            if len(best_model_heap) < best_models_to_keep:
                heapq.heappush(best_model_heap, (-avg_val_loss, model_path, epoch + 1))
                saved_models[model_path] = -avg_val_loss
            elif -avg_val_loss > best_model_heap[0][0]:
                # remove worst model from saved models
                _, worst_path, _ = heapq.heappop(best_model_heap)
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                del saved_models[worst_path]

                # add new model to best models
                heapq.heappush(best_model_heap, (-avg_val_loss, model_path, epoch + 1))
                saved_models[model_path] = -avg_val_loss
            else:
                # not among best models, delete it
                if os.path.exists(model_path):
                    os.remove(model_path)

            print(
                f"Model saved at epoch {epoch+1}. Keeping best {len(saved_models)} models."
            )

            # save checkpoint with metadata
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "metrics": {
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                    },
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            if bool_wandb:
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                # save best model
                torch.save(model.state_dict(), best_model_path)

                # only log to wandb if the file exists
                if os.path.exists(best_model_path):
                    # create model artifact
                    model_artifact = wandb.Artifact(
                        name=f"model-{wandb.run.id}",
                        type="model",
                        description="Best model based on validation loss",
                        metadata={
                            "epoch": epoch + 1,
                            "val_loss": avg_val_loss,
                            "val_rmse": rmse,
                            "val_r2": r2,
                        },
                    )
                    # add file to artifact
                    model_artifact.add_file(best_model_path)
                    # log artifact
                    wandb.log_artifact(model_artifact)
                    print("Best model saved and logged to wandb!")
                else:
                    print(
                        f"Warning: Could not find best model file at {best_model_path}"
                    )
            else:
                best_model_path = "best_combined_model.pt"
                torch.save(model.state_dict(), best_model_path)
                print("Best model saved!")

    # print summary of kept models
    if bool_wandb:
        print("\nBest models summary:")
        sorted_models = sorted(
            [(val, path) for path, val in saved_models.items()], key=lambda x: x[0]
        )
        for i, (neg_loss, path) in enumerate(sorted_models):
            print(f"{i+1}. Model at {path}: Val Loss = {-neg_loss:.6f}")

    return model, training_stats


def evaluate_model_with_wandb(
    model: nn.Module,
    test_dataloader: DataLoader,
    scaler: object,
    plot_dir: str,
):
    """Evaluate the trained model on the test set with wandb logging.

    Parameters
    ----------
    model : nn.Module
        The trained model to evaluate
    test_dataloader : DataLoader
        DataLoader containing test data
    scaler : object
        Scaler used for standardizing the data
    plot_dir : str
        Directory to save plots

    Returns
    -------
    dict
        Dictionary containing evaluation metrics

    """
    device = return_device()
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            smiles_input_ids = batch["smiles_input_ids"].to(device)
            smiles_attention_mask = batch["smiles_attention_mask"].to(device)
            klifs_input_ids = batch["klifs_input_ids"].to(device)
            klifs_attention_mask = batch["klifs_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_drug=smiles_input_ids,
                mask_drug=smiles_attention_mask,
                input_kinase=klifs_input_ids,
                mask_kinase=klifs_attention_mask,
            )

            # Store predictions and labels
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics on standardized scale
    mse_std = mean_squared_error(all_labels, all_preds)
    rmse_std = np.sqrt(mse_std)
    mae_std = mean_absolute_error(all_labels, all_preds)
    r2_std = r2_score(all_labels, all_preds)

    # Log standardized metrics
    wandb.log(
        {
            "test/mse_standardized": mse_std,
            "test/rmse_standardized": rmse_std,
            "test/mae_standardized": mae_std,
            "test/r2_standardized": r2_std,
        }
    )

    # Convert back to original scale
    all_preds_original = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))
    all_labels_original = scaler.inverse_transform(np.array(all_labels).reshape(-1, 1))

    # Calculate metrics on original scale
    mse = mean_squared_error(all_labels_original, all_preds_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels_original, all_preds_original)
    r2 = r2_score(all_labels_original, all_preds_original)

    # Log original scale metrics
    wandb.log({"test/mse": mse, "test/rmse": rmse, "test/mae": mae, "test/r2": r2})

    print("Test set metrics (original scale):")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

    # Plot predictions vs. actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(all_labels_original, all_preds_original, alpha=0.5)
    plt.plot(
        [min(all_labels_original), max(all_labels_original)],
        [min(all_labels_original), max(all_labels_original)],
        "r--",
    )
    plt.xlabel("True percent_displacement")
    plt.ylabel("Predicted percent_displacement")
    plt.title("Predictions vs. Actual Values (Test Set)")

    plot_path = "test_prediction_scatter.png"
    plt.savefig(os.path.join(plot_dir, plot_path))
    plt.close()

    # Log plot to wandb
    wandb.log({"test/predictions": wandb.Image(plot_path)})

    # Create and log error distribution histogram
    errors = all_preds_original.flatten() - all_labels_original.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Distribution of Prediction Errors")

    error_plot_path = "error_distribution.png"
    plt.savefig(os.path.join(plot_dir, error_plot_path))
    plt.close()

    # Log error plot to wandb
    wandb.log({"test/error_distribution": wandb.Image(error_plot_path)})

    # Create a table of actual vs predicted values for wandb
    data = []
    for i in range(min(len(all_labels_original), 100)):  # Limit to 100 examples
        data.append(
            [i, float(all_labels_original[i][0]), float(all_preds_original[i][0])]
        )

    table = wandb.Table(columns=["Index", "Actual", "Predicted"], data=data)
    wandb.log({"predictions_table": table})

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": all_preds_original,
        "labels": all_labels_original,
    }


def run_pipeline_with_wandb(
    model: nn.Module,
    dataset_train: Dataset,
    dataset_test: Dataset,
    # scaler: object = None,
    model_name: str,
    batch_size=32,
    epochs=100,
    learning_rate=2e-5,  # optimizer
    weight_decay=0.01,  # optimizer
    percent_warmup: float = 0.1,  # scheduler
    bool_clip_grad: bool = True,
    save_every: int = 1,
    moving_avg_window: int = 100,
    log_interval: int = 10,
    validation_step_interval: int = 1000,
    best_models_to_keep: int = 5,
    project_name: str = "ki_llm_mxfactor",
    entity_name: str | None = "tansey-lab",
):
    """Run the complete training and evaluation pipeline with wandb integration.

    Parameters
    ----------
    model : nn.Module
        The model to train
    dataset_train : Dataset
        Dataset containing training data
    dataset_test : Dataset
        Dataset containing test data
    model_name : str
        Informative name of the model in wandb logging
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay for optimizer
    bool_clip_grad : bool
        Whether to clip gradients
    bool_wandb : bool
        Whether to use wandb logging
    checkpoint_dir : str
        Directory to save model checkpoints
    save_every : int
        Save checkpoint every N epochs
    moving_avg_window : int
        Window size for moving average of loss
    log_interval : int
        Interval for logging metrics to wandb
    validation_step_interval : int
        Run validation every N steps
    best_models_to_keep : int
        Number of best models to keep based on validation loss

    """
    # arguments to be logged only
    config_log = {
        "model_name": model_name,
        "optimizer": "AdamW",  # only thing currently implemented
        "scheduler": "linear_with_warmup",  # only thing currently implemented
        "batch_size": batch_size,  # only in dataloader
    }

    # configs passed to train_model()
    config_train = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "percent_warmup": percent_warmup,
        "bool_clip_grad": bool_clip_grad,
        "save_every": save_every,
        "moving_avg_window": moving_avg_window,
        "log_interval": log_interval,
        "validation_step_interval": validation_step_interval,
        "best_models_to_keep": best_models_to_keep,
    }

    config_log.update(config_train)

    run = setup_wandb(project_name, entity_name, config_log)

    try:
        wandb.run.summary.update(
            {"train_size": len(dataset_train), "test_size": len(dataset_test)}
        )

        train_dataloader, test_dataloader = create_dataloaders(
            dataset_train,
            dataset_test,
            batch_size,
        )

        # create checkpoint directory locally with run ID
        checkpoint_dir = f"checkpoints/{wandb.run.id}"
        plot_dir = f"{checkpoint_dir}/figures"
        os.makedirs(plot_dir, exist_ok=True)
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        logger.info(f"Plot directory: {plot_dir}\n")

        # train model with wandb logging
        trained_model, training_stats = train_model(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            checkpoint_dir=checkpoint_dir,
            plot_dir=plot_dir,
            bool_wandb=True,  # wandb logging by definition
            **config_train,
        )

        # evaluate model with wandb logging
        # eval_results = evaluate_model_with_wandb(
        #     model=trained_model,
        #     test_dataloader=test_dataloader,
        #     scaler=scaler,
        #     plot_dir=plot_dir, #TODO
        # )

        # log a summary of the best results
        wandb.run.summary.update(
            {
                "best_val_loss": min([stat["val_loss"] for stat in training_stats]),
                "best_val_rmse": min([stat["val_rmse"] for stat in training_stats]),
                "best_val_r2": max([stat["val_r2"] for stat in training_stats]),
                # "test_rmse": eval_results["rmse"],
                # "test_r2": eval_results["r2"],
            }
        )

        # mark the run as successful
        wandb.run.tags = wandb.run.tags + ("success",)

        # return trained_model, training_stats, eval_results
        return trained_model, training_stats

    except Exception as e:
        wandb.run.tags = wandb.run.tags + ("error",)
        wandb.run.summary.update({"error": str(e)})
        logger.error(f"Error during training: {e}")

    finally:
        run.finish()
