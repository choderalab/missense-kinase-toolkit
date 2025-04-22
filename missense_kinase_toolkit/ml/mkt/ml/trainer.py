import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datasets import Dataset
from mkt.ml.datasets.pkis2 import PKIS2Dataset
from mkt.ml.models.pooling import CombinedPoolingModel
from mkt.ml.utils import return_device
from mkt.ml.utils_wandb import (
    log_metrics_to_wandb,
    log_model_to_wandb,
    log_plots_to_wandb,
    save_checkpoint,
    setup_wandb,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def create_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
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


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    bool_clip_grad: bool = True,
    bool_wandb: float = False,
    checkpoint_dir: str = "checkpoints",
    save_every: int = 1,
    moving_avg_window: int = 20,
    log_interval: int = 10,
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
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps,
    )

    # checkpoint directory if using wandb
    if bool_wandb:
        os.makedirs(checkpoint_dir, exist_ok=True)
        wandb.watch(model, criterion, log="all", log_freq=10)

    best_val_loss = float("inf")
    training_stats = []
    global_step = 0

    # moving average for loss
    recent_losses = []
    recent_lr = []

    # training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0

        for batch in train_dataloader:
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

                # Keep lists at window size
                if len(recent_losses) > moving_avg_window:
                    recent_losses.pop(0)
                if len(recent_lr) > moving_avg_window:
                    recent_lr.pop(0)

                # Log smoothed values every log_interval batches
                if batch_count % log_interval == 0:
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    avg_lr = sum(recent_lr) / len(recent_lr)

                    log_metrics_to_wandb(
                        {
                            "loss": avg_loss,
                            "loss_raw": loss.item(),  # Also log raw value for comparison
                            "learning_rate": avg_lr,
                        },
                        step=global_step,
                        prefix="train/",
                    )

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0
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

                # Calculate loss
                loss = criterion(outputs, labels)

                # Accumulate loss
                total_val_loss += loss.item()

                # Store predictions and labels for metrics
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_dataloader)

        # Calculate metrics
        mse = mean_squared_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)

        # Save training stats
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

        print(f"Epoch: {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

        # Handle wandb-specific operations
        if bool_wandb:
            # Create metrics dictionary for wandb
            metrics = {
                "loss": avg_val_loss,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }

            # Log metrics to wandb
            log_metrics_to_wandb(metrics, step=epoch, prefix="val/")

            # Save checkpoint
            if epoch % save_every == 0 or epoch == epochs - 1:
                checkpoint_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    metrics=metrics,
                    checkpoint_dir=checkpoint_dir,
                )
                print(f"Checkpoint saved to {checkpoint_path}")

            # Create and log validation prediction plot
            plt.figure(figsize=(10, 6))
            plt.scatter(all_labels, all_preds, alpha=0.5)
            plt.plot(
                [min(all_labels), max(all_labels)],
                [min(all_labels), max(all_labels)],
                "r--",
            )
            plt.xlabel("True standardized percent_displacement")
            plt.ylabel("Predicted standardized percent_displacement")
            plt.title(f"Predictions vs. Actual Values (Epoch {epoch+1})")

            plot_path = os.path.join(
                checkpoint_dir, f"val_predictions_epoch_{epoch+1}.png"
            )
            plt.savefig(plot_path)
            plt.close()

            # Log plot to wandb
            log_plots_to_wandb({"val_predictions": plot_path}, step=epoch + 1)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            if bool_wandb:
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                # Log best model to wandb
                log_model_to_wandb(
                    model_path=best_model_path,
                    metadata={
                        "epoch": epoch + 1,
                        "val_loss": avg_val_loss,
                        "val_rmse": rmse,
                        "val_r2": r2,
                    },
                )
            else:
                best_model_path = "best_combined_model.pt"

            torch.save(model.state_dict(), best_model_path)
            print("Best model saved!")

    return model, training_stats


def evaluate_model_with_wandb(model, test_dataloader, scaler):
    """Evaluate the trained model on the test set with wandb logging."""
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

            print()

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
    log_metrics_to_wandb(
        {
            "mse_standardized": mse_std,
            "rmse_standardized": rmse_std,
            "mae_standardized": mae_std,
            "r2_standardized": r2_std,
        },
        prefix="test/",
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
    log_metrics_to_wandb(
        {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, prefix="test/"
    )

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
    plt.savefig(plot_path)
    plt.close()

    # Log plot to wandb
    log_plots_to_wandb({"test_predictions": plot_path})

    # Create and log error distribution histogram
    errors = all_preds_original.flatten() - all_labels_original.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Distribution of Prediction Errors")

    error_plot_path = "error_distribution.png"
    plt.savefig(error_plot_path)
    plt.close()

    # Log error plot to wandb
    log_plots_to_wandb({"error_distribution": error_plot_path})

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
    # csv_file_path,
    # test_kincore_groups,
    batch_size=32,
    epochs=100,
    learning_rate=2e-5,
    hidden_size=256,
    project_name="tansey-lab/ki_llm_mxfactor",
):
    """Run the complete training and evaluation pipeline with wandb integration."""
    # Set up wandb config
    config = {
        "model_name": "Pooling_ESM2_ChemBERTa",
        # "dataset": csv_file_path,
        # "test_groups": test_kincore_groups,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "scheduler": "linear_with_warmup",
    }

    run = setup_wandb(project_name, config)

    try:
        dataset_pkis2 = PKIS2Dataset()

        wandb.run.summary.update(
            {
                "train_size": len(dataset_pkis2.dataset_train),
                "test_size": len(dataset_pkis2.dataset_test),
            }
        )

        train_dataloader, test_dataloader = create_dataloaders(
            dataset_pkis2.dataset_train,
            dataset_pkis2.dataset_test,
        )

        model = CombinedPoolingModel(
            model_name_drug=dataset_pkis2.model_drug,
            model_name_kinase=dataset_pkis2.model_kinase,
            hidden_size=hidden_size,
        )

        # create checkpoint directory locally with run ID
        checkpoint_dir = f"checkpoints/{wandb.run.id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # train model with wandb logging
        trained_model, training_stats = train_model(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=epochs,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            bool_wandb=True,
            save_every=1,
            moving_avg_window=20,
            log_interval=10,
        )

        # evaluate model with wandb logging
        eval_results = evaluate_model_with_wandb(
            model=trained_model,
            test_dataloader=test_dataloader,
            scaler=dataset_pkis2.scaler,
        )

        # log a summary of the best results
        wandb.run.summary.update(
            {
                "best_val_loss": min([stat["val_loss"] for stat in training_stats]),
                "best_val_rmse": min([stat["val_rmse"] for stat in training_stats]),
                "best_val_r2": max([stat["val_r2"] for stat in training_stats]),
                "test_rmse": eval_results["rmse"],
                "test_r2": eval_results["r2"],
            }
        )

        # mark the run as successful
        wandb.run.tags = wandb.run.tags + ("success",)

        return trained_model, training_stats, eval_results

    except Exception as e:
        wandb.run.tags = wandb.run.tags + ("error",)
        wandb.run.summary.update({"error": str(e)})
        raise e

    finally:
        run.finish()
