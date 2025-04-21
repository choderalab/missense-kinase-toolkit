import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datasets import Dataset
from mkt.ml.datasets.pkis2 import PKIS2Datasets
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
    batch_size=32,
    bool_train_shuffle=True,
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

    """

    # Define a custom collate function to handle the dataset format
    def collate_fn(batch):
        smiles_input_ids = torch.stack([item["smiles_input_ids"] for item in batch])
        smiles_attention_mask = torch.stack(
            [item["smiles_attention_mask"] for item in batch]
        )
        klifs_input_ids = torch.stack([item["klifs_input_ids"] for item in batch])
        klifs_attention_mask = torch.stack(
            [item["klifs_attention_mask"] for item in batch]
        )
        labels = torch.tensor(
            [item["labels"] for item in batch], dtype=torch.float
        ).view(-1, 1)

        return {
            "smiles_input_ids": smiles_input_ids.squeeze(1),
            "smiles_attention_mask": smiles_attention_mask.squeeze(1),
            "klifs_input_ids": klifs_input_ids.squeeze(1),
            "klifs_attention_mask": klifs_attention_mask.squeeze(1),
            "labels": labels,
        }

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool_train_shuffle,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, test_dataloader


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
):
    """Train the combined model."""
    device = return_device()
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Create a learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float("inf")
    training_stats = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            # Move batch to device
            smiles_input_ids = batch["smiles_input_ids"].to(device)
            smiles_attention_mask = batch["smiles_attention_mask"].to(device)
            klifs_input_ids = batch["klifs_input_ids"].to(device)
            klifs_attention_mask = batch["klifs_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                smiles_input_ids=smiles_input_ids,
                smiles_attention_mask=smiles_attention_mask,
                klifs_input_ids=klifs_input_ids,
                klifs_attention_mask=klifs_attention_mask,
            )

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Accumulate loss
            total_train_loss += loss.item()

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
                    smiles_input_ids=smiles_input_ids,
                    smiles_attention_mask=smiles_attention_mask,
                    klifs_input_ids=klifs_input_ids,
                    klifs_attention_mask=klifs_attention_mask,
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
        training_stats.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_mse": mse,
                "val_rmse": rmse,
                "val_mae": mae,
                "val_r2": r2,
            }
        )

        print(f"Epoch: {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_combined_model.pt")
            print("Best model saved!")

    return model, training_stats


def train_model_with_wandb(
    model,
    train_dataloader,
    test_dataloader,
    epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    checkpoint_dir="checkpoints",
    save_every=1,
):
    """Train the combined model with wandb logging."""
    device = return_device()
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Create a learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    training_stats = []
    global_step = 0

    # Initialize WandB to watch the model
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        batch_count = 0

        for batch in train_dataloader:
            batch_count += 1
            global_step += 1

            # Move batch to device
            smiles_input_ids = batch["smiles_input_ids"].to(device)
            smiles_attention_mask = batch["smiles_attention_mask"].to(device)
            klifs_input_ids = batch["klifs_input_ids"].to(device)
            klifs_attention_mask = batch["klifs_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                smiles_input_ids=smiles_input_ids,
                smiles_attention_mask=smiles_attention_mask,
                klifs_input_ids=klifs_input_ids,
                klifs_attention_mask=klifs_attention_mask,
            )

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Accumulate loss
            total_train_loss += loss.item()

            # Log batch loss to wandb
            if batch_count % 10 == 0:  # Log every 10 batches
                log_metrics_to_wandb(
                    {"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]},
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
                    smiles_input_ids=smiles_input_ids,
                    smiles_attention_mask=smiles_attention_mask,
                    klifs_input_ids=klifs_input_ids,
                    klifs_attention_mask=klifs_attention_mask,
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

        # Create metrics dictionary
        metrics = {"loss": avg_val_loss, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

        # Log metrics to wandb
        log_metrics_to_wandb(metrics, step=epoch, prefix="val/")

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

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)

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

            print("Best model saved!")

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

        plot_path = os.path.join(checkpoint_dir, f"val_predictions_epoch_{epoch+1}.png")
        plt.savefig(plot_path)
        plt.close()

        # Log plot to wandb
        log_plots_to_wandb({"val_predictions": plot_path}, step=epoch + 1)

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

            # Forward pass
            outputs = model(
                smiles_input_ids=smiles_input_ids,
                smiles_attention_mask=smiles_attention_mask,
                klifs_input_ids=klifs_input_ids,
                klifs_attention_mask=klifs_attention_mask,
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


# Full pipeline execution function with wandb integration
def run_pipeline_with_wandb(
    # csv_file_path,
    # test_kincore_groups,
    batch_size=16,
    epochs=10,
    learning_rate=2e-5,
    hidden_size=256,
    project_name="ki_llm_mxfactor",
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

        wandb.log(
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
            model_name_kinase=dataset_pkis2.model_kinase,
            model_name_drug=dataset_pkis2.model_drug,
            hidden_size=hidden_size,
        )

        # Create checkpoint directory with run ID
        checkpoint_dir = f"checkpoints/{wandb.run.id}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Train model with wandb logging
        trained_model, training_stats = train_model_with_wandb(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=epochs,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
        )

        # Evaluate model with wandb logging
        eval_results = evaluate_model_with_wandb(
            model=trained_model, test_dataloader=test_dataloader, scaler=scaler
        )

        # Log a summary of the best results
        wandb.run.summary.update(
            {
                "best_val_loss": min([stat["val_loss"] for stat in training_stats]),
                "best_val_rmse": min([stat["val_rmse"] for stat in training_stats]),
                "best_val_r2": max([stat["val_r2"] for stat in training_stats]),
                "test_rmse": eval_results["rmse"],
                "test_r2": eval_results["r2"],
            }
        )

        # Mark the run as successful
        wandb.run.tags = wandb.run.tags + ("success",)

        return trained_model, training_stats, eval_results

    except Exception as e:
        # Log the error
        wandb.run.tags = wandb.run.tags + ("error",)
        wandb.run.summary.update({"error": str(e)})
        raise e

    finally:
        # Finish the wandb run
        run.finish()


# # Example usage
# if __name__ == "__main__":
#     # Example file path and test groups
#     csv_file_path = "your_dataset.csv"
#     test_kincore_groups = ["group1", "group2"]  # Replace with actual groups

#     model, stats, results = run_pipeline_with_wandb(
#         csv_file_path=csv_file_path,
#         test_kincore_groups=test_kincore_groups,
#         batch_size=16,
#         epochs=10,
#         project_name="percent-displacement-prediction"
#     )

#     print("Pipeline execution complete!")
