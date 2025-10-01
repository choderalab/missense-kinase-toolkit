import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mkt.ml.utils import set_seed


class LogisticTensorFactorModel(nn.Module):
    """A logistic tensor factor model for multi-dimensional data."""
    def __init__(
        self,
        I: int,
        J: int,
        K: int,
        D: int = 100,
        init_std: float = 0.1,
    ):
        """Initialize the LogisticTensorFactorModel.

        Parameters
        ----------
        I : int
            Dimensionality of input matrix I.
        J : int
            Dimensionality of input matrix J.
        K : int
            Dimensionality of input matrix K.
        D : int, optional
            Dimensionality of latent factors, by default 100.
        init_std : float, optional
            Standard deviation for weight initialization, by default 0.1.
        """
        super().__init__()

        self.I, self.J, self.K, self.D = I, J, K, D
        
        # initialize factor matrices
        self.W = nn.Parameter(torch.randn(I, D) * init_std)  # I x D
        self.V = nn.Parameter(torch.randn(J, D) * init_std)  # J x D  
        self.U = nn.Parameter(torch.randn(K, D) * init_std)  # K x D
        
    def forward(self, mask=None):
        """Forward pass of the LogisticTensorFactorModel.

        Parameters
        ----------
        mask : torch.Tensor | None, optional
            Boolean mask to select specific entries, by default None.

        Returns
        -------
        torch.Tensor
            The output probabilities after applying the logistic function.
        """
        # compute full tensor using einsum
        theta = torch.einsum('id,jd,kd->ijk', self.W, self.V, self.U)
        
        # logistic function: 1 / (1 + exp(-theta))
        probs = torch.sigmoid(theta)
        
        if mask is not None:
            return probs[mask]
        else:
            return probs

@dataclass
class TensorFactorTrainer:
    """Trainer class for the logistic tensor factor model."""
    model: LogisticTensorFactorModel = field(init=False)
    """The logistic tensor factor model to be trained."""
    optimizer: optim.Optimizer = field(init=False)
    """Optimizer for training."""
    loss_fn: nn.Module = field(init=False)
    """Loss function for training."""
    learning_rate: float = 0.01
    """Learning rate for the optimizer."""
    weight_decay: float = 1e-4
    """Weight decay (L2 regularization) for the optimizer."""

    def __post_init__(self):
        if hasattr(self, 'model') and self.model is not None:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        self.loss_fn = nn.BCELoss()
        
    def train_step(self, mask, targets):
        """Single training step.
        
        Parameters
        ----------
        mask : torch.Tensor
            Boolean tensor of shape (I, J, K) indicating observed entries.
        targets : torch.Tensor
            Tensor with target probabilities for masked entries.
        """
        self.optimizer.zero_grad()
        
        pred_probs = self.model(mask)
        
        loss = self.loss_fn(pred_probs, targets)

        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        """Train for one epoch.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing batches of (mask, targets).
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_mask, batch_targets in dataloader:
            # remove the extra dimension added by unsqueeze
            batch_mask = batch_mask.squeeze(0)
            batch_targets = batch_targets.squeeze(0)
            
            loss = self.train_step(batch_mask, batch_targets)
            total_loss += loss
            num_batches += 1
            
        return total_loss / num_batches
    
    def evaluate(self, dataloader):
        """Evaluate model on validation data.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing batches of (mask, targets).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_mask, batch_targets in dataloader:
                # Remove the extra dimension added by unsqueeze
                batch_mask = batch_mask.squeeze(0)
                batch_targets = batch_targets.squeeze(0)
                
                pred_probs = self.model(batch_mask)
                loss = self.loss_fn(pred_probs, batch_targets)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches

@dataclass(kw_only=True)
class TensorFactorPipeline(TensorFactorTrainer):
    """Pipeline class for end-to-end training and evaluation."""
    df: pd.DataFrame
    """DataFrame with data and splits."""
    seed: int = 24
    """Random seed for reproducibility."""
    percent_split: float = 0.2
    """Percentage of data to hold out for validation."""
    list_cols: list[str] = field(default_factory=lambda: ["hgnc", "klifs", "drug"])
    """Column names for the three dimensions."""
    target_col: str = "p(drug|mut)"
    """Column name for target probabilities."""
    val_col: str = "val"
    """Column name for validation split."""
    epochs: int = 100
    """Number of training epochs."""
    D: int = 100
    """Dimensionality of latent factors."""
    dataloader_train: DataLoader = field(init=False)
    """DataLoader for training data."""
    dataloader_val: DataLoader = field(init=False)
    """DataLoader for validation data."""
    I: int = field(init=False)
    """Dimension I."""
    J: int = field(init=False)
    """Dimension J."""
    K: int = field(init=False)
    """Dimension K."""

    def __post_init__(self):
        print("Initializing TensorFactorPipeline...")

        # Step 0: Set seed
        set_seed(self.seed)

        # Step 1: Generate split dataframe
        print("Generating train/validation split...")
        self.df = self.generate_split_dataframe()
        
        # Step 2: Get dimensions
        self.I, self.J, self.K = self.get_dimensions_from_df()
        print(f"    Tensor dimensions: I={self.I}, J={self.J}, K={self.K}")

        # Step 3: Create model
        print(f"Initializing model with D={self.D} latent factors...")
        model = LogisticTensorFactorModel(self.I, self.J, self.K, self.D)
        self.model = model
        
        # Step 4: Initialize parent class (creates optimizer)
        super().__post_init__()
        print(f"    Optimizer: {self.optimizer.__class__.__name__}, LR={self.learning_rate}")

        # Step 5: Create dataloaders
        print("Creating dataloaders...")
        self.dataloader_train, self.dataloader_val = self.create_dataloader_from_df()
        
        # Diagnostic: Check data
        self.diagnose_data()

    def diagnose_data(self):
        """Run diagnostics on the data to identify potential issues."""
        print("\n" + "="*50)
        print("DATA DIAGNOSTICS")
        print("="*50)
        
        # Check target value range
        targets = self.df[self.target_col].values
        print(f"Target column: {self.target_col}")
        print(f"  Range: [{targets.min():.4f}, {targets.max():.4f}]")
        print(f"  Mean: {targets.mean():.4f}")
        print(f"  Std: {targets.std():.4f}")
        
        if targets.min() < 0 or targets.max() > 1:
            print("  ⚠️  WARNING: Targets outside [0,1] range!")
        
        # Check split sizes
        n_train = (~self.df[self.val_col]).sum()
        n_val = self.df[self.val_col].sum()
        print(f"\nSplit sizes:")
        print(f"  Train: {n_train} ({n_train/len(self.df)*100:.1f}%)")
        print(f"  Val:   {n_val} ({n_val/len(self.df)*100:.1f}%)")
        
        # Check for overlap (should be zero)
        train_set = set(self.df[~self.df[self.val_col]].index)
        val_set = set(self.df[self.df[self.val_col]].index)
        overlap = train_set & val_set
        if overlap:
            print(f"  ⚠️  WARNING: {len(overlap)} samples in both train and val!")
        else:
            print(f"  ✓ No overlap between train and val")
        
        # Check model initialization
        print(f"\nModel parameters:")
        for name, param in self.model.named_parameters():
            print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
        
        # Check initial predictions
        with torch.no_grad():
            sample_mask = torch.zeros(self.I, self.J, self.K, dtype=torch.bool)
            sample_mask[0, 0, 0] = True
            sample_pred = self.model(sample_mask)
            print(f"\nInitial prediction at (0,0,0): {sample_pred.item():.4f}")
        
        print("="*50 + "\n")

    def generate_split_dataframe(self) -> pd.DataFrame:
        """Shuffle dataframe and create train-validation split with contiguous 3D blocks.

        For 3D tensor cross-validation, each validation set is a contiguous cuboid
        representing approximately percent_split of the total volume.
        
        Returns
        -------
        pd.DataFrame
            Annotated dataframe with split info.
        """
        df_split = self.df.sample(frac=1, random_state=self.seed).reset_index(drop=True).copy()
        dim_frac = self.percent_split ** (1/3)

        dict_splits = {}
        for col_name, col_series in df_split[self.list_cols].items():
            n_unique, series_unique = col_series.nunique(), col_series.unique()
            idx_end = max(1, int(np.round(dim_frac * n_unique)))
            dict_splits[col_name] = {
                "dict_map": dict(zip(range(n_unique), series_unique)),
                "list_idx": list(np.arange(n_unique)[0:idx_end])
            }

        for col in self.list_cols:
            dict_reverse = {v: k for k, v in dict_splits[col]["dict_map"].items()}
            df_split[col + "_idx"] = df_split[col].apply(lambda x: dict_reverse[x])
        df_split[self.val_col] = np.logical_and.reduce(
            [df_split[k].isin([v["dict_map"][i] for i in v["list_idx"]]) for k, v in dict_splits.items()])
        df_split.insert(len(df_split.columns)-1, self.target_col, df_split.pop(self.target_col))

        return df_split

    def get_dimensions_from_df(self) -> tuple[int, int, int]:
        """Get dimensions I, J, K from the dataframe based on index columns.

        Returns
        -------
        tuple[int, int, int]
            I, J, K dimensions.
        """
        cols_idx = [i for i in self.df.columns if i.endswith("_idx")]
        assert len(cols_idx) == 3
        I, J, K = [self.df[col].nunique() for col in cols_idx]
        return I, J, K

    def create_dataloader_from_df(self) -> tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders from a dataframe with indexed entries.
        
        Returns
        -------
        tuple[DataLoader, DataLoader]
            train_loader, val_loader: DataLoaders with masks and targets.
        """
        # separate train and validation data
        df_train = self.df[~self.df[self.val_col]].copy()
        df_val = self.df[self.df[self.val_col]].copy()

        # create masks for train and validation sets
        train_mask = torch.zeros(self.I, self.J, self.K, dtype=torch.bool)
        val_mask = torch.zeros(self.I, self.J, self.K, dtype=torch.bool)

        # More efficient mask creation using vectorized operations
        train_indices = df_train[[col + "_idx" for col in self.list_cols]].values
        val_indices = df_val[[col + "_idx" for col in self.list_cols]].values
        
        for idx in train_indices:
            train_mask[idx[0], idx[1], idx[2]] = True
            
        for idx in val_indices:
            val_mask[idx[0], idx[1], idx[2]] = True
        
        # extract target values
        train_targets = torch.tensor(df_train[self.target_col].values, dtype=torch.float32)
        val_targets = torch.tensor(df_val[self.target_col].values, dtype=torch.float32)

        # create datasets
        train_dataset = TensorDataset(train_mask.unsqueeze(0), train_targets.unsqueeze(0))
        val_dataset = TensorDataset(val_mask.unsqueeze(0), val_targets.unsqueeze(0))

        # create dataloaders - batch_size=1 because we're passing full masks
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader

    def train(self):
        """Complete training pipeline using dataframe with train/validation split.

        Returns
        -------
        tuple[list, list]
            train_losses, val_losses
        """
        train_losses = []
        val_losses = []

        print(
            f"Starting training: {(~self.df[self.val_col]).sum():,} train samples, "
            f"{self.df[self.val_col].sum():,} validation samples"
        )

        for epoch in tqdm(range(self.epochs), desc="Training epochs"):
            train_loss = self.train_epoch(self.dataloader_train)
            val_loss = self.evaluate(self.dataloader_val)

            if epoch % 100 == 0 or epoch < 5:
                # show more frequent updates early to catch issues
                print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
                
                # check if loss is stuck
                if epoch > 0 and abs(train_loss - train_losses[-1]) < 1e-6:
                    print("  ⚠️  WARNING: Loss not changing - model may not be learning!")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        return train_losses, val_losses
