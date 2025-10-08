import torch
import torch.nn as nn

from missense_kinase_toolkit.ml.mkt.ml.models.base_model import AbstractTransformModel


class CrossAttentionModel(AbstractTransformModel):
    """cross attention model for kinase and drug data using last_hidden_state."""

    def __init__(
        self,
        model_name_drug,
        model_name_kinase,
        layer_drug="last_hidden_state",  # hardcoded for last hidden state
        layer_kinase="last_hidden_state",  # hardcoded for last hidden state
        hidden_size=256,
        bool_drug_freeze=False,
        bool_kinase_freeze=False,
        num_heads=8,
        dropout_rate=0.1,
    ):
        super().__init__(
            model_name_drug,
            model_name_kinase,
            layer_drug,
            layer_kinase,
            hidden_size,
            bool_drug_freeze,
            bool_kinase_freeze,
        )
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # cross attention layers
        self.cross_attention_drug = nn.MultiheadAttention(
            embed_dim=self.model_drug.config.hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.cross_attention_kinase = nn.MultiheadAttention(
            embed_dim=self.model_kinase.config.hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        # learnable query vectors for pooling
        self.drug_query = nn.Parameter(
            torch.randn(1, 1, self.model_drug.config.hidden_size)
        )
        self.kinase_query = nn.Parameter(
            torch.randn(1, 1, self.model_kinase.config.hidden_size)
        )

        # linear transformation layers
        self.linear_drug = nn.Sequential(
            nn.Linear(self.model_drug.config.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.linear_kinase = nn.Sequential(
            nn.Linear(self.model_kinase.config.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def transform_drug(self, drug_output):
        """transform drug model output using cross attention.

        Parameters
        ----------
        drug_output : torch.Tensor
            Shape: (N, L, hidden_size) from last_hidden_state

        Returns
        -------
        torch.Tensor
            Shape: (N, hidden_size)
        """
        batch_size = drug_output.shape[0]

        # expand query to batch size: (1, 1, hidden) -> (N, 1, hidden)
        query = self.drug_query.expand(batch_size, -1, -1)

        # cross attention: query attends to all sequence positions
        # output shape: (N, 1, hidden)
        attended_output, _ = self.cross_attention_drug(
            query=query, key=drug_output, value=drug_output
        )

        # squeeze to remove sequence dimension: (N, 1, hidden) -> (N, hidden)
        pooled_output = attended_output.squeeze(1)

        # apply linear transformation
        return self.linear_drug(pooled_output)

    def transform_kinase(self, kinase_output):
        """transform kinase model output using cross attention.

        Parameters
        ----------
        kinase_output : torch.Tensor
            Shape: (N, L, hidden_size) from last_hidden_state

        Returns
        -------
        torch.Tensor
            Shape: (N, hidden_size)
        """
        batch_size = kinase_output.shape[0]

        # expand query to batch size: (1, 1, hidden) -> (N, 1, hidden)
        query = self.kinase_query.expand(batch_size, -1, -1)

        # cross attention: query attends to all sequence positions
        # output shape: (N, 1, hidden)
        attended_output, _ = self.cross_attention_kinase(
            query=query, key=kinase_output, value=kinase_output
        )

        # squeeze to remove sequence dimension: (N, 1, hidden) -> (N, hidden)
        pooled_output = attended_output.squeeze(1)

        # apply linear transformation
        return self.linear_kinase(pooled_output)
