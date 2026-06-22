import torch.nn as nn
from mkt.ml.models.base_model import AbstractTransformModel


class CombinedPoolingModel(AbstractTransformModel):
    """Combined pooling model for kinase and drug data."""

    def __init__(
        self,
        model_name_drug,
        model_name_kinase,
        layer_drug="pooler_output",  # hardcoded for pooling layer
        layer_kinase="pooler_output",  # hardcoded for pooling layer
        hidden_size=256,
        bool_drug_freeze=False,
        bool_kinase_freeze=False,
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
        self.dropout_rate = dropout_rate

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
        """Transform drug model output using linear layers."""
        return self.linear_drug(drug_output)

    def transform_kinase(self, kinase_output):
        """Transform kinase model output using linear layers."""
        return self.linear_kinase(kinase_output)
