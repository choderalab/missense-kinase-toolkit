import torch
import torch.nn as nn
from transformers import AutoModel


class CombinedPoolingModel(nn.Module):
    """Combined pooling model for kinase and drug data."""

    def __init__(
        self,
        model_name_drug,
        model_name_kinase,
        hidden_size=256,
        bool_drug_freeze=False,
        bool_kinase_freeze=False,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.model_name_drug = model_name_drug
        self.model_name_kinase = model_name_kinase
        self.hidden_size = hidden_size
        self.bool_freeze = bool_freeze
        self.dropout_rate = dropout_rate

        self.model_drug = AutoModel.from_pretrained(self.model_name_drug)
        self.model_kinase = AutoModel.from_pretrained(self.model_name_kinase)

        if self.bool_drug_freeze:
            for param in self.model_drug.parameters():
                param.requires_grad = False

        if self.bool_kinase_freeze:
            for param in self.model_kinase.parameters():
                param.requires_grad = False

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

    def forward(
        self,
        input_drug,
        mask_drug,
        input_kinase,
        mask_kinase,
    ):

        mx_drug = self.model_drug(
            input_ids=input_drug,
            attention_mask=mask_drug,
            output_hidden_states=True,
        )

        mx_kinase = self.model_kinase(
            input_ids=input_kinase,
            attention_mask=mask_kinase,
            output_hidden_states=True,
        )

        linear_drug = self.linear_drug(mx_drug.pooler_output)
        linear_kinase = self.linear_kinase(mx_kinase.pooler_output)

        # take the dot product of the two vectors per batch
        output = torch.einsum("bi,bi->b", linear_drug, linear_kinase).unsqueeze(1)

        return output
