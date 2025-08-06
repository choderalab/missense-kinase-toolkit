from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from mkt.ml.utils import rgetattr
from transformers import AutoModel


class BaseCombinedModel(nn.Module):
    """Base class with model loading and einsum computation.

    Parameters
    ----------
    model_name_drug : str
        Name of the drug model to load.
    model_name_kinase : str
        Name of the kinase model to load.
    layer_drug : str
        Layer name to extract from the drug model.
    layer_kinase : str
        Layer name to extract from the kinase model.
    hidden_size : int, optional
        Size of the hidden layer, by default 256.
    bool_drug_freeze : bool, optional
        If True, freeze the drug model parameters, by default False.
    bool_kinase_freeze : bool, optional
        If True, freeze the kinase model parameters, by default False.
    """

    def __init__(
        self,
        model_name_drug: str,
        model_name_kinase: str,
        layer_drug: str,
        layer_kinase: str,
        hidden_size: int = 256,
        bool_drug_freeze: bool = False,
        bool_kinase_freeze: bool = False,
    ):
        super().__init__()
        self.model_name_drug = model_name_drug
        self.model_name_kinase = model_name_kinase
        self.layer_drug = layer_drug
        self.layer_kinase = layer_kinase
        self.hidden_size = hidden_size
        self.bool_drug_freeze = bool_drug_freeze
        self.bool_kinase_freeze = bool_kinase_freeze

        self.model_drug = AutoModel.from_pretrained(self.model_name_drug)
        self.model_kinase = AutoModel.from_pretrained(self.model_name_kinase)

        if self.bool_drug_freeze:
            for param in self.model_drug.parameters():
                param.requires_grad = False

        if self.bool_kinase_freeze:
            for param in self.model_kinase.parameters():
                param.requires_grad = False

    def compute_similarity(self, linear_drug, linear_kinase):
        """Compute similarity using einsum dot product."""
        return torch.einsum("bi,bi->b", linear_drug, linear_kinase).unsqueeze(1)


class AbstractTransformModel(BaseCombinedModel, ABC):
    """Abstract model with transform methods for drug and kinase."""

    def __init__(
        self,
        model_name_drug,
        layer_drug,
        model_name_kinase,
        layer_kinase,
        hidden_size=256,
        bool_drug_freeze=False,
        bool_kinase_freeze=False,
    ):
        super().__init__(
            model_name_drug,
            layer_drug,
            model_name_kinase,
            layer_kinase,
            hidden_size,
            bool_drug_freeze,
            bool_kinase_freeze,
        )

    @abstractmethod
    def transform_drug(self, drug_output):
        """Transform drug model output."""
        pass

    @abstractmethod
    def transform_kinase(self, kinase_output):
        """Transform kinase model output."""
        pass

    def forward(
        self,
        input_drug: torch.Tensor,
        mask_drug: torch.Tensor,
        input_kinase: torch.Tensor,
        mask_kinase: torch.Tensor,
    ):
        """Forward pass through the model.

        Parameters
        ----------
        input_drug : torch.Tensor
            Input tensor for the drug model.
        mask_drug : torch.Tensor
            Attention mask for the drug model.
        input_kinase : torch.Tensor
            Input tensor for the kinase model.
        mask_kinase : torch.Tensor
            Attention mask for the kinase model.

        Returns
        -------
        torch.Tensor
            Similarity scores between drug and kinase representations.
        """
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

        linear_drug = self.transform_drug(rgetattr(mx_drug, self.layer_drug))
        linear_kinase = self.transform_kinase(rgetattr(mx_kinase, self.layer_kinase))

        return self.compute_similarity(linear_drug, linear_kinase)
