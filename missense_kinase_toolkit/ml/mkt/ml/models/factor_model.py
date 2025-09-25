import torch
import torch.nn as nn


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
            Dimensionality of input maxtrix I.
        J : int
            Dimensionality of input maxtrix J.
        K : int
            Dimensionality of input maxtrix K.
        D : int, optional
            Dimensionality of latent factors, by default 100.
        init_std : float, optional
            Standard deviation for weight initialization, by default 0.1.
        """
        super().__init__()

        self.I, self.J, self.K, self.D = I, J, K, D

        self.W = nn.Parameter(torch.randn(self.I, self.D) * init_std)
        self.V = nn.Parameter(torch.randn(self.J, self.D) * init_std)
        self.U = nn.Parameter(torch.randn(self.K, self.D) * init_std)

    def forward(self, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the LogisticTensorFactorModel.

        Parameters
        ----------
        indices : torch.Tensor | None, optional
            Specific indices to compute the output for, by default None.

        Returns
        -------
        torch.Tensor
            The output probabilities after applying the logistic function.
        """
        if indices is not None:
            # compute for specific indices only (memory efficient)
            i_idx, j_idx, k_idx = indices[:, 0], indices[:, 1], indices[:, 2]

            # get relevant rows from factor matrices
            w_selected = self.W[i_idx]  # (N, D)
            v_selected = self.V[j_idx]  # (N, D)
            u_selected = self.U[k_idx]  # (N, D)

            # compute theta = sum_d w_id * v_jd * u_kd
            theta = torch.sum(w_selected * v_selected * u_selected, dim=1)  # (N,)

        else:
            theta = torch.einsum("id,jd,ke->ijk", self.W, self.V, self.U)

        probs = torch.sigmoid(theta)

        return probs
