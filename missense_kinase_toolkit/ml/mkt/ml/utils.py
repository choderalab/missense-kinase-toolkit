import torch

def return_device():
    """Return device

    Returns:
    --------
    str
        Device; either "cuda" or "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def generate_similarity_matrix(
    mx_input: torch.Tensor,
    bool_row: bool = True,
):
    """Generate similarity matrix

    Params:
    -------
    mx_input: torch.Tensor
        Input matrix
    bool_row: bool
        Whether to calculate similarity by row; default is True

    Returns:
    --------
    mx_similarity: torch.Tensor
        Square, symmetrix similarity matrix containing pairwise cosine similarities
    """
    if bool_row:
        mx_norm = mx_input / mx_input.norm(dim=1, p=2, keepdim=True)
        mx_similarity = mx_norm @ mx_norm.T
    else:
        mx_norm = mx_input / mx_input.norm(dim=0, p=2, keepdim=True)
        mx_similarity = mx_norm.T @ mx_norm

    return mx_similarity