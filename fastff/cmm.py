import torch
from typing import Optional, Tuple

def conditional_matmul(
    tensor: torch.Tensor,
    weights: torch.Tensor,
    max_depth: int,
    matmul_tensors = None
):
    """
    Conditional MatMul
    ------------------
    Computes the conditional matmul of a tensor and a set of weights.
    The weights are selected based on the sign of the logits computed
    by the previous weights.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be multiplied.
    weights : torch.Tensor
        The set of weights to be multiplied by.
    max_depth : int
        The maximum depth of the tree.

    Returns
    -------
    logits : torch.Tensor
        The logits computed at each depth. 
    node_indices : torch.Tensor
        The node indices used to select the weights.

    Notes
    -----
    The conditional matmul is computed as follows:
    1. Compute the logits for each batch element at each depth.
    2. Update the node indices based on the logits.
    3. Select the weights based on the node indices.
    4. Compute the logits for each batch element at each depth.
    5. Repeat steps 2-4 until the maximum depth is reached.

    Examples
    --------
    >>> import torch
    >>> from fastff.cmm import conditional_matmul
    >>> tensor = torch.randn(2, 3)
    >>> weights = torch.randn(7, 3)
    >>> max_depth = 4
    >>> logits, node_indices = conditional_matmul(tensor, weights, max_depth)
    >>> logits
    
    """
    batch_size, hidden_size = tensor.size()
    logits = torch.zeros(
        batch_size,
        max_depth,
        device=tensor.device
    )
    node_indices = torch.zeros(
        batch_size,
        max_depth,
        dtype=torch.long,
        device=tensor.device
    )

    max_weight_index = weights.size(0) - 1

    for depth in range(1, max_depth):
        # Clamp indices to be within the bounds of the weigh tensor
        clamped_indices = torch.clamp(
            node_indices[:, depth - 1],
            0,
            max_weight_index,
        )

        # Select weights based on clamped indices
        selected_weights = weights[clamped_indices]

        # Compute logits for each batch element using dot prodduct
        for batch in range(batch_size):
            logits[batch, depth] = torch.dot(
                tensor[batch],
                selected_weights[batch]
            )
        
        # Update and clamp node indices based on lgoits
        updated_indices = 2 * clamped_indices + 1 + (logits[:, depth] > 0).long()
        node_indices[:, depth] = torch.clamp(
            updated_indices,
            0,
            max_weight_index
        )

    # result = None
    # if matmul_tensors and len(matmul_tensors) == 2:
    #     result = torch.matmul(
    #         matmul_tensors[0],
    #         matmul_tensors[1]
        # )

    return logits, node_indices


batch_size, hidden_size, max_depth = 32, 768, 11
tensor = torch.randn(batch_size, hidden_size)
weights = torch.randn(2 * max_depth - 1, hidden_size)

logits, node_indices = conditional_matmul(
    tensor,
    weights,
    max_depth
)

print(logits)
print(node_indices)
