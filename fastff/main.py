import torch
import torch.nn as nn
import torch.nn.functional as F


class FastFeedForward(nn.Module):
    """
    Fast Feedforward Network (FFF) as described in the FastBERT paper.
    This network uses Conditional Matrix Multiplication (CMM) for efficient inference.

    Attributes:
    - input_dim (int): Dimension of the input layer.
    - output_dim (int): Dimension of the output layer.
    - depth (int): Depth of the FFF tree.
    - activation (torch.nn.Module): Activation function used in the network.
    """

    def __init__(self, input_dim, output_dim, depth):
        super(FastFeedForward, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth

        # Initialize weights for CMM
        self.weights_in = nn.Parameter(torch.randn(2 * depth - 1, input_dim))
        self.weights_out = nn.Parameter(torch.randn(2 * depth - 1, output_dim))

        # Activation function
        self.activation = nn.GELU()

    def cmm(self, inputs):
        """
        Conditional Matrix Multiplication (CMM) operation.
        """
        batch_size, seq_len, _ = inputs.size()
        logits = torch.zeros(batch_size, seq_len, self.depth, device=inputs.device)
        node_indices = torch.zeros(
            batch_size, seq_len, self.depth, dtype=torch.long, device=inputs.device
        )

        for d in range(1, self.depth):
            # Compute logits and update node indices
            logits[:, :, d], node_indices[:, :, d] = self.compute_logits_and_indices(
                inputs, node_indices[:, :, d - 1], d
            )

        return logits, node_indices

    def compute_logits_and_indices(self, inputs, prev_indices, depth_level):
        """
        Compute logits and indices for a given depth level in the CMM tree.
        """
        # Ensure the index does not exceed the size of weights_in
        max_index = self.weights_in.size(0) - 1

        # Clamp previous indices to be within the valid range
        selected_indices = torch.clamp(prev_indices, 0, max_index)
        weight = self.weights_in[selected_indices]

        # Perform the batch-wise matrix-vector multiplication
        logits = torch.einsum("bij,bij->bi", inputs, weight)

        # Calculate new indices and clamp them within the valid range
        new_indices = 2 * selected_indices + 1 + (logits > 0).long()
        new_indices = torch.clamp(new_indices, 0, max_index)

        return logits, new_indices

    def forward(self, inputs):
        """
        Forward pass of the Fast Feedforward Network.
        """
        logits, node_indices = self.cmm(inputs)
        activated_logits = self.activation(logits)

        outputs = torch.zeros_like(inputs)
        for d in range(self.depth):
            # Select the weights for the current depth
            weight_out = self.weights_out[node_indices[:, :, d]]

            # Corrected einsum operation
            # 'bij,bij->bi' performs batch-wise matrix-vector multiplication
            outputs += torch.einsum(
                "bij,bij->bi",
                activated_logits[:, :, d].unsqueeze(-1),
                weight_out.unsqueeze(-1),
            )

        return outputs


# Example Usage
input_dim = 768  # Example input dimension
output_dim = 768  # Output dimension
depth = 11  # Depth of the FFF tree

# Create the Fast Feedforward module
fast_ff = FastFeedForward(input_dim=input_dim, output_dim=output_dim, depth=depth)

# Example input tensor (batch_size, seq_len, input_dim)
example_input = torch.randn(32, 128, input_dim)

# Forward pass
output = fast_ff(example_input)
print(output)
