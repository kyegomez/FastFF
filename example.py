import torch
from fastff import FastFeedForward

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
