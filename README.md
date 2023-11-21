[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

Below is a template for a technical README.md file for the implementation of the FastBERT paper. This README provides an overview of the project, including a description, installation instructions, usage guidelines, details on the architecture, and the algorithmic pseudocode.

---

# FastBERT Implementation

## Description
This project implements the feedforward from FastBERT (Fast Bidirectional Encoder Representations from Transformers) model. FastBERT is a BERT-like model optimized for efficient inference, utilizing a novel Conditional Matrix Multiplication (CMM) technique within a Fast Feedforward Network (FFF). The model aims to achieve high performance on natural language processing tasks with significantly reduced computational cost.

## Installation

To use this implementation, ensure you have Python and PyTorch installed. You can install the required dependencies using the following command:

```bash
pip install torch
```

## Usage

To use the FastBERT model, first import the necessary classes and create an instance of the model. You can then pass input data to the model for training or inference. Example usage is as follows:

```python
from fastbert import FastFeedForward
import torch

# Parameters
input_dim = 768
output_dim = 768
depth = 11

# Model initialization
fast_ff = FastFeedForward(input_dim, output_dim, depth)

# Example input (batch_size, seq_len, input_dim)
example_input = torch.randn(32, 128, input_dim)

# Forward pass
output = fast_ff(example_input)
```

## Architecture

FastBERT's architecture starts from the crammedBERT model but replaces the feedforward networks in the transformer encoder layers with fast feedforward networks. Each transformer encoder layer uses multiple FFF trees to compute the intermediate layer outputs, which are then summed to form the final output.

### Key Components:
- **Conditional Matrix Multiplication (CMM)**: A technique used for efficient computation within the FFF.
- **Fast Feedforward Network (FFF)**: Replaces traditional dense feedforward layers, using fewer neurons selectively for inference.
- **Activation Function**: GeLU (Gaussian Error Linear Unit) is used across all nodes in the FFF.

## Algorithmic Pseudocode

### Fast Feedforward Network (FFF)
1. **Initialization**:
   - Define `input_dim`, `output_dim`, and `depth`.
   - Initialize `weights_in` and `weights_out` for CMM.

2. **CMM Function**:
   - For each depth level, compute logits and update node indices.
   - Perform batch-wise matrix-vector multiplication using `einsum`.

3. **Forward Pass**:
   - Apply CMM to input.
   - Apply activation function.
   - Aggregate outputs for each depth using `einsum`.

### Training and Inference
- FastBERT is trained following the crammedBERT procedure, with dropout disabled and a 1-cycle triangular learning rate schedule.
- For inference, FastBERT utilizes the FFF with a reduced number of active neurons, achieving efficient computation.
