# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/ZackAXue/LOHRbench_OpenVLA_oft_baseline.git
cd LOHRbench_OpenVLA_oft_baseline
pip install -e .

```

This LoHRbench fork uses standard Transformers 4.40.1 and explicit SDPA attention.
Standalone `flash-attn` is not required. Follow [ATTENTION.md](ATTENTION.md) to
select a checkpoint-compatible mode and migrate an existing OFT-fork environment.
