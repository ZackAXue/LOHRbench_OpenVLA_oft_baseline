# LoHRbench: OpenVLA-OFT Baseline

Training code for **OpenVLA-OFT** (Optimized Fine-Tuning) on LoHRbench.
Built on top of the [openvla-oft](https://github.com/moojink/openvla-oft) reference implementation
(Kim et al., 2025 — [arXiv:2502.19645](https://arxiv.org/abs/2502.19645)).

GitHub: [https://github.com/jhqiu21/LOHRbench_OpenVLA_oft_baseline](https://github.com/jhqiu21/LOHRbench_OpenVLA_oft_baseline)

## Repository Structure

```
LOHRbench_OpenVLA_oft_baseline/
├── vla-scripts/
│   ├── finetune.py                    # LoRA fine-tuning entry point
│   ├── deploy.py                      # FastAPI inference server
│   ├── eval_overfit.py
│   └── merge_lora_weights_and_save.py
├── prismatic/                         # Core OpenVLA model code
│   ├── vla/
│   │   ├── constants.py               # ACTION_DIM / NUM_ACTIONS_CHUNK / PROPRIO_DIM
│   │   └── datasets/rlds/             # RLDS dataset pipeline
│   └── models/
│       ├── action_heads.py            # L1RegressionActionHead, DiffusionActionHead
│       └── projectors.py              # ProprioProjector, NoisyActionProjector
├── experiments/robot/                 # Evaluation & inference utils
│   ├── openvla_utils.py
│   ├── libero/
│   └── aloha/
├── SETUP.md
├── LIBERO.md
├── ALOHA.md
└── README.md
```

## Setup

```bash
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch (use the command for your CUDA version)
pip3 install torch torchvision torchaudio

# Install OpenVLA-OFT in editable mode
pip install -e .
```

Attention mode must match the checkpoint. The default installation now uses
standard **Transformers 4.40.1 causal SDPA**; standalone `flash-attn` is not
required. Legacy checkpoints must explicitly select `--attention_mode causal`
or `bidirectional`. The loader checks actual attention behavior and rejects
mismatches. See [ATTENTION.md](ATTENTION.md) for migration, the explicit OFT-fork
bidirectional setup, checkpoint metadata and regression tests.

## Dataset

Download the demonstration dataset from HuggingFace: **[oldTOM/LoHRbench](https://huggingface.co/datasets/oldTOM/LoHRbench)**

OpenVLA-OFT consumes datasets in **RLDS** format. Convert the LoHRbench HDF5 files to
RLDS before fine-tuning, then place the resulting datasets under `--data_root_dir`.

## Training

```bash
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --attention_mode causal \
    --data_root_dir /path/to/lohrbench/rlds \
    --dataset_name lohrbench_<suite> \
    --run_root_dir /path/to/checkpoints \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 150005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity "<your_wandb_entity>" \
    --wandb_project "LoHRbench"
```

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Base model | OpenVLA-7B (`openvla/openvla-7b`) |
| Vision backbone | DINOv2 + SigLIP (PrismaticVisionBackbone, 224×224) |
| Action head | L1 regression MLP (continuous actions) |
| Language conditioning | Llama-2 7B; explicitly selected and checkpoint-persisted causal/bidirectional SDPA |
| FiLM | Disabled (set `--use_film True` for stronger language grounding) |
| Input images | 2 (base camera + wrist camera) |
| Proprio input | True (joint state) |
| Action chunk size | 8 |
| Batch size | 8 per GPU |
| Learning rate | 5e-4 |
| LR schedule | MultiStepLR (10× decay at 100K steps) |
| Optimizer | AdamW |
| Total steps | 150,005 |
| LoRA rank | 32 |
| LoRA dropout | 0.0 |
| Precision | bfloat16 |
| Image augmentation | Random crop (90% area) |
| GPU | 4× NVIDIA A100 40GB|

After training, optionally merge the LoRA adapter into the base model:

```bash
python vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint openvla/openvla-7b \
    --lora_finetuned_checkpoint_dir /path/to/checkpoints/<run-id>
```

## Evaluation

Evaluation is done via the unified evaluation framework in [`LoHRbench/baseline/`](../LoHRbench/baseline/). See the [evaluation README](../LoHRbench/baseline/README.md) for details.

```bash
python baseline/eval.py \
    --policy openvla_delta7 \
    --checkpoint /home/hehehe/LoHRbench/openvla-oft/runs/openvla-7b+lohrbench_rlds+b6+lr-0.0002+lora-r32+dropout-0.0--image_aug--200000_chkpt \
    --step 200000 \
    --base-checkpoint /home/hehehe/LoHRbench/openvla-oft/runs/openvla-7b+lohrbench_rlds+b6+lr-0.0002+lora-r32+dropout-0.0--image_aug--130000_chkpt \
    --merge-lora \
    --center-crop \
    --no-proprio \
    --benchmark-root benchmark/table-top \
    --task-names fruit_placement repackage \
    --num-episodes 100 \
    --max-steps 3100 \
    --camera-width 256 \
    --camera-height 256 \
    --num-workers 1 \
    --use-action-chunking \
    --chunk-size 8 \
    --record \
    --results-dir /home/jinhang/results_openvla_new_h_tool \
    --record-dir /home/jinhang/records_openvla_new_h_tool
```

> `--center-crop` is important when image augmentation was enabled during training
> (random 90% crop at train time → center 90% crop at test time). `--merge-lora` merges the
> LoRA adapter from `--checkpoint` into `--base-checkpoint` before evaluation.

## Acknowledgements

The OpenVLA-OFT implementation is adapted from [openvla-oft](https://github.com/moojink/openvla-oft)
(Kim, Finn, Liang — arXiv:2502.19645) with LoHRbench dataset wiring and the unified LoHRbench evaluation interface.

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
