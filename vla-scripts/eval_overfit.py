"""
eval_overfit.py

Evaluate a fine-tuned OpenVLA-OFT checkpoint on its training data to verify overfitting.
Loads the RLDS dataset directly, runs model inference on every timestep, and computes
per-step / per-dimension error metrics with visualization.

Usage:
    conda run --no-capture-output -n lohrbench_openvla python vla-scripts/eval_overfit.py \
        --pretrained_checkpoint "runs/openvla-7b+lohrbench_rlds+b2+lr-0.0005+lora-r32+dropout-0.0--overfit_test" \
        --data_root_dir "/data1/LoHRbench_rlds/lohrbench_rlds_micro_short" \
        --dataset_name "lohrbench_rlds" \
        --unnorm_key "lohrbench_rlds"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch  # Must be imported before tensorflow to avoid CUDA symbol conflicts
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from peft import PeftModel
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# -- Project imports --
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper", "dim7"]


def _detect_and_patch_action_dim(checkpoint_dir: str):
    """Detect the actual ACTION_DIM from the saved action head checkpoint and patch constants if needed."""
    import prismatic.vla.constants as const_mod
    import prismatic.models.action_heads as ah_mod

    # Find the action head checkpoint
    action_head_files = [f for f in os.listdir(checkpoint_dir) if "action_head" in f and f.endswith(".pt")]
    if not action_head_files:
        return ACTION_DIM
    action_head_path = os.path.join(checkpoint_dir, action_head_files[0])

    # Load state dict and infer output dim from fc2 bias
    state_dict = torch.load(action_head_path, weights_only=True, map_location="cpu")
    # Handle DDP prefix
    keys = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    actual_action_dim = keys["model.fc2.bias"].shape[0]

    if actual_action_dim != ACTION_DIM:
        print(f"  [patch] Checkpoint trained with ACTION_DIM={actual_action_dim}, constants say {ACTION_DIM}. Patching.")
        const_mod.ACTION_DIM = actual_action_dim
        ah_mod.ACTION_DIM = actual_action_dim
        # Also patch in modeling_prismatic since it imports ACTION_DIM at load time
        import prismatic.extern.hf.modeling_prismatic as mp_mod
        mp_mod.ACTION_DIM = actual_action_dim
    return actual_action_dim


# Lazy project imports (after potential patching)
def _get_project_imports():
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_vla,
        resize_image_for_policy,
    )
    return get_action_head, get_processor, get_vla, resize_image_for_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_checkpoint", type=str, required=True)
    p.add_argument("--attention_mode", choices=["causal", "bidirectional"], default=None)
    p.add_argument("--data_root_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="lohrbench_rlds")
    p.add_argument("--unnorm_key", type=str, default="lohrbench_rlds")
    p.add_argument("--use_l1_regression", type=bool, default=True)
    p.add_argument("--use_diffusion", type=bool, default=False)
    p.add_argument("--use_film", type=bool, default=False)
    p.add_argument("--num_images_in_input", type=int, default=1)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--load_in_8bit", type=bool, default=False)
    p.add_argument("--load_in_4bit", type=bool, default=False)
    p.add_argument("--center_crop", type=bool, default=False)
    p.add_argument("--merge_lora", action="store_true", default=False,
                   help="Load and merge LoRA adapter from checkpoint. Required for unmerged checkpoints.")
    p.add_argument("--output_dir", type=str, default=None, help="Where to save plots. Defaults to checkpoint dir.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading: read raw RLDS dataset and extract (image, instruction, action)
# ---------------------------------------------------------------------------
def load_rlds_episodes(data_root_dir: str, dataset_name: str):
    """Load all episodes from the RLDS dataset, returning list of episodes.
    Each episode is a list of dicts with keys: base_rgb, hand_rgb, qpos, action, language_instruction.
    """
    data_dir = os.path.join(data_root_dir, dataset_name)
    # tfds.builder_from_directory expects the versioned subdirectory containing dataset_info.json
    # Try the base dir first, then look for a versioned subdir (e.g. 0.1.0/)
    if not os.path.exists(os.path.join(data_dir, "dataset_info.json")):
        subdirs = sorted(os.listdir(data_dir))
        for sd in subdirs:
            candidate = os.path.join(data_dir, sd)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "dataset_info.json")):
                data_dir = candidate
                break
    builder = tfds.builder_from_directory(data_dir)
    ds = builder.as_dataset(split="train")

    episodes = []
    for episode in ds:
        steps = []
        for step in episode["steps"]:
            steps.append({
                "base_rgb": step["observation"]["base_rgb"].numpy(),  # (H, W, 3) uint8
                "hand_rgb": step["observation"]["hand_rgb"].numpy(),
                "qpos": step["observation"]["qpos"].numpy(),  # (9,)
                "action": step["action"].numpy(),  # (8,)
                "language_instruction": step["language_instruction"].numpy().decode("utf-8"),
            })
        episodes.append(steps)
    return episodes


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def prepare_image(img_np: np.ndarray) -> Image.Image:
    """Resize uint8 image to 224x224 and return PIL Image."""
    if img_np.shape[:2] != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
        _resize = globals().get("resize_image_for_policy")
        if _resize is None:
            from experiments.robot.openvla_utils import resize_image_for_policy as _resize
        img_np = _resize(img_np, OPENVLA_IMAGE_SIZE)
    return Image.fromarray(img_np).convert("RGB")


@torch.inference_mode()
def predict_actions_for_step(vla, processor, action_head, step_data, cfg):
    """Run model inference for a single timestep. Returns predicted actions (NUM_ACTIONS_CHUNK, ACTION_DIM) numpy."""
    instruction = step_data["language_instruction"]
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    primary_image = prepare_image(step_data["base_rgb"])
    inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

    if action_head is None:
        actions, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
    else:
        actions, _ = vla.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            action_head=action_head,
            use_film=cfg.use_film,
        )
    return np.array(actions)  # (NUM_ACTIONS_CHUNK, ACTION_DIM)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred_all, gt_all):
    """
    pred_all: (N, ACTION_DIM) predicted first-step actions
    gt_all:   (N, ACTION_DIM) ground-truth actions (first 7 dims)
    Returns dict of metrics.
    """
    err = pred_all - gt_all
    mse_per_dim = np.mean(err ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(err), axis=0)
    mse = np.mean(err ** 2)
    mae = np.mean(np.abs(err))
    max_ae = np.max(np.abs(err))
    return {
        "mse": float(mse),
        "mae": float(mae),
        "max_ae": float(max_ae),
        "mse_per_dim": mse_per_dim.tolist(),
        "mae_per_dim": mae_per_dim.tolist(),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_results(pred_all, gt_all, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    N = pred_all.shape[0]
    dim = pred_all.shape[1]
    timesteps = np.arange(N)

    # --- 1. Per-step L1 error curve ---
    per_step_mae = np.mean(np.abs(pred_all - gt_all), axis=1)
    per_step_mse = np.mean((pred_all - gt_all) ** 2, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, per_step_mae, label="MAE", color="tab:blue")
    ax.plot(timesteps, per_step_mse, label="MSE", color="tab:red")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Error")
    ax.set_title(f"Per-step error (overall MAE={metrics['mae']:.5f}, MSE={metrics['mse']:.5f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_step_error.png"), dpi=150)
    plt.close(fig)

    # --- 2. Per-dimension bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(dim)
    labels = ACTION_LABELS[:dim]

    axes[0].bar(x, metrics["mae_per_dim"], color="tab:blue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_title("MAE per action dimension")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, metrics["mse_per_dim"], color="tab:red", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].set_title("MSE per action dimension")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_dim_error.png"), dpi=150)
    plt.close(fig)

    # --- 3. Predicted vs GT overlay per dimension ---
    fig, axes = plt.subplots(dim, 1, figsize=(12, 3 * dim), sharex=True)
    if dim == 1:
        axes = [axes]
    for d in range(dim):
        axes[d].plot(timesteps, gt_all[:, d], label="GT", color="tab:green", linewidth=1.5)
        axes[d].plot(timesteps, pred_all[:, d], label="Pred", color="tab:orange", linewidth=1.5, linestyle="--")
        axes[d].set_ylabel(labels[d])
        axes[d].legend(loc="upper right", fontsize=8)
        axes[d].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Predicted vs Ground Truth actions", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pred_vs_gt.png"), dpi=150)
    plt.close(fig)

    # --- 4. Error heatmap (timestep x dim) ---
    abs_err = np.abs(pred_all - gt_all)  # (N, dim)
    fig, ax = plt.subplots(figsize=(10, max(4, N * 0.08)))
    im = ax.imshow(abs_err.T, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_yticks(np.arange(dim))
    ax.set_yticklabels(labels[:dim])
    ax.set_xlabel("Timestep")
    ax.set_title("Absolute error heatmap")
    fig.colorbar(im, ax=ax, label="|pred - gt|")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = parse_args()

    output_dir = cfg.output_dir or os.path.join(cfg.pretrained_checkpoint, "eval_overfit")

    # Detect actual ACTION_DIM from checkpoint and patch constants before loading model
    actual_action_dim = _detect_and_patch_action_dim(cfg.pretrained_checkpoint)

    # Now import project modules (after patching)
    get_action_head, get_processor, get_vla, resize_image_for_policy = _get_project_imports()
    # Make resize_image_for_policy available to prepare_image
    globals()["resize_image_for_policy"] = resize_image_for_policy

    print("=" * 70)
    print("Overfit Evaluation")
    print("=" * 70)
    print(f"Checkpoint : {cfg.pretrained_checkpoint}")
    print(f"Dataset    : {cfg.dataset_name}")
    print(f"ACTION_DIM : {actual_action_dim}")
    print(f"CHUNK_SIZE : {NUM_ACTIONS_CHUNK}")
    print(f"Output dir : {output_dir}")
    print()

    # --- Load data ---
    print("Loading RLDS dataset...")
    episodes = load_rlds_episodes(cfg.data_root_dir, cfg.dataset_name)
    total_steps = sum(len(ep) for ep in episodes)
    print(f"  {len(episodes)} episode(s), {total_steps} total steps")

    # --- Load model + merge LoRA ---
    print("Loading VLA model...")
    vla = get_vla(cfg)
    processor = get_processor(cfg)

    # LoRA merge (only for unmerged checkpoints — pass --merge_lora)
    if cfg.merge_lora:
        ckpt_dir = Path(cfg.pretrained_checkpoint)
        adapter_dir = ckpt_dir / "lora_adapter"
        if adapter_dir.exists():
            print(f"Loading LoRA adapter from {adapter_dir} ...")
            vla = PeftModel.from_pretrained(vla, str(adapter_dir))
            print("Merging LoRA weights (merge_and_unload) ...")
            vla = vla.merge_and_unload()
            vla.eval()
            print("LoRA merged successfully.")
        else:
            print("WARNING: --merge_lora set but no lora_adapter/ found. Proceeding without merge.")

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, vla.llm_dim)

    # --- Run inference ---
    print("Running inference on all timesteps...")
    all_pred_first = []  # first action in chunk
    all_gt_first = []    # ground truth action
    all_pred_chunks = [] # full predicted chunks

    step_idx = 0
    for ep_i, episode in enumerate(episodes):
        for t, step_data in enumerate(episode):
            gt_action = step_data["action"]  # (8,)
            # Match GT dims to model output dims
            gt_action = gt_action[:actual_action_dim]

            pred_chunk = predict_actions_for_step(vla, processor, action_head, step_data, cfg)
            # pred_chunk shape: (NUM_ACTIONS_CHUNK, actual_action_dim)
            pred_first = pred_chunk[0]  # first action in chunk

            all_pred_first.append(pred_first)
            all_gt_first.append(gt_action)
            all_pred_chunks.append(pred_chunk)

            step_idx += 1
            if step_idx % 10 == 0 or step_idx == total_steps:
                # Print running MAE
                _pred = np.array(all_pred_first)
                _gt = np.array(all_gt_first)
                running_mae = np.mean(np.abs(_pred - _gt))
                print(f"  [{step_idx}/{total_steps}] running MAE={running_mae:.6f}")

    all_pred_first = np.array(all_pred_first)  # (N, 7)
    all_gt_first = np.array(all_gt_first)      # (N, 7)

    # --- Compute metrics ---
    metrics = compute_metrics(all_pred_first, all_gt_first)

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  MSE (overall)   : {metrics['mse']:.6f}")
    print(f"  MAE (overall)   : {metrics['mae']:.6f}")
    print(f"  Max |error|     : {metrics['max_ae']:.6f}")
    print()
    print("  Per-dimension MAE:")
    for d in range(len(metrics["mae_per_dim"])):
        label = ACTION_LABELS[d] if d < len(ACTION_LABELS) else f"dim{d}"
        print(f"    {label:>10s} : {metrics['mae_per_dim'][d]:.6f}")
    print()
    print("  Per-dimension MSE:")
    for d in range(len(metrics["mse_per_dim"])):
        label = ACTION_LABELS[d] if d < len(ACTION_LABELS) else f"dim{d}"
        print(f"    {label:>10s} : {metrics['mse_per_dim'][d]:.6f}")

    # --- Save metrics JSON ---
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # --- Save raw predictions ---
    np.savez(
        os.path.join(output_dir, "predictions.npz"),
        pred_first=all_pred_first,
        gt_first=all_gt_first,
        pred_chunks=np.array(all_pred_chunks),
    )
    print(f"Predictions saved to {os.path.join(output_dir, 'predictions.npz')}")

    # --- Plot ---
    plot_results(all_pred_first, all_gt_first, metrics, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
