"""
Preflight checks for local artifacts and setup prerequisites.
"""

import os

import pyarrow.parquet as pq

from nanochat_mlx.common import SetupError, get_base_dir
from nanochat_mlx.dataset import get_split_parquet_files, list_parquet_files


def require_training_data():
    """Ensure there is enough downloaded data to build a train split."""
    train_paths = get_split_parquet_files("train")

    for path in train_paths:
        pf = pq.ParquetFile(path)
        for row_group_idx in range(pf.num_row_groups):
            if pf.metadata.row_group(row_group_idx).num_rows > 0:
                return train_paths

    raise SetupError(
        "Training data is present but the train split is empty. "
        "Download at least 2 non-empty shards: python -m nanochat_mlx.dataset -n 2"
    )


def get_tokenizer_paths(base_dir=None):
    """Return the expected local tokenizer artifact paths."""
    base_dir = get_base_dir() if base_dir is None else base_dir
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return {
        "dir": tokenizer_dir,
        "tokenizer": os.path.join(tokenizer_dir, "tokenizer.pkl"),
        "token_bytes_npy": os.path.join(tokenizer_dir, "token_bytes.npy"),
        "token_bytes_pt": os.path.join(tokenizer_dir, "token_bytes.pt"),
    }


def require_tokenizer():
    """Ensure a trained or imported tokenizer is available locally."""
    paths = get_tokenizer_paths()
    if not os.path.exists(paths["tokenizer"]):
        raise SetupError(
            "Tokenizer not found. Train or import one first:\n"
            "  python -m scripts.tok_train\n"
            "or\n"
            "  python -m scripts.convert_from_hf --repo nanochat-students/base-d20"
        )
    return paths


def get_checkpoint_dir(depth, source="base", base_dir=None):
    """Return the checkpoint directory for a given depth/source pair."""
    base_dir = get_base_dir() if base_dir is None else base_dir
    suffix = "_sft" if source == "sft" else ""
    return os.path.join(base_dir, "mlx_checkpoints", f"d{depth}{suffix}")


def list_checkpoint_weights(depth, source="base", base_dir=None):
    """List model weight files for a depth/source pair."""
    ckpt_dir = get_checkpoint_dir(depth, source=source, base_dir=base_dir)
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".safetensors") and not f.endswith("_optim.safetensors")
    )


def require_checkpoint(depth, source="base", step=None, base_dir=None):
    """Ensure a checkpoint exists and return its weights/meta paths."""
    ckpt_dir = get_checkpoint_dir(depth, source=source, base_dir=base_dir)
    label = "SFT checkpoint" if source == "sft" else "base checkpoint"
    label_title = "SFT checkpoint" if source == "sft" else "Base checkpoint"
    setup_cmd = (
        f"python -m scripts.sft --depth={depth}"
        if source == "sft"
        else f"python -m scripts.train --depth={depth}"
    )

    if not os.path.isdir(ckpt_dir):
        extra = (
            f"{label_title} directory not found for depth {depth}. "
            f"Create one first:\n  {setup_cmd}"
        )
        if source == "base":
            extra += (
                "\nor\n"
                "  python -m scripts.convert_from_hf --repo nanochat-students/base-d20"
            )
        raise SetupError(extra)

    if step is None:
        weight_files = list_checkpoint_weights(depth, source=source, base_dir=base_dir)
        if not weight_files:
            raise SetupError(
                f"No {label} found for depth {depth}. Create one first:\n  {setup_cmd}"
            )
        weights_name = weight_files[-1]
        weights_path = os.path.join(ckpt_dir, weights_name)
        meta_path = weights_path.replace(".safetensors", "_meta.json")
    else:
        weights_path = os.path.join(ckpt_dir, f"step_{step:06d}.safetensors")
        meta_path = os.path.join(ckpt_dir, f"step_{step:06d}_meta.json")
        if not os.path.exists(weights_path):
            raise SetupError(
                f"Checkpoint step {step} was not found in {ckpt_dir}. "
                "Check /checkpoints in the quickstart UI or omit --step to use the latest checkpoint."
            )

    if not os.path.exists(meta_path):
        raise SetupError(
            f"Checkpoint metadata is missing for {weights_path}. "
            "Re-run training or import to regenerate a complete checkpoint."
        )

    return weights_path, meta_path


def count_downloaded_shards(base_dir=None):
    """Return the number of downloaded pretraining shards."""
    base_dir = get_base_dir() if base_dir is None else base_dir
    data_dir = os.path.join(base_dir, "base_data")
    if not os.path.isdir(data_dir):
        return 0
    return len(list_parquet_files(data_dir))
