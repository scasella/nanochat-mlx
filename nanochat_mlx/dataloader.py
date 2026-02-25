"""
Dataloader for MLX training.

BOS-aligned best-fit packing, adapted from nanochat/dataloader.py.
Reuses the existing tokenizer and dataset infrastructure.
No DDP, no torch tensors — yields mx.array directly.
"""

import mlx.core as mx
import pyarrow.parquet as pq

from nanochat_mlx.dataset import list_parquet_files


def _document_batches(split, resume_state_dict=None, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files (no DDP)."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run: python -m nanochat_mlx.dataset -n 8?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)

            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                rg_idx = resume_rg_idx + 1
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = 0

            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += 1
            pq_idx += 1
        first_pass = False
        epoch += 1


def dataloader_bos_bestfit(tokenizer, B, T, split, resume_state_dict=None, buffer_size=1000):
    """
    BOS-aligned dataloader with Best-Fit Cropping for MLX.

    Yields (inputs, targets) as mx.array of shape (B, T), dtype int32.
    """
    assert split in ["train", "val"]

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    while True:
        # Build B rows of length (T+1)
        all_rows = []
        for row_idx in range(B):
            row = []
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Best-fit: find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
                    pos += remaining

            all_rows.append(row[:row_capacity])

        # Convert to mx.array
        row_array = mx.array(all_rows, dtype=mx.int32)  # (B, T+1)
        inputs = row_array[:, :-1]   # (B, T)
        targets = row_array[:, 1:]   # (B, T)

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        yield inputs, targets, state_dict


def dataloader_bos_bestfit_no_state(tokenizer, B, T, split, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, _ in dataloader_bos_bestfit(tokenizer, B, T, split, **kwargs):
        yield inputs, targets
