"""
SFT dataloader for MLX training.

BOS-aligned best-fit packing with padding for conversation data.
Port of the SFT dataloader from scripts/chat_sft.py.
"""

import mlx.core as mx


def sft_dataloader_bos_bestfit(dataset, tokenizer, B, T, buffer_size=100, num_iterations=-1):
    """
    BOS-aligned dataloader for SFT with best-fit PAD packing.

    Each row starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit. When no conversation fits,
    the row is padded with BOS tokens (instead of cropping).
    Padding positions have targets=-1 (ignored in loss).

    Args:
        dataset: TaskMixture — calls dataset[cursor] to get conversations
        tokenizer: tokenizer with render_conversation() method
        B: batch size (number of rows)
        T: sequence length (max_seq_len)
        buffer_size: conversation buffer size for best-fit packing
        num_iterations: max iterations (-1 = full epoch)

    Yields:
        (inputs, targets, is_last, progress) where:
        - inputs: mx.array (B, T) int32
        - targets: mx.array (B, T) int32, -1 for padding positions
        - is_last: bool, True when epoch is complete or iteration limit reached
        - progress: float 0→1 over the course of the epoch
    """
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = T + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    conv_buffer = []
    cursor = 0
    consumed = 0

    def refill_buffer():
        nonlocal cursor
        while len(conv_buffer) < buffer_size and cursor < dataset_size:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            conv_buffer.append(ids)
            cursor += 1

    it = 0
    while True:
        rows = []
        row_lengths = []  # actual content length (before padding) for each row

        for _ in range(B):
            row = []
            content_len = row_capacity  # assume full unless padded

            while len(row) < row_capacity:
                refill_buffer()

                if len(conv_buffer) == 0:
                    # Dataset exhausted and buffer empty — pad remainder
                    content_len = len(row)
                    row.extend([bos_token] * (row_capacity - len(row)))
                    break

                remaining = row_capacity - len(row)

                # Best-fit: find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, conv in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    conv = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    consumed += 1
                else:
                    # No conversation fits — pad the remainder
                    content_len = len(row)
                    row.extend([bos_token] * (row_capacity - len(row)))
                    break

            row_lengths.append(content_len)
            rows.append(row[:row_capacity])

        it += 1
        progress = min(consumed / dataset_size, 1.0)
        is_last = consumed >= dataset_size
        if 0 < num_iterations <= it:
            is_last = True

        # Build input rows and target rows with padding mask
        input_rows = [row[:-1] for row in rows]
        target_rows = []
        for row, clen in zip(rows, row_lengths):
            target_row = list(row[1:])  # shift by 1 for next-token prediction
            if clen < row_capacity:
                # Mask targets from content boundary onwards
                mask_start = max(clen - 1, 0)
                for j in range(mask_start, T):
                    target_row[j] = -1
            target_rows.append(target_row)

        inputs = mx.array(input_rows, dtype=mx.int32)
        targets = mx.array(target_rows, dtype=mx.int32)

        yield inputs, targets, is_last, progress

        if is_last:
            break
