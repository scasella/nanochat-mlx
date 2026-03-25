"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""

import argparse
import os
import time

import numpy as np

from nanochat_mlx.common import SetupError, get_base_dir, print_setup_error
from nanochat_mlx.dataset import parquets_iter_batched
from nanochat_mlx.preflight import require_training_data
from nanochat_mlx.tokenizer import RustBPETokenizer


def build_parser():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2_000_000_000,
        help="Maximum characters to train on (default: 10B)",
    )
    parser.add_argument(
        "--doc-cap",
        type=int,
        default=10_000,
        help="Maximum characters per document (default: 10,000)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        help="Vocabulary size (default: 32768 = 2^15)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"max_chars: {args.max_chars:,}")
    print(f"doc_cap: {args.doc_cap:,}")
    print(f"vocab_size: {args.vocab_size:,}")

    try:
        require_training_data()

        def text_iterator():
            """
            1) Flatten the batches into a single iterator
            2) Crop every document to args.doc_cap characters
            3) Break when we've seen args.max_chars characters
            """
            nchars = 0
            for batch in parquets_iter_batched(split="train"):
                for doc in batch:
                    doc_text = doc[:args.doc_cap] if len(doc) > args.doc_cap else doc
                    nchars += len(doc_text)
                    yield doc_text
                    if nchars > args.max_chars:
                        return

        t0 = time.time()
        tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
        t1 = time.time()
        print(f"Training time: {t1 - t0:.2f}s")

        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        tokenizer.save(tokenizer_dir)

        test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == test_text

        vocab_size = tokenizer.get_vocab_size()
        special_set = set(tokenizer.get_special_tokens())
        token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
        token_bytes = []
        for token_id in range(vocab_size):
            token_str = token_strings[token_id]
            if token_str in special_set:
                token_bytes.append(0)
            else:
                token_bytes.append(len(token_str.encode("utf-8")))
        token_bytes = np.array(token_bytes, dtype=np.int32)
        token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.npy")
        np.save(token_bytes_path, token_bytes)
        print(f"Saved token_bytes to {token_bytes_path}")
        return 0
    except SetupError as exc:
        print_setup_error(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
