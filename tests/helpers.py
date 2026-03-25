import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_module(module, args=None, env=None, input_text=None, timeout=180):
    command = [sys.executable, "-m", module]
    if args:
        command.extend(args)

    full_env = os.environ.copy()
    full_env["PYTHONUNBUFFERED"] = "1"
    if env:
        full_env.update(env)

    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=full_env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def make_base_env(base_dir, extra_pythonpath=None):
    env = {"NANOCHAT_BASE_DIR": str(base_dir)}
    if extra_pythonpath:
        existing = os.environ.get("PYTHONPATH")
        parts = [str(extra_pythonpath)]
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def write_parquet_shards(base_dir, shard_texts):
    data_dir = Path(base_dir) / "base_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for idx, texts in enumerate(shard_texts):
        table = pa.table({"text": texts})
        pq.write_table(table, data_dir / f"shard_{idx:05d}.parquet")
    return data_dir


def write_identity_file(base_dir):
    identity_path = Path(base_dir) / "identity_conversations.jsonl"
    conversation = [
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello."},
    ]
    identity_path.write_text(json.dumps(conversation) + "\n", encoding="utf-8")
    return identity_path
