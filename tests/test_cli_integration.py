import pytest

from tests.helpers import make_base_env, run_module, write_parquet_shards


def test_tok_train_rejects_empty_dataset(tmp_path):
    env = make_base_env(tmp_path)
    result = run_module("scripts.tok_train", env=env)

    assert result.returncode == 2
    assert "No dataset shards found" in result.stderr
    assert "python -m nanochat_mlx.dataset -n 2" in result.stderr


def test_tok_train_rejects_single_shard(tmp_path):
    write_parquet_shards(tmp_path, [["only one shard means no train split"]])
    env = make_base_env(tmp_path)

    result = run_module("scripts.tok_train", env=env)

    assert result.returncode == 2
    assert "at least 2 shards" in result.stderr
    assert "reserved for validation" in result.stderr


@pytest.mark.parametrize(
    ("module", "args", "expected"),
    [
        ("scripts.train", ["--depth=4", "--num-iterations=1"], "No dataset shards found"),
        ("scripts.sft", ["--depth=4", "--num-iterations=1"], "Base checkpoint directory not found"),
        ("scripts.chat", ["--depth=4", "-p", "hello"], "Base checkpoint directory not found"),
        ("scripts.chat_eval", ["--depth=4", "-x", "1"], "SFT checkpoint directory not found"),
    ],
)
def test_cli_clean_cache_failure_messages(tmp_path, module, args, expected):
    env = make_base_env(tmp_path)
    result = run_module(module, args, env=env)

    assert result.returncode == 2
    assert expected in result.stderr


def test_base_pipeline_cli_smoke(trained_base_dir):
    env = make_base_env(trained_base_dir)

    prompt = run_module(
        "scripts.chat",
        [
            "--depth=4",
            "--source=base",
            "-p",
            "The capital of France is",
            "--max-tokens=8",
            "--temperature=0",
            "--top-k=1",
            "--memory-limit-gb=2",
        ],
        env=env,
    )
    assert prompt.returncode == 0, prompt.stdout + prompt.stderr
    assert "Loading checkpoint:" in prompt.stdout


def test_sft_and_eval_cli_smoke(sft_base_dir, stub_tasks_dir):
    env = make_base_env(sft_base_dir, extra_pythonpath=stub_tasks_dir)

    interactive = run_module(
        "scripts.chat",
        [
            "--depth=4",
            "--source=sft",
            "--interactive",
            "--max-tokens=8",
            "--temperature=0",
            "--top-k=1",
            "--memory-limit-gb=2",
        ],
        env=env,
        input_text="hello\nquit\n",
    )
    assert interactive.returncode == 0, interactive.stdout + interactive.stderr
    assert "Interactive chat mode" in interactive.stdout

    generative_eval = run_module(
        "scripts.chat_eval",
        [
            "--depth=4",
            "--source=sft",
            "-a",
            "SpellingBee",
            "-x",
            "1",
            "--memory-limit-gb=2",
        ],
        env=env,
    )
    assert generative_eval.returncode == 0, generative_eval.stdout + generative_eval.stderr
    assert "SpellingBee accuracy:" in generative_eval.stdout

    categorical_eval = run_module(
        "scripts.chat_eval",
        [
            "--depth=4",
            "--source=sft",
            "-a",
            "MMLU",
            "-x",
            "1",
            "--memory-limit-gb=2",
        ],
        env=env,
    )
    assert categorical_eval.returncode == 0, categorical_eval.stdout + categorical_eval.stderr
    assert "MMLU accuracy:" in categorical_eval.stdout
