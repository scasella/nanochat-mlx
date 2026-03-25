import textwrap

import pytest

from tests.helpers import make_base_env, run_module, write_identity_file, write_parquet_shards


@pytest.fixture(scope="session")
def stub_tasks_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("stub-tasks")
    tasks_dir = root / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "__init__.py").write_text("", encoding="utf-8")

    common_py = textwrap.dedent(
        """
        import random


        class Task:
            def __init__(self, start=0, stop=None, step=1, **kwargs):
                self.start = start
                self.stop = stop
                self.step = step

            @property
            def eval_type(self):
                raise NotImplementedError

            def num_examples(self):
                raise NotImplementedError

            def get_example(self, index):
                raise NotImplementedError

            def __len__(self):
                stop = self.num_examples() if self.stop is None else self.stop
                span = stop - self.start
                return max(0, (span + self.step - 1) // self.step)

            def __getitem__(self, index):
                return self.get_example(self.start + index * self.step)

            def evaluate(self, problem, completion):
                return True


        class TaskMixture(Task):
            def __init__(self, tasks, **kwargs):
                super().__init__(**kwargs)
                self.tasks = tasks
                self.index_map = []
                for task_idx, task in enumerate(tasks):
                    for local_idx in range(len(task)):
                        self.index_map.append((task_idx, local_idx))
                random.Random(42).shuffle(self.index_map)

            def num_examples(self):
                return len(self.index_map)

            def get_example(self, index):
                task_idx, local_idx = self.index_map[index]
                return self.tasks[task_idx][local_idx]


        def render_mc(question, letters, choices):
            query = f"Multiple Choice question: {question}\\n"
            query += "".join([f"- {choice}={letter}\\n" for letter, choice in zip(letters, choices)])
            query += "\\nRespond only with the letter of the correct answer."
            return query
        """
    )
    (tasks_dir / "common.py").write_text(common_py, encoding="utf-8")

    task_template = textwrap.dedent(
        """
        from tasks.common import Task, render_mc


        class _SingleConversationTask(Task):
            eval_kind = "generative"

            def __init__(self, *args, **kwargs):
                super().__init__(**kwargs)

            @property
            def eval_type(self):
                return self.eval_kind

            def num_examples(self):
                return 1

            def get_example(self, index):
                if self.eval_kind == "categorical":
                    letters = ("A", "B", "C", "D")
                    return {
                        "messages": [
                            {"role": "user", "content": render_mc("Pick A", letters, ["A", "B", "C", "D"])},
                            {"role": "assistant", "content": "A"},
                        ],
                        "letters": letters,
                    }
                return {
                    "messages": [
                        {"role": "user", "content": "Say hello."},
                        {"role": "assistant", "content": "Hello."},
                    ],
                    "entry_point": "hello",
                    "test": "def check(fn):\\n    assert True\\n",
                }

            def evaluate(self, problem, completion):
                if self.eval_kind == "categorical":
                    return completion in problem["letters"]
                return True
        """
    )

    modules = {
        "smoltalk.py": "from tasks.common import Task\n\n\nclass SmolTalk(Task):\n    def __init__(self, split, **kwargs):\n        super().__init__(**kwargs)\n    def num_examples(self):\n        return 1\n    def get_example(self, index):\n        return {'messages': [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there'}]}\n",
        "gsm8k.py": task_template + "\n\nclass GSM8K(_SingleConversationTask):\n    pass\n",
        "spellingbee.py": task_template + "\n\nclass SpellingBee(_SingleConversationTask):\n    pass\n\n\nclass SimpleSpelling(_SingleConversationTask):\n    pass\n",
        "humaneval.py": task_template + "\n\nclass HumanEval(_SingleConversationTask):\n    pass\n",
        "mmlu.py": task_template + "\n\nclass MMLU(_SingleConversationTask):\n    eval_kind = 'categorical'\n",
        "arc.py": task_template + "\n\nclass ARC(_SingleConversationTask):\n    eval_kind = 'categorical'\n",
        "customjson.py": textwrap.dedent(
            """
            import json

            from tasks.common import Task


            class CustomJSON(Task):
                def __init__(self, filepath, **kwargs):
                    super().__init__(**kwargs)
                    self.rows = []
                    with open(filepath, "r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                self.rows.append(json.loads(line))

                def num_examples(self):
                    return len(self.rows)

                def get_example(self, index):
                    return {"messages": self.rows[index]}
            """
        ),
    }

    for name, body in modules.items():
        (tasks_dir / name).write_text(body, encoding="utf-8")

    return root


@pytest.fixture(scope="session")
def trained_base_dir(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("nanochat-base")
    write_parquet_shards(
        base_dir,
        [
            [
                "Alpha beta gamma delta. Repeated text for tokenizer and training.",
                "Short training document with simple punctuation.",
            ]
            * 8,
            [
                "Second shard with more text for training coverage and token variety.",
                "Another training example with letters A B C D and numbers 1 2 3 4.",
            ]
            * 8,
            [
                "Validation shard text for tiny smoke tests.",
                "Final shard kept for validation only.",
            ]
            * 4,
        ],
    )

    env = make_base_env(base_dir)

    tok = run_module(
        "scripts.tok_train",
        ["--max-chars", "50000", "--doc-cap", "200", "--vocab-size", "1024"],
        env=env,
    )
    assert tok.returncode == 0, tok.stdout + tok.stderr

    train = run_module(
        "scripts.train",
        [
            "--depth=4",
            "--max-seq-len=64",
            "--device-batch-size=1",
            "--total-batch-size=64",
            "--num-iterations=1",
            "--eval-every=-1",
            "--sample-every=-1",
            "--save-every=1",
            "--memory-limit-gb=2",
            "--use-simple-adamw",
        ],
        env=env,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    return base_dir


@pytest.fixture(scope="session")
def sft_base_dir(trained_base_dir, stub_tasks_dir):
    write_identity_file(trained_base_dir)
    env = make_base_env(trained_base_dir, extra_pythonpath=stub_tasks_dir)

    sft = run_module(
        "scripts.sft",
        [
            "--depth=4",
            "--num-iterations=1",
            "--device-batch-size=1",
            "--max-seq-len=128",
            "--mmlu-epochs=1",
            "--gsm8k-epochs=1",
            "--eval-every=-1",
            "--save-every=1",
            "--memory-limit-gb=2",
            "--load-optimizer=0",
        ],
        env=env,
    )
    assert sft.returncode == 0, sft.stdout + sft.stderr
    return trained_base_dir
