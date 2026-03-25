"""
Evaluate an MLX chat (SFT) model.

Supports both generative evaluation (GSM8K, HumanEval, SpellingBee) and
categorical evaluation (ARC, MMLU). No DDP — single-device only.

Usage:
    python -m scripts.chat_eval --depth=12
    python -m scripts.chat_eval --depth=12 -a MMLU
    python -m scripts.chat_eval --depth=12 -a "GSM8K|MMLU"
    python -m scripts.chat_eval --depth=12 --source=base
"""

import argparse
from functools import partial

import mlx.core as mx

from nanochat_mlx.common import SetupError, print0, print_setup_error, set_memory_limit
from scripts.chat import load_model
from nanochat_mlx.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee


def run_generative_eval(task_object, tokenizer, engine, num_samples, max_new_tokens,
                        temperature, top_k, max_problems=None):
    """Generative evaluation: sample completions and check correctness."""
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    for i in range(num_problems):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)

        # Get completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode completions
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(r[prefix_length:]) for r in results]

        # Evaluate
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)
        print(f"\r\033[K{num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")
    return num_passed / total if total > 0 else 0.0


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):
    """Categorical evaluation: check logits for correct answer letter."""
    bos = tokenizer.get_bos_token_id()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_batches = -(-num_problems // batch_size)  # ceil div

    letter_to_id_cache = {}
    num_passed, total = 0, 0

    for i in range(num_batches):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_tensor = mx.array(padded, dtype=mx.int32)

        # Forward pass
        logits = model(prompt_tensor)  # (B, T, V)
        mx.eval(logits)

        # Check each problem
        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded = tokenizer.encode(letter)
                    assert len(encoded) == 1, f"Letter '{letter}' must encode to single token"
                    letter_to_id_cache[letter] = encoded[0]
                letter_ids.append(letter_to_id_cache[letter])

            answer_pos = answer_positions[idx]
            answer_logits = logits[idx, answer_pos]  # (V,)
            focus_logits = answer_logits[mx.array(letter_ids, dtype=mx.int32)]
            argmax_idx = mx.argmax(focus_logits).item()
            predicted_letter = letters[argmax_idx]

            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")
    return num_passed / total if total > 0 else 0.0


def run_chat_eval(task_name, model, tokenizer, engine,
                  batch_size=1, num_samples=1, max_new_tokens=512,
                  temperature=0.0, top_k=50, max_problems=None):
    """Run evaluation for a single task."""
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()

    if task_object.eval_type == 'generative':
        acc = run_generative_eval(
            task_object, tokenizer, engine, num_samples, max_new_tokens,
            temperature, top_k, max_problems=max_problems,
        )
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(
            task_object, tokenizer, model, batch_size, max_problems=max_problems,
        )
    else:
        raise ValueError(f"Unsupported eval type: {task_object.eval_type}")
    return acc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate MLX chat model")
    parser.add_argument('-d', '--depth', type=int, default=12, help='Model depth')
    parser.add_argument('-i', '--source', type=str, default='sft', help='Checkpoint source: sft|base')
    parser.add_argument('-a', '--task-name', type=str, default=None,
                        help='Task name (default=all). Use | to split multiple tasks.')
    parser.add_argument('-s', '--step', type=int, default=None, help='Checkpoint step (default: latest)')
    parser.add_argument('-t', '--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512, help='Max new tokens')
    parser.add_argument('-n', '--num-samples', type=int, default=1, help='Number of samples per problem')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical eval')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems per task')
    parser.add_argument('--memory-limit-gb', type=float, default=16.0, help='MLX memory limit')
    args = parser.parse_args(argv)

    set_memory_limit(args.memory_limit_gb)

    # Load model
    from nanochat_mlx.tokenizer import get_tokenizer
    model = load_model(depth=args.depth, step=args.step, source=args.source)
    tokenizer = get_tokenizer()
    engine = Engine(model, tokenizer)

    # Tasks
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    baseline_accuracies = {
        'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
        'GSM8K': 0.0, 'HumanEval': 0.0, 'SpellingBee': 0.0,
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # Run evaluations
    results = {}
    for task_name in task_names:
        print0(f"\n--- {task_name} ---")
        acc = run_chat_eval(
            task_name, model, tokenizer, engine,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
        )
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # ChatCORE metric
    if all(t in results for t in all_tasks):
        centered_sum = 0
        for t, acc in results.items():
            baseline = baseline_accuracies.get(t, 0.0)
            centered_sum += (acc - baseline) / (1.0 - baseline)
        chatcore = centered_sum / len(results)
        print0(f"\nChatCORE: {chatcore:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as exc:
        print_setup_error(exc)
        raise SystemExit(2)
