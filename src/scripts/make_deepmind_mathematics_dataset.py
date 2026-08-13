#!/usr/bin/env python3
"""Generate a local Hugging Face copy of DeepMind's Mathematics Dataset.

The default produces roughly half a million training examples plus smaller
interpolation and extrapolation evaluation splits. Counts are per upstream
module, keeping the module distribution balanced.
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import random
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import datasets as hfds
import numpy as np
import tqdm

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'data' / 'deepmind_mathematics'
_MODULE_CACHE = {}


def _make_entropy_fn(level = 0, num_levels = 1):
    lower = level / num_levels
    upper = (level + 1) / num_levels

    def modify_entropy(entropy_range):
        length = entropy_range[1] - entropy_range[0]
        return entropy_range[0] + lower * length, entropy_range[0] + upper * length

    return modify_entropy


def _flatten_modules(nested_modules, prefix = None):
    flattened = {}
    for name, module_or_modules in nested_modules.items():
        full_name = f'{prefix}__{name}' if prefix else name
        if isinstance(module_or_modules, Mapping):
            flattened.update(_flatten_modules(module_or_modules, full_name))
        else:
            flattened[full_name] = module_or_modules
    return flattened


def _modules_for_split(split):
    if split in _MODULE_CACHE:
        return _MODULE_CACHE[split]

    try:
        from mathematics_dataset.modules import modules
    except ImportError as error:
        raise RuntimeError('The generator requires mathematics-dataset. Install the project dependencies '
                           'before running this script.') from error

    if split == 'train':
        nested_modules = modules.train(_make_entropy_fn())
    elif split == 'interpolate':
        nested_modules = modules.test()
    elif split == 'extrapolate':
        nested_modules = modules.test_extra()
    else:
        raise ValueError(f'Unknown split: {split}')
    _MODULE_CACHE[split] = _flatten_modules(nested_modules)
    return _MODULE_CACHE[split]


def _task_seed(seed, split, module_name):
    digest = hashlib.sha256(f'{seed}:{split}:{module_name}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder = 'big')


def _generate_module(task):
    split, module_name, count, seed, output_path = task
    random.seed(seed)
    np.random.seed(seed)

    from mathematics_dataset import generate_settings

    generator = _modules_for_split(split)[module_name]
    dropped = 0
    with open(output_path, 'w', encoding = 'utf-8') as output_file:
        for _ in range(count):
            while True:
                problem = generator()
                question = str(problem.question)
                answer = str(problem.answer)
                if len(question) <= generate_settings.MAX_QUESTION_LENGTH and len(answer) <= generate_settings.MAX_ANSWER_LENGTH:
                    break
                dropped += 1

            output_file.write(json.dumps({
                'split': split,
                'module': module_name,
                'question': question,
                'answer': answer,
                'question_length': len(question),
                'answer_length': len(answer),
            }, ensure_ascii = False) + '\n')

    return split, module_name, count, dropped, output_path


def _generate_tasks(tasks, workers):
    if workers == 1:
        for task in tqdm.tqdm(tasks, desc = 'Generating modules'):
            yield _generate_module(task)
        return

    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers = workers, mp_context = context) as executor:
        futures = [executor.submit(_generate_module, task) for task in tasks]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total = len(futures), desc = 'Generating modules'):
            yield future.result()


def _parse_args():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--output', type = Path, default = DEFAULT_OUTPUT, help = 'Directory passed to datasets.load_from_disk().')
    parser.add_argument('--train-per-module', type = int, default = 10_000, help = 'Training examples generated for every upstream module.')
    parser.add_argument('--eval-per-module', type = int, default = 1_000, help = 'Examples generated per interpolation/extrapolation module.')
    parser.add_argument('--splits', nargs = '+', choices = ('train', 'interpolate', 'extrapolate'), default = ('train', 'interpolate', 'extrapolate'))
    parser.add_argument('--module-filter', default = '', help = 'Optional regular expression matched against flattened module names.')
    parser.add_argument('--workers', type = int, default = min(8, os.cpu_count() or 1))
    parser.add_argument('--seed', type = int, default = 696969)
    parser.add_argument('--overwrite', action = 'store_true', help = 'Replace an existing generated dataset.')
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.train_per_module < 1 or args.eval_per_module < 1:
        raise ValueError('Per-module example counts must be positive.')
    if args.workers < 1:
        raise ValueError('--workers must be positive.')

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents = True, exist_ok = True)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f'{output_path} already exists; pass --overwrite to replace it.')
        shutil.rmtree(output_path)

    staging_path = output_path.with_name(f'.{output_path.name}.incomplete')
    if staging_path.exists():
        shutil.rmtree(staging_path)

    module_pattern = re.compile(args.module_filter)
    tasks = []
    with tempfile.TemporaryDirectory(prefix = 'deepmind-mathematics-', dir = output_path.parent) as temporary_dir:
        temporary_dir = Path(temporary_dir)
        for split in args.splits:
            module_names = sorted(name for name in _modules_for_split(split) if module_pattern.search(name))
            if not module_names:
                raise ValueError(f"No modules in split '{split}' matched {args.module_filter!r}.")

            count = args.train_per_module if split == 'train' else args.eval_per_module
            for module_index, module_name in enumerate(module_names):
                shard_path = temporary_dir / f'{split}-{module_index:04d}.jsonl'
                tasks.append((
                    split,
                    module_name,
                    count,
                    _task_seed(args.seed, split, module_name),
                    str(shard_path),
                ))

        print(f'Generating {sum(task[2] for task in tasks):,} examples across {len(tasks)} module shards with {args.workers} workers.')
        completed_shards = {split: [] for split in args.splits}
        for split, module_name, count, dropped, shard_path in _generate_tasks(tasks, args.workers):
            completed_shards[split].append(shard_path)
            if dropped:
                print(f'{split}/{module_name}: retained {count:,}, dropped {dropped:,} over-length samples')

        data_files = {split: sorted(shard_paths) for split, shard_paths in completed_shards.items()}
        dataset = hfds.load_dataset(
            'json',
            data_files = data_files,
            cache_dir = str(temporary_dir / 'huggingface-cache'),
        )
        save_kwargs = {'num_proc': args.workers} if args.workers > 1 else {}
        dataset.save_to_disk(str(staging_path), **save_kwargs)
        os.replace(staging_path, output_path)

    print(f'Saved {dataset} to {output_path}')


if __name__ == '__main__':
    main()
