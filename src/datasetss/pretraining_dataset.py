import os
import re
import torch
import functools
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from transformers import AutoTokenizer
from lib.accelerator import AcumenAccelerator
from torch.utils.data import DataLoader
from lib.dataset_extra import AcumenDataset
from torch.utils.data.distributed import DistributedSampler
from datasets.fingerprint import Hasher
from filelock import FileLock

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import datasets as hfds

AVG_TOKS_PER_CHUNK = 149  # computed separately, gpt-2 tokens
TOKS_PER_PARAM = 19  # chinchilla scaling
BYTE_VOCAB_SIZE = 256
PAD_TOKEN = '<pad>'
EOS_TOKEN = '<eos>'
CHUNK_CACHE_VERSION = 1

comments_re = re.compile(r'//.*?$|/\*.*?\*/', re.DOTALL | re.MULTILINE)


def strip_comments(text):
    return comments_re.sub('', text)


def _collate(list_of_dicts):
    if not len(list_of_dicts):
        raise Exception("What the fuck are you doing burv?")

    keys = list_of_dicts[0].keys()
    return {key: [x[key] for x in list_of_dicts] for key in keys}


def percent_non_ascii(text):
    return sum(1 for c in text if ord(c) >= 128) / len(text)


def remove_non_ascii(text):
    return ''.join(c for c in text if ord(c) < 128)


def split_chunks(example, chunk_size):
    text_chunks = [text[i:i + chunk_size] for text in example["text"] for i in range(0, len(text), chunk_size)]
    texts = []
    for t in text_chunks:
        if percent_non_ascii(t) <= 0.3:
            texts.append(remove_non_ascii(t))
    return {'text': texts}


def _chunk_cache_path(dataset, chunking_fn, chunk_size, cache_root = None):
    """Return a stable cache path for a chunked version of ``dataset``."""
    cache_key = {
        'dataset_fingerprint': dataset._fingerprint,
        'chunking_fingerprint': Hasher.hash(chunking_fn),
        'chunk_size': chunk_size,
        'chunk_cache_version': CHUNK_CACHE_VERSION,
    }
    digest = hashlib.sha256(json.dumps(cache_key, sort_keys = True).encode('utf-8')).hexdigest()
    cache_root = Path(cache_root or hfds.config.HF_DATASETS_CACHE)
    return cache_root / 'looping-bootstrap' / 'pretraining-chunks' / digest


def _load_or_create_chunked_dataset(dataset, chunking_fn, chunk_size, map_num_proc):
    cache_path = _chunk_cache_path(dataset, chunking_fn, chunk_size)
    cache_path.parent.mkdir(parents = True, exist_ok = True)

    # Training processes commonly start together. Only one of them should run
    # the expensive map while the others wait and then load the finished cache.
    with FileLock(f'{cache_path}.lock'):
        if cache_path.is_dir():
            print(f"::: Loading chunked dataset from cache: {cache_path}")
            return hfds.load_from_disk(str(cache_path))

        print(f"::: Building chunked dataset cache: {cache_path}")
        chunked_dataset = dataset.map(
            chunking_fn,
            batched = True,
            batch_size = 2048,
            num_proc = map_num_proc,
        )

        temporary_path = Path(tempfile.mkdtemp(prefix = f'.{cache_path.name}.', dir = cache_path.parent))
        try:
            chunked_dataset.save_to_disk(str(temporary_path))
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)

        return chunked_dataset


def _coerce_token_ids(encoded):
    if hasattr(encoded, 'ids'):
        encoded = encoded.ids

    if torch.is_tensor(encoded):
        encoded = encoded.tolist()

    return [int(token_id) for token_id in encoded]


def _shift_right(input_ids, attention_mask, bos_token_id, pad_token_id):
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)

    shifted_input_ids = input_ids.new_full(input_ids.shape, pad_token_id)
    shifted_input_ids[:, 0] = bos_token_id
    shifted_input_ids[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, pad_token_id)
    return shifted_input_ids, labels


def _batch_tokenize_texts(texts, tokenizer):
    input_ids = []
    attention_masks = []

    for text in texts:
        token_ids = _coerce_token_ids(tokenizer.encode(text, add_special_tokens = False))
        token_ids.append(int(tokenizer.eos_token_id))
        input_ids.append(torch.tensor(token_ids, dtype = torch.long))
        attention_masks.append(torch.ones(len(token_ids), dtype = torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first = True,
        padding_value = int(tokenizer.pad_token_id),
        padding_side = 'left',
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_masks,
        batch_first = True,
        padding_value = 0,
        padding_side = 'left',
    )
    return input_ids, attention_mask


class HFDataset(AcumenDataset):

    def __init__(self, args, kind = 'train'):
        super().__init__(args)

        self.args = args
        self.kind = kind
        self.hf_token = os.environ.get('HF_TOKEN', None)


        self.chunk_size = args.dataset_args.chunk_size
        self.subset = None if args.dataset_args.subset is None or not len(args.dataset_args.subset.replace("'", "").replace('"', "").strip()) else args.dataset_args.subset

        self.token_tokenizer = AutoTokenizer.from_pretrained(self.args.input_tokenizer.path, trust_remote_code = True)

        if self.token_tokenizer.pad_token is None:
            self.token_tokenizer.pad_token = self.token_tokenizer.eos_token
            print(f":: Setting pad token to {self.token_tokenizer.pad_token}")

        if self.token_tokenizer.bos_token is None:
            self.token_tokenizer.bos_token = self.token_tokenizer.eos_token
            print(f":: Setting bos token to {self.token_tokenizer.bos_token}")

        self.tokenizer = self.token_tokenizer

        # needs to be at the end.
        self.dataset = self.load_dataset()

    def __len__(self):
        return len(self.dataset)

    def preprocess(self, text):
        encoded = text.encode('utf-8', errors = 'ignore')

        if self.kind == 'train':
            if len(encoded) > self.chunk_size - 4:
                start = np.random.randint(0, len(encoded) - self.chunk_size + 4)
                encoded = encoded[start:start + self.chunk_size - 4]
        else:
            if len(encoded) > self.chunk_size:
                encoded = encoded[:self.chunk_size]

        text = encoded.decode('utf-8', errors = 'ignore')

        assert len(text) <= self.chunk_size, f"Text length {len(text)} exceeds chunk size {self.chunk_size}"
        return text

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample = {'text': self.preprocess(sample['text'])}
        return sample

    @classmethod
    def train_dataloader(cls, args):
        dataset = cls(args = args, kind = 'train')
        sampler = DistributedSampler(dataset, shuffle = True, drop_last = False) if AcumenAccelerator().is_distributed else None
        dataset.sampler = sampler

        return DataLoader(
            dataset,
            num_workers = args.environment.extra_args.num_workers,
            batch_size = args.batch_size,
            pin_memory = True,
            sampler = sampler,
            shuffle = (sampler is None),
            collate_fn = cls.collate_fn(args, dataset.token_tokenizer, kind = 'train'),
            prefetch_factor = 2 if args.environment.extra_args.num_workers > 0 else None,
        )

    @classmethod
    def val_dataloader(cls, args, kind = 'test'):
        dataset = cls(args = args, kind = kind)
        sampler = DistributedSampler(dataset, shuffle = False, drop_last = False) if AcumenAccelerator().is_distributed else None
        dataset.sampler = sampler

        return DataLoader(
            dataset,
            num_workers = args.environment.extra_args.num_workers,
            batch_size = args.eval_batch_size,
            pin_memory = True,
            shuffle = False,
            sampler = sampler,
            collate_fn = cls.collate_fn(args, dataset.token_tokenizer, kind = kind),
            prefetch_factor = 2 if args.environment.extra_args.num_workers > 0 else None,
        )

    @classmethod
    def collate_fn(cls, args, token_tokenizer, kind = 'train'):

        def _collate_fn(batch):
            formatted_input = _collate(batch)

            input_ids, attention_mask = _batch_tokenize_texts(formatted_input['text'], token_tokenizer)
            shifted_input_ids, labels = _shift_right(
                input_ids = input_ids,
                attention_mask = attention_mask,
                bos_token_id = int(token_tokenizer.bos_token_id),
                pad_token_id = int(token_tokenizer.pad_token_id),
            )

            return {
                'input_ids': shifted_input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

        return _collate_fn


class PretrainingDataset(HFDataset):

    def load_dataset(self):
        chunking_fn = functools.partial(split_chunks, chunk_size = self.chunk_size)

        print("[PretrainingDataset] Loading dataset", self.args.dataset_args.dataset_name, "subset", self.subset, "split", 'train')
        ds_names = self.args.dataset_args.dataset_name
        ds_names = ds_names if isinstance(ds_names, list) else [ds_names]
        map_num_proc = max(1, int(self.args.environment.extra_args.num_workers))

        if not len(ds_names):
            raise Exception("Expected at least one dataset name for pretraining.")

        if self.kind == 'train':
            final_ds = []
            for name in ds_names:
                final_ds.append(hfds.load_dataset(name, self.subset, split = 'train', token = self.hf_token))

            _dataset = hfds.concatenate_datasets(final_ds)

            print(f"::: Initially, the dataset has {len(_dataset)} samples.")
            _dataset = _load_or_create_chunked_dataset(
                dataset = _dataset,
                chunking_fn = chunking_fn,
                chunk_size = self.chunk_size,
                map_num_proc = map_num_proc,
            )
            print(f"::: After splitting into chunks of size {self.chunk_size}, the dataset has {len(_dataset)} samples or {len(_dataset) * AVG_TOKS_PER_CHUNK} tokens.")

            model_num_params = self.args.num_params
            num_chunks_compute_optimal = int(model_num_params * TOKS_PER_PARAM // AVG_TOKS_PER_CHUNK)

            if num_chunks_compute_optimal > len(_dataset):
                raise Exception(f"Dataset too small for model size {model_num_params} params! Need at least {num_chunks_compute_optimal} chunks, but dataset has only {len(_dataset)} samples.")

            print(f"::: Optimal number of tokens for model of size {model_num_params} is {model_num_params * TOKS_PER_PARAM}, average tokens per chunk is {AVG_TOKS_PER_CHUNK}")
            print(f"::: Sampling {num_chunks_compute_optimal} chunks for pretraining based on model size {model_num_params} params")
            _dataset = _dataset.shuffle(seed = 696969).select(range(num_chunks_compute_optimal))

            dataset = _dataset.shuffle(seed = 424242)
        else:
            eval_dataset_name = ds_names[-1]
            _dataset = hfds.load_dataset(eval_dataset_name, self.subset, split = 'train', token = self.hf_token)
            _dataset = _load_or_create_chunked_dataset(
                dataset = _dataset,
                chunking_fn = chunking_fn,
                chunk_size = self.chunk_size,
                map_num_proc = map_num_proc,
            )
            num_batches_of_interest = min(int(256 * 128), len(_dataset))
            dataset = _dataset.select(range(num_batches_of_interest))

        return dataset
