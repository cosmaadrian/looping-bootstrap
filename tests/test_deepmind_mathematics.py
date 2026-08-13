import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import datasets as hfds
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from lib import nomenclature
from evaluators.deepmind_mathematics_evaluator import mathematics_answers_match
from utils_generation import generate

DeepMindMathematicsDataset = nomenclature.DATASETS['deepmind_mathematics']


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    pad_token = '<pad>'
    bos_token = '<bos>'
    eos_token = '<eos>'

    def encode(self, text, add_special_tokens = False):
        del add_special_tokens
        return [3 + ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens = True):
        special_tokens = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return ''.join('x' if token_id == 4 else chr(token_id - 3) for token_id in token_ids if not skip_special_tokens or token_id not in special_tokens)


class ConstantTokenModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.batch_sizes = []
        self.loop_depths = []

    def forward(self, batch, intended_num_loops = None):
        self.batch_sizes.append(batch['input_ids'].shape[0])
        self.loop_depths.append((batch.get('num_loops'), intended_num_loops))
        batch_size, sequence_length = batch['input_ids'].shape
        logits = torch.zeros(batch_size, sequence_length, 10)
        logits[..., 4] = 1
        return logits


class DeepMindMathematicsTest(unittest.TestCase):

    def _args(self, dataset_path = 'unused'):
        return EasyDict({
            'seed': 7,
            'input_tokenizer': {
                'path': 'fake'
            },
            'dataset_args': {
                'dataset_name': dataset_path,
                'subset': '',
                'chunk_size': 512,
                'train_split': 'train',
                'test_split': 'interpolate',
            },
        })

    def test_training_masks_problem_labels_but_keeps_problem_visible(self):
        tokenizer = FakeTokenizer()
        collate = DeepMindMathematicsDataset.collate_fn(self._args(), tokenizer, kind = 'train')
        batch = collate([{'question': '1 + 1?', 'answer': '2', 'module': 'arithmetic'}])

        problem_tokens = batch['loss_attention_mask'].eq(0)
        answer_tokens = batch['loss_attention_mask'].eq(1)
        self.assertTrue(torch.all(batch['labels'][problem_tokens] == -100))
        self.assertTrue(torch.all(batch['labels'][answer_tokens] != -100))
        self.assertTrue(torch.all(batch['attention_mask'] == 1))

    def test_evaluation_split_is_sorted_by_tokenized_answer_length(self):
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset = hfds.DatasetDict({
                'train': hfds.Dataset.from_dict({
                    'question': ['q'],
                    'answer': ['a'],
                    'module': ['m']
                }),
                'interpolate': hfds.Dataset.from_dict({
                    'question': ['q1', 'q2', 'q3'],
                    'answer': ['abcd', 'z', 'xy'],
                    'module': ['m', 'm', 'm'],
                }),
            })
            dataset.save_to_disk(temporary_dir)
            args = self._args(temporary_dir)
            with mock.patch('datasetss.pretraining_dataset.AutoTokenizer.from_pretrained', return_value = tokenizer):
                loaded = DeepMindMathematicsDataset(args, kind = 'interpolate')

            self.assertEqual(loaded.dataset['answer'], ['z', 'xy', 'abcd'])
            self.assertEqual(loaded.dataset['answer_token_length'], [2, 3, 5])

    def test_generation_honors_each_problem_token_budget(self):
        tokenizer = FakeTokenizer()
        model = ConstantTokenModel()
        generated = generate(
            model = model,
            input_ids = torch.tensor([[1, 5, 0], [1, 6, 7]]),
            attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]]),
            tokenizer = tokenizer,
            max_new_tokens = torch.tensor([1, 3]),
            temperature = 0,
            model_kwargs = {'num_loops': 8},
            forward_kwargs = {'intended_num_loops': 8},
        )

        self.assertEqual(generated['answer_attention_mask'].sum(dim = 1).tolist(), [1, 3])
        self.assertEqual(generated['answer_str'], ['x', 'xxx'])
        self.assertEqual(model.batch_sizes, [2, 1, 1])
        self.assertEqual(model.loop_depths, [(8, 8), (8, 8), (8, 8)])

    def test_correctness_ignores_only_inconsequential_whitespace(self):
        self.assertTrue(mathematics_answers_match(' 54*a   - 30 ', '54*a - 30'))
        self.assertFalse(mathematics_answers_match('54*a - 31', '54*a - 30'))


if __name__ == '__main__':
    unittest.main()
