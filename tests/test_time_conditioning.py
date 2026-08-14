import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from models.llm import TimestepEmbedder, TransformerDecoder


class RecordingEmbedder(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.calls = []

    def forward(self, normalized_time, step_size):
        self.calls.append((normalized_time.item(), step_size.item()))
        return torch.zeros((1, self.hidden_size), device = normalized_time.device)


class IdentityRecurrentStack(nn.Module):

    def forward(self, x, **kwargs):
        return x


class TimeConditioningTest(unittest.TestCase):

    def test_embedder_uses_time_and_step_size_paths(self):
        embedder = TimestepEmbedder(hidden_size = 8, frequency_embedding_size = 6)
        normalized_time = torch.tensor([0.0, 0.5])
        step_size = torch.tensor([0.25, 0.25])

        output = embedder(normalized_time, step_size)
        output.sum().backward()

        self.assertEqual(output.shape, (2, 8))
        self.assertIsNotNone(embedder.mlp[0].weight.grad)
        self.assertIsNotNone(embedder.step_size_mlp[0].weight.grad)

    def test_loop_uses_normalized_time_and_uniform_step_size(self):
        decoder = TransformerDecoder.__new__(TransformerDecoder)
        nn.Module.__init__(decoder)
        decoder.model = IdentityRecurrentStack()
        decoder.timestep_embedder = RecordingEmbedder(hidden_size = 4)

        decoder.loop_layers(
            embeddings = torch.zeros((1, 2, 4)),
            attention_mask = None,
            num_steps_pair = (0, 4),
            intended_num_loops = 4,
        )

        expected = [
            (0.0, 0.25),
            (0.25, 0.25),
            (0.5, 0.25),
            (0.75, 0.25),
        ]
        for actual, target in zip(decoder.timestep_embedder.calls, expected):
            self.assertAlmostEqual(actual[0], target[0])
            self.assertAlmostEqual(actual[1], target[1])

        final_time, final_step_size = decoder.timestep_embedder.calls[-1]
        self.assertAlmostEqual(final_time + final_step_size, 1.0)


if __name__ == '__main__':
    unittest.main()
