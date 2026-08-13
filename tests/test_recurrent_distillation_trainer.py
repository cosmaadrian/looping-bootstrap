import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from trainers.recurrent_distillation_trainer import RecurrentDistillationTrainer


class RecordingModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, batch, intended_num_loops = None, is_teacher = False):
        self.calls.append({
            'num_steps_pair': batch['num_steps_pair'],
            'intended_num_loops': intended_num_loops,
            'is_teacher': is_teacher,
        })
        return batch['input_ids'].float()


class RecurrentDistillationTrainerTest(unittest.TestCase):

    def _trainer(self):
        trainer = object.__new__(RecurrentDistillationTrainer)
        trainer.mean_backprop_depth = 2
        trainer.temperature = 1.0
        trainer.student_model = RecordingModel()
        return trainer

    def test_forward_depth_executes_exact_requested_number_of_steps(self):
        trainer = self._trainer()
        batch = {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.ones(1, 2),
        }

        trainer._forward_at_depth(batch, depth = 1)
        trainer._forward_at_depth(batch, depth = 5)

        self.assertEqual(trainer.student_model.calls, [
            {
                'num_steps_pair': (0, 1),
                'intended_num_loops': 1,
                'is_teacher': False,
            },
            {
                'num_steps_pair': (3, 2),
                'intended_num_loops': 5,
                'is_teacher': False,
            },
        ])

    def test_distillation_loss_handles_batch_without_valid_tokens(self):
        trainer = self._trainer()
        student_logits = torch.randn(1, 2, 3, requires_grad = True)
        teacher_logits = torch.randn(1, 2, 3)
        labels = torch.full((1, 2), -100)

        loss, percent_teacher_is_better = trainer._distillation_loss(
            student_logits,
            teacher_logits,
            labels,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(percent_teacher_is_better, 0.0)
        loss.backward()
        self.assertIsNotNone(student_logits.grad)

    def test_distillation_loss_handles_batch_where_teacher_is_never_better(self):
        trainer = self._trainer()
        student_logits = torch.tensor([[[0.0, 2.0, 0.0]]], requires_grad = True)
        teacher_logits = student_logits.detach().clone()
        labels = torch.tensor([[1]])

        loss, percent_teacher_is_better = trainer._distillation_loss(
            student_logits,
            teacher_logits,
            labels,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(percent_teacher_is_better, 0.0)
        loss.backward()
        self.assertIsNotNone(student_logits.grad)


if __name__ == '__main__':
    unittest.main()
