import sys
import numpy as np
import torch

np.set_printoptions(threshold = sys.maxsize)

from .optim import MuAdamW
from lib.trainer_extra import AcumenTrainer


class LMTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__(args, model)
        from lib.accelerator import AcumenAccelerator
        self.accelerator = AcumenAccelerator()
        self.next_token_prediction_loss = torch.nn.CrossEntropyLoss()
        self.iter_idx = 0

    def _sample_loop_steps(self, batch_idx):
        mean_recurrence = self.args.model_args.mean_recurrence
        mean_backprop_depth = self.args.model_args.mean_backprop_depth
        if mean_recurrence < 1:
            raise ValueError('mean_recurrence must be at least 1')
        if not 1 <= mean_backprop_depth <= mean_recurrence:
            raise ValueError('mean_backprop_depth must be between 1 and mean_recurrence')

        generator = torch.Generator(device = 'cpu')
        generator.manual_seed(self.args.seed + batch_idx)
        num_loops = torch.randint(
            low = 1,
            high = 2 * mean_recurrence,
            size = (),
            generator = generator,
        )
        num_steps_with_grad = torch.minimum(num_loops, torch.tensor(mean_backprop_depth))

        return torch.stack([num_loops - num_steps_with_grad, num_steps_with_grad])

    def configure_optimizers(self, lr = 0.1):
        if self._optimizer is not None:
            return self._optimizer

        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [{'params': decay_params, 'weight_decay': self.args.optimizer_args.weight_decay}, {'params': nodecay_params, 'weight_decay': 0.0}]
        self._optimizer = MuAdamW(
            params = optim_groups,
            lr = lr,
            betas = [self.args.optimizer_args.beta1, self.args.optimizer_args.beta2],
            weight_decay = self.args.optimizer_args.weight_decay,
            eps = float(self.args.optimizer_args.eps),
            fused = True,
        )

        return self._optimizer

    def training_step(self, batch, batch_idx):
        self.iter_idx += 1
        input_ids = batch['input_ids']
        labels = batch['labels']
        num_steps_pair = self._sample_loop_steps(batch_idx)
        intended_num_loops = int(num_steps_pair.sum().item())

        model_output = self.model({
            'input_ids': input_ids,
            'attention_mask': batch['attention_mask'],
            'num_steps_pair': num_steps_pair,
        }, intended_num_loops = intended_num_loops)

        next_token_loss = self.next_token_prediction_loss(model_output.view(-1, model_output.size(-1)), labels.view(-1))

        if self.iter_idx % self.args.log_every == 0:  # prevent doing .item() too often
            self.log(
                'train/loss:ntp',
                next_token_loss.item(),
                on_step = True,
                force_log = True,
            )

        return next_token_loss
