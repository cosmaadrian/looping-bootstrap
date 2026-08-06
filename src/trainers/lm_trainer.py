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

        model_output = self.model({
            'input_ids': input_ids,
            'attention_mask': batch['attention_mask'],
        })

        next_token_loss = self.next_token_prediction_loss(model_output.view(-1, model_output.size(-1)), labels.view(-1))

        if self.iter_idx % self.args.log_every == 0:  # prevent doing .item() too often
            self.log(
                'train/loss:ntp',
                next_token_loss.item(),
                on_step = True,
                force_log = True,
            )

        return next_token_loss
