import copy
import tqdm
import torch
import torch.distributed as dist
from lib.accelerator import AcumenAccelerator
from lib.evaluator_extra import Metric
from lib.evaluator_extra.acumen_evaluator import AcumenEvaluator


class PretrainingEvaluator(AcumenEvaluator):

    def __init__(self, args, model, evaluator_args, logger = None):
        super().__init__(args, model, evaluator_args, logger)
        from lib import device, nomenclature

        self.device = device

        arg_copy = copy.deepcopy(args)
        arg_copy.dataset_args.dataset_name = evaluator_args.dataset_name if 'dataset_name' in evaluator_args else args.dataset_args.dataset_name
        arg_copy.dataset_args.subset = evaluator_args.subset if 'subset' in evaluator_args else args.dataset_args.subset

        kind = evaluator_args.kind if 'kind' in evaluator_args else 'test'
        self.kind = kind

        dataset = evaluator_args.dataset if 'dataset' in evaluator_args else args.dataset
        self.dataset_loader = nomenclature.DATASETS[dataset].val_dataloader(arg_copy, kind = kind)
        self.dataset = self.dataset_loader.dataset

        self.accelerator = AcumenAccelerator()

        print(f"::: [{self.display_name} - {AcumenAccelerator().local_rank}] Started :::")
        print(f"::: [{self.display_name} - {AcumenAccelerator().local_rank}] Dataset: {arg_copy.dataset_args.dataset_name} ({len(self.dataset_loader)} batches) :::")

    @property
    def display_name(self):
        return 'pretraining' if 'display_name' not in self.evaluator_args else self.evaluator_args.display_name

    @torch.no_grad()
    def trainer_evaluate(self, global_step = -1):
        counter = 0
        total_loss = 0
        num_examples = 0
        num_batches = 0

        for batch in tqdm.tqdm(self.dataset_loader, total = len(self.dataset_loader), desc = f"[{self.display_name} - {AcumenAccelerator().local_rank}] Evaluating...", dynamic_ncols = True, disable = not self.accelerator.is_master()):
            counter += 1

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            logits = self.model({
                "input_ids": batch["input_ids"],
                'attention_mask': batch['attention_mask'],
            })

            ntp = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch["labels"].view(-1), reduction = 'sum')
            total_loss += ntp.item()
            num_examples += (batch["labels"] != -100).sum().item()
            num_batches += 1

        if AcumenAccelerator().is_distributed:
            # barrier for sync
            dist.barrier()

            counter = torch.tensor(counter, device = self.device)
            dist.all_reduce(counter, op = dist.ReduceOp.SUM)
            counter = counter.item()

            num_examples = torch.tensor(num_examples, device = self.device)
            dist.all_reduce(num_examples, op = dist.ReduceOp.SUM)
            num_examples = num_examples.item()

            total_loss = torch.tensor(total_loss, device = self.device)
            dist.all_reduce(total_loss, op = dist.ReduceOp.SUM)
            total_loss = total_loss.item()

        if not self.accelerator.is_master():
            return []

        total_loss = total_loss / num_examples
        print(f"::: [{self.display_name} - {AcumenAccelerator().local_rank}] Evaluated {counter // self.accelerator.world_size} batches, total loss: {total_loss:.4f} :::")

        return [
            Metric('loss', value = total_loss, monotonicity = ['instant'], evaluator = self),
        ]

    @torch.no_grad()
    def evaluate(self, global_step = -1):
        return self.trainer_evaluate(global_step)
