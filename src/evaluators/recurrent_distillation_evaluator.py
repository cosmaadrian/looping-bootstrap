import tqdm
import torch
import torch.distributed as dist

from lib.accelerator import AcumenAccelerator
from lib.evaluator_extra import Metric

from .pretraining_evaluator import PretrainingEvaluator


class RecurrentDistillationEvaluator(PretrainingEvaluator):
    """Evaluate recurrent depths and compare the run with its initial generation."""

    DEFAULT_DEPTHS = (1, 2, 4, 8, 16, 32)
    tracks_recurrent_generations = True

    def __init__(self, args, model, evaluator_args, logger = None):
        super().__init__(args, model, evaluator_args, logger)

        configured_depths = evaluator_args.get('depths', self.DEFAULT_DEPTHS)
        self.depths = tuple(int(depth) for depth in configured_depths)
        if not self.depths or any(depth < 1 for depth in self.depths):
            raise ValueError('evaluation depths must contain positive integers')

        student_depths = tuple(int(depth) for depth in args.student_depths)
        self.student_depths = tuple(depth for depth in student_depths if depth in self.depths)
        self.original_teacher_depth = max(student_depths) * int(args.teacher_depth_multiplier)
        if self.original_teacher_depth not in self.depths:
            raise ValueError(f'evaluation depths must include the original teacher depth '
                             f'{self.original_teacher_depth}')

        self._generation_zero = None

    def _reduce_totals(self, loss_sums, correct_sums, num_tokens, counter):
        if not self.accelerator.is_distributed:
            return loss_sums, correct_sums, num_tokens, counter

        dist.barrier()
        totals = torch.cat([
            loss_sums,
            correct_sums,
            num_tokens.view(1),
            counter.view(1),
        ])
        dist.all_reduce(totals, op = dist.ReduceOp.SUM)

        num_depths = len(self.depths)
        loss_sums = totals[:num_depths]
        correct_sums = totals[num_depths:2 * num_depths]
        num_tokens = totals[-2]
        counter = totals[-1]
        return loss_sums, correct_sums, num_tokens, counter

    def _metrics(self, losses, accuracies):
        metrics = []
        for index, depth in enumerate(self.depths):
            metrics.extend([
                Metric(
                    f'loss@{depth}_loops',
                    value = losses[index],
                    monotonicity = ['instant', 'down'],
                    evaluator = self,
                ),
                Metric(
                    f'accuracy@{depth}_loops',
                    value = accuracies[index],
                    monotonicity = ['instant', 'up'],
                    evaluator = self,
                ),
            ])

        accuracy_by_depth = dict(zip(self.depths, accuracies))
        for depth in self.depths:
            teacher_depth = 2 * depth
            if teacher_depth not in accuracy_by_depth:
                continue

            metrics.append(Metric(
                f'accuracy_delta@{depth}_to_{teacher_depth}_loops',
                value = accuracy_by_depth[teacher_depth] - accuracy_by_depth[depth],
                monotonicity = ['instant', 'up'],
                evaluator = self,
            ))

        if self._generation_zero is None:
            self._generation_zero = {
                'losses': dict(zip(self.depths, losses)),
                'accuracies': accuracy_by_depth,
            }

        baseline_losses = self._generation_zero['losses']
        baseline_accuracies = self._generation_zero['accuracies']
        for depth in self.student_depths:
            metrics.extend([
                Metric(
                    f'shallow_accuracy_improvement@{depth}_loops',
                    value = accuracy_by_depth[depth] - baseline_accuracies[depth],
                    monotonicity = ['instant', 'up'],
                    evaluator = self,
                ),
                Metric(
                    f'shallow_loss_improvement@{depth}_loops',
                    value = baseline_losses[depth] - losses[self.depths.index(depth)],
                    monotonicity = ['instant', 'up'],
                    evaluator = self,
                ),
            ])

        depth = self.original_teacher_depth
        deep_accuracy_improvement = accuracy_by_depth[depth] - baseline_accuracies[depth]
        deep_loss_improvement = baseline_losses[depth] - losses[self.depths.index(depth)]
        metrics.extend([
            Metric(
                f'previous_teacher_loss@{depth}_loops',
                value = baseline_losses[depth],
                monotonicity = ['instant'],
                evaluator = self,
            ),
            Metric(
                f'deep_model_loss_improvement@{depth}_loops',
                value = deep_loss_improvement,
                monotonicity = ['instant', 'up'],
                evaluator = self,
            ),
        ])
        return metrics

    @torch.no_grad()
    def trainer_evaluate(self, global_step = -1):
        num_depths = len(self.depths)
        loss_sums = torch.zeros(num_depths, device = self.device, dtype = torch.float64)
        correct_sums = torch.zeros(num_depths, device = self.device, dtype = torch.float64)
        num_tokens = torch.zeros((), device = self.device, dtype = torch.float64)
        counter = torch.zeros((), device = self.device, dtype = torch.float64)

        progress = tqdm.tqdm(
            self.dataset_loader,
            total = len(self.dataset_loader),
            desc = f'[{self.display_name} - {AcumenAccelerator().local_rank}] Evaluating depths...',
            dynamic_ncols = True,
            disable = not self.accelerator.is_master(),
        )
        for batch in progress:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            labels = batch['labels']
            valid_tokens = labels.ne(-100)
            num_tokens += valid_tokens.sum()
            counter += 1

            for index, depth in enumerate(self.depths):
                logits = self.model({
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'num_loops': depth,
                })
                loss_sums[index] += torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index = -100,
                    reduction = 'sum',
                ).double()
                predictions = logits.argmax(dim = -1)
                correct_sums[index] += predictions[valid_tokens].eq(labels[valid_tokens]).sum()

        loss_sums, correct_sums, num_tokens, counter = self._reduce_totals(
            loss_sums,
            correct_sums,
            num_tokens,
            counter,
        )

        if not self.accelerator.is_master():
            return []
        if num_tokens.item() == 0:
            raise ValueError('recurrent depth evaluation received no supervised tokens')

        losses = (loss_sums / num_tokens).tolist()
        accuracies = (correct_sums / num_tokens).tolist()
        rendered_losses = ', '.join(f'{depth}: {loss:.4f}' for depth, loss in zip(self.depths, losses))
        print(f'::: [{self.display_name} - {AcumenAccelerator().local_rank}] '
              f'Evaluated {int(counter.item()) // self.accelerator.world_size} batches; '
              f'loss by loops: {rendered_losses} :::')
        return self._metrics(losses, accuracies)

    @torch.no_grad()
    def evaluate(self, global_step = -1):
        return self.trainer_evaluate(global_step)
