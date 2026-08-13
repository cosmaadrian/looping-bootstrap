import torch
import torch.distributed as dist
import tqdm

from lib.accelerator import AcumenAccelerator
from lib.evaluator_extra import Metric
from utils_generation import generate

from .pretraining_evaluator import PretrainingEvaluator


def normalize_mathematics_answer(answer):
    """Normalize inconsequential outer and repeated whitespace for exact match."""
    return ' '.join(str(answer).strip().split())


def mathematics_answers_match(prediction, answer):
    return normalize_mathematics_answer(prediction) == normalize_mathematics_answer(answer)


class DeepMindMathematicsEvaluator(PretrainingEvaluator):
    """Measure generated-answer accuracy at several recurrent looping depths."""

    DEFAULT_DEPTHS = (1, 2, 4, 8, 16)

    def __init__(self, args, model, evaluator_args, logger = None):
        super().__init__(args, model, evaluator_args, logger)

        configured_depths = evaluator_args.get('depths', self.DEFAULT_DEPTHS)
        self.depths = tuple(int(depth) for depth in configured_depths)
        if not self.depths or any(depth < 1 for depth in self.depths):
            raise ValueError('Mathematics evaluation depths must contain positive integers.')

        self.temperature = float(evaluator_args.get('temperature', 0.0))
        self.top_p = float(evaluator_args.get('top_p', 1.0))
        self.tokenizer = self.dataset.tokenizer

        # Generation can take a different number of decode steps on each rank.
        # Bypass the DDP wrapper during read-only evaluation to avoid per-forward
        # synchronization while retaining the already synchronized parameters.
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            self.generation_model = model.module
        else:
            self.generation_model = model

    def _reduce_counts(self, correct, total):
        counts = torch.cat([correct, total.view(1)])
        if self.accelerator.is_distributed:
            dist.barrier()
            dist.all_reduce(counts, op = dist.ReduceOp.SUM)
        return counts[:-1], counts[-1]

    @torch.no_grad()
    def trainer_evaluate(self, global_step = -1):
        correct = torch.zeros(len(self.depths), device = self.device, dtype = torch.float64)
        total = torch.zeros((), device = self.device, dtype = torch.float64)

        progress = tqdm.tqdm(
            self.dataset_loader,
            total = len(self.dataset_loader),
            desc = f'[{self.display_name} - {AcumenAccelerator().local_rank}] Generating answers...',
            dynamic_ncols = True,
            disable = not self.accelerator.is_master(),
        )
        for batch in progress:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            answer_token_lengths = batch['answer_token_length'].to(self.device)
            expected_answers = batch['answer']
            total += len(expected_answers)

            for depth_index, depth in enumerate(self.depths):
                generated = generate(
                    model = self.generation_model,
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    tokenizer = self.tokenizer,
                    # This is a per-problem cap, not merely a batch-wide max.
                    max_new_tokens = answer_token_lengths,
                    temperature = self.temperature,
                    top_p = self.top_p,
                    stop_on_eos = True,
                    model_kwargs = {'num_loops': depth},
                    forward_kwargs = {'intended_num_loops': depth},
                )
                predictions = generated['answer_str']
                correct[depth_index] += sum(mathematics_answers_match(prediction, answer) for prediction, answer in zip(predictions, expected_answers))

        correct, total = self._reduce_counts(correct, total)
        if not self.accelerator.is_master():
            return []
        if total.item() == 0:
            raise ValueError('DeepMind Mathematics evaluation received no examples.')

        accuracies = (correct / total).tolist()
        rendered = ', '.join(f'{depth}: {accuracy:.4f}' for depth, accuracy in zip(self.depths, accuracies))
        print(f'::: [{self.display_name}] Accuracy by loops ({int(total.item())} examples): {rendered} :::')
        return [Metric(
            f'accuracy@{depth}_loops',
            value = accuracy,
            monotonicity = ['instant', 'up'],
            evaluator = self,
        ) for depth, accuracy in zip(self.depths, accuracies)]

    @torch.no_grad()
    def evaluate(self, global_step = -1):
        return self.trainer_evaluate(global_step)
