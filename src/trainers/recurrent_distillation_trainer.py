import torch
import torch.nn.functional as F

from .lm_trainer import LMTrainer


class RecurrentDistillationTrainer(LMTrainer):
    """Train a sampled recurrent depth from a deeper, detached prediction."""

    def __init__(self, args, model):
        super().__init__(args, model)

        self.recurrent_distillation = bool(args.recurrent_distillation)
        self.mean_recurrence = int(args.model_args.mean_recurrence)
        self.mean_backprop_depth = int(args.model_args.mean_backprop_depth)
        self.max_student_depth = 2 * self.mean_recurrence - 1
        self.teacher_depth_offset_min = int(args.teacher_depth_offset_min)
        self.teacher_depth_offset_max = int(args.teacher_depth_offset_max)
        self.distillation_weight = float(args.distillation_weight)
        self.temperature = float(args.temperature)
        self.anchor_all_depths = bool(args.anchor_all_depths)

        if self.mean_recurrence < 1:
            raise ValueError('mean_recurrence must be at least 1')
        if self.mean_backprop_depth < 1:
            raise ValueError('mean_backprop_depth must be at least 1')
        if self.teacher_depth_offset_min < 1:
            raise ValueError('teacher_depth_offset_min must be at least 1')
        if self.teacher_depth_offset_max < self.teacher_depth_offset_min:
            raise ValueError('teacher_depth_offset_max must be at least teacher_depth_offset_min')
        if self.distillation_weight < 0:
            raise ValueError('distillation_weight must be non-negative')
        if self.temperature <= 0:
            raise ValueError('temperature must be greater than zero')

    def _sample_depths(self, batch_idx):
        # Use one CPU generator so every DDP rank samples the same depth pair.
        generator = torch.Generator(device = 'cpu')
        generator.manual_seed(self.args.seed + batch_idx)
        student_depth = torch.randint(
            low = 1,
            high = self.max_student_depth + 1,
            size = (),
            generator = generator,
        ).item()
        teacher_offset = torch.randint(
            low = self.teacher_depth_offset_min,
            high = self.teacher_depth_offset_max + 1,
            size = (),
            generator = generator,
        ).item()
        return student_depth, student_depth + teacher_offset

    def _forward_at_depth(self, batch, depth):
        grad_depth = min(depth, self.mean_backprop_depth)
        return self.model({
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'num_steps_pair': (depth - grad_depth, grad_depth),
        })

    def _distillation_loss(self, student_logits, teacher_logits, labels):
        valid_tokens = labels.ne(-100)
        student_logits = student_logits[valid_tokens]
        teacher_logits = teacher_logits[valid_tokens]

        if student_logits.numel() == 0:
            return student_logits.sum()

        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim = -1)
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim = -1)

        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction = 'batchmean',
        ) * temperature**2

    def training_step(self, batch, batch_idx):
        self.iter_idx += 1
        labels = batch['labels']
        accumulation_steps = int(self.args.get('accumulation_steps', 1))
        microbatch_idx = (self.iter_idx - 1) % accumulation_steps
        sample_idx = batch_idx * accumulation_steps + microbatch_idx
        student_depth, teacher_depth = self._sample_depths(sample_idx)

        student_logits = self._forward_at_depth(batch, student_depth)
        next_token_loss = self.next_token_prediction_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
        )

        if self.recurrent_distillation:
            with torch.no_grad():
                teacher_logits = self._forward_at_depth(batch, teacher_depth)
            teacher_logits = teacher_logits.detach()
            distillation_loss = self._distillation_loss(student_logits, teacher_logits, labels)
        else:
            distillation_loss = student_logits.new_zeros(())

        # With the default anchor_all_depths=true, every sampled student is
        # directly supervised. Disabling it leaves only the deepest student
        # anchored to labels and trains shallower depths through distillation.
        use_ce_anchor = self.anchor_all_depths or student_depth == self.max_student_depth
        anchored_next_token_loss = next_token_loss if use_ce_anchor else next_token_loss * 0
        total_loss = anchored_next_token_loss + self.distillation_weight * distillation_loss

        if self.iter_idx % self.args.log_every == 0:  # prevent doing .item() too often
            self.log(
                'train/loss:ntp',
                next_token_loss.item(),
                on_step = False,
                force_log = True
            )
            self.log(
                'train/loss:distillation',
                distillation_loss.item(),
                on_step = False,
                force_log = True
            )
            self.log(
                'train/loss:total',
                total_loss.item(),
                on_step = False,
                force_log = True
            )

        return total_loss
