import torch
import torch.nn.functional as F

from .lm_trainer import LMTrainer


class RecurrentDistillationTrainer(LMTrainer):
    """Train a sampled recurrent depth from a deeper, detached prediction."""

    def __init__(self, args, model):
        super().__init__(args, model)

        self.recurrent_distillation = bool(args.recurrent_distillation)
        self.student_depths = tuple(int(depth) for depth in args.student_depths)
        self.teacher_depth_multiplier = int(args.teacher_depth_multiplier)
        self.distillation_weight = float(args.distillation_weight)
        self.temperature = float(args.temperature)
        self.anchor_all_depths = bool(args.anchor_all_depths)

        if not self.student_depths:
            raise ValueError('student_depths must contain at least one depth')
        if any(depth < 1 for depth in self.student_depths):
            raise ValueError('student_depths must contain only positive integers')
        if self.teacher_depth_multiplier < 1:
            raise ValueError('teacher_depth_multiplier must be at least 1')
        if self.distillation_weight < 0:
            raise ValueError('distillation_weight must be non-negative')
        if self.temperature <= 0:
            raise ValueError('temperature must be greater than zero')

    def _sample_student_depth(self, batch_idx):
        # Use a CPU generator so every DDP rank samples the same depth.
        generator = torch.Generator(device = 'cpu')
        generator.manual_seed(self.args.seed + batch_idx)
        index = torch.randint(len(self.student_depths), size = (), generator = generator).item()
        return self.student_depths[index]

    def _forward_at_depth(self, batch, depth):
        return self.model({
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'num_loops': depth,
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
        student_depth = self._sample_student_depth(sample_idx)
        teacher_depth = student_depth * self.teacher_depth_multiplier

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
        use_ce_anchor = self.anchor_all_depths or student_depth == max(self.student_depths)
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
