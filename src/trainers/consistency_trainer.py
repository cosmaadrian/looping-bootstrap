import torch
import torch.nn.functional as F

from .lm_trainer import LMTrainer


class ConsistencyTrainer(LMTrainer):
    """Self-distill between two recurrent depths of one shared model."""

    def __init__(self, args, model):
        super().__init__(args, model)

        self.student_model = model
        self.recurrent_distillation = bool(args.recurrent_distillation)
        self.mean_recurrence = int(args.model_args.mean_recurrence)
        self.mean_backprop_depth = int(args.model_args.mean_backprop_depth)
        self.teacher_depth_mode = str(args.get('teacher_depth_mode', 'additive')).lower()
        self.teacher_depth_offset_min = int(args.get('teacher_depth_offset_min', 2))
        self.teacher_depth_offset_max = int(args.get('teacher_depth_offset_max', 6))
        self.teacher_depth_multiplier = int(args.get('teacher_depth_multiplier', 2))
        self.distillation_weight = float(args.distillation_weight)
        self.temperature = float(args.temperature)

        if self.temperature <= 0:
            raise ValueError('temperature must be greater than zero')

    def _sample_depths(self, batch_idx):
        # Use one CPU generator so every DDP rank samples the same depth pair.
        generator = torch.Generator(device = 'cpu')
        generator.manual_seed(self.args.seed + batch_idx)

        student_depth = int(torch.poisson(
            torch.tensor(float(self.mean_recurrence)),
            generator = generator,
        ).clamp_(1, self.mean_recurrence * 4).item())

        if self.teacher_depth_mode == 'multiplicative':
            teacher_depth = self.teacher_depth_multiplier * student_depth
        else:
            teacher_offset = torch.randint(
                low = self.teacher_depth_offset_min,
                high = self.teacher_depth_offset_max + 1,
                size = (),
                generator = generator,
            ).item()
            teacher_depth = student_depth + teacher_offset

        return student_depth, teacher_depth

    def _forward_at_depth(self, batch, depth, is_teacher = False):
        grad_depth = min(depth, self.mean_backprop_depth)

        return self.student_model({
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'num_steps_pair': (depth - grad_depth, grad_depth),
        }, intended_num_loops = depth, is_teacher = is_teacher)

    def _consistency_loss(self, student_logits, teacher_logits, labels):
        valid_tokens = labels.ne(-100)
        student_logits = student_logits[valid_tokens]
        teacher_logits = teacher_logits[valid_tokens]
        labels = labels[valid_tokens]

        if student_logits.numel() == 0:
            return student_logits.sum() + teacher_logits.sum(), 0.0

        with torch.no_grad():
            student_token_losses = F.cross_entropy(
                student_logits.float(),
                labels,
                reduction = 'none'
            )
            teacher_token_losses = F.cross_entropy(
                teacher_logits.float(),
                labels,
                reduction = 'none'
            )
            teacher_is_better = teacher_token_losses < student_token_losses
            student_is_better = student_token_losses < teacher_token_losses

        percent_teacher_is_better = 100 * teacher_is_better.sum().item() / labels.numel()
        temperature = self.temperature

        consistency_loss = student_logits.new_zeros(())

        if teacher_is_better.any():
            student_log_probs = F.log_softmax(
                student_logits[teacher_is_better].float() / temperature,
                dim = -1,
            )
            teacher_probs = F.softmax(
                teacher_logits[teacher_is_better].detach().float() / temperature,
                dim = -1,
            )
            consistency_loss = consistency_loss + F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction = 'sum',
            ) / labels.numel()

        if student_is_better.any():
            teacher_log_probs = F.log_softmax(
                teacher_logits[student_is_better].float() / temperature,
                dim = -1,
            )
            student_probs = F.softmax(
                student_logits[student_is_better].detach().float() / temperature,
                dim = -1,
            )
            consistency_loss = consistency_loss + F.kl_div(
                teacher_log_probs,
                student_probs,
                reduction = 'sum',
            ) / labels.numel()

        return consistency_loss * temperature**2, percent_teacher_is_better

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
            teacher_logits = self._forward_at_depth(
                batch,
                teacher_depth,
                is_teacher = True,
            )
            teacher_loss = self.next_token_prediction_loss(
                teacher_logits.detach().view(-1, teacher_logits.size(-1)),
                labels.view(-1),
            )
            distillation_loss, percent_teacher_is_better = self._consistency_loss(student_logits, teacher_logits, labels)
        else:
            teacher_loss = None
            distillation_loss = student_logits.new_zeros(())

        # When not anchoring every sampled depth, retain direct supervision at
        # the nominal recurrence depth and train the other depths by consistency.
        anchored_next_token_loss = next_token_loss
        total_loss = anchored_next_token_loss + self.distillation_weight * distillation_loss

        if self.iter_idx % self.args.log_every == 0:  # prevent doing .item() too often
            log_dict = {
                'train/loss:ntp': next_token_loss.item(),
                'train/loss:total': total_loss.item(),
                'train/student_depth': student_depth,
            }

            if self.recurrent_distillation:
                log_dict['train/perc_teacher_is_better'] = percent_teacher_is_better
                log_dict['train/loss:distillation'] = distillation_loss.item()
                if teacher_loss is not None:
                    log_dict['train/loss:teacher'] = teacher_loss.item()
                    log_dict['train/teacher_depth'] = teacher_depth

            self.log_dict(log_dict, on_step = False, force_log = True)

        return total_loss
