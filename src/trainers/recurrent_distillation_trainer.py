import copy

import torch
import torch.nn.functional as F

from .lm_trainer import LMTrainer


class RecurrentDistillationTrainer(LMTrainer):
    """Train a sampled recurrent depth from a deeper, detached prediction."""

    def __init__(self, args, model, teacher_model = None):
        super().__init__(args, model)

        self.student_model = model
        self.recurrent_distillation = bool(args.recurrent_distillation)
        self.teacher_momentum = float(args.get('teacher_momentum', 0.0))
        self.mean_recurrence = int(args.model_args.mean_recurrence)
        self.mean_backprop_depth = int(args.model_args.mean_backprop_depth)
        self.teacher_depth_mode = str(args.get('teacher_depth_mode', 'additive')).lower()
        self.teacher_depth_offset_min = int(args.get('teacher_depth_offset_min', 2))
        self.teacher_depth_offset_max = int(args.get('teacher_depth_offset_max', 6))
        self.teacher_depth_multiplier = int(args.get('teacher_depth_multiplier', 2))
        self.distillation_weight = float(args.distillation_weight)
        self.temperature = float(args.temperature)
        self.anchor_all_depths = bool(args.anchor_all_depths)



        self.teacher_model = None
        if self.recurrent_distillation:
            if teacher_model is None:
                # This fallback keeps direct construction useful in tests and
                # other entry points. main.py supplies a pre-DDP copy.
                teacher_model = copy.deepcopy(self._unwrap_model(self.student_model))

            self.teacher_model = teacher_model
            self._copy_student_to_teacher()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

    @staticmethod
    def _unwrap_model(model):
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            model = model.module
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        return model

    @torch.no_grad()
    def _copy_student_to_teacher(self):
        student_model = self._unwrap_model(self.student_model)
        student_parameter = next(student_model.parameters(), None)
        if student_parameter is not None:
            self.teacher_model.to(student_parameter.device)
        self.teacher_model.load_state_dict(student_model.state_dict())

    @torch.no_grad()
    def _update_teacher(self):
        student_model = self._unwrap_model(self.student_model)
        student_parameters = dict(student_model.named_parameters())
        teacher_parameters = dict(self.teacher_model.named_parameters())

        if student_parameters.keys() != teacher_parameters.keys():
            raise RuntimeError('teacher and student parameters do not match')

        momentum = self.teacher_momentum
        for name, teacher_parameter in teacher_parameters.items():
            student_parameter = student_parameters[name].detach()
            teacher_parameter.mul_(momentum).add_(student_parameter, alpha = 1 - momentum)

        # Keep stateful layers consistent as well. Integer buffers are copied;
        # floating-point buffers use the same moving average as parameters.
        student_buffers = dict(student_model.named_buffers())
        for name, teacher_buffer in self.teacher_model.named_buffers():
            if name not in student_buffers or teacher_buffer.shape != student_buffers[name].shape:
                continue

            student_buffer = student_buffers[name].detach()
            if teacher_buffer.is_floating_point() or teacher_buffer.is_complex():
                teacher_buffer.mul_(momentum).add_(student_buffer, alpha = 1 - momentum)
            else:
                teacher_buffer.copy_(student_buffer)

    def optimizer_step_end(self):
        if self.teacher_model is not None:
            self._update_teacher()

    def training_start(self):
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def get_checkpoint_state(self):
        if self.teacher_model is None:
            return {}
        return {'teacher_model_state_dict': self.teacher_model.state_dict()}

    def load_checkpoint_state(self, state_dict):
        if self.teacher_model is None:
            return

        teacher_state_dict = state_dict.get('teacher_model_state_dict')
        if teacher_state_dict is None:
            # Older checkpoints did not store a teacher. Initialize it from the
            # just-restored student so they remain backwards compatible.
            self._copy_student_to_teacher()
            return

        self.teacher_model.load_state_dict(teacher_state_dict)
        self.teacher_model.eval()

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

    def _forward_at_depth(self, batch, depth, model = None, is_teacher = False):
        grad_depth = min(depth, self.mean_backprop_depth)
        model = self.student_model if model is None else model

        return model({
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'num_steps_pair': (depth - grad_depth, grad_depth),
        }, intended_num_loops = depth, is_teacher = is_teacher)

    def _distillation_loss(self, student_logits, teacher_logits, labels):
        valid_tokens = labels.ne(-100)
        student_logits = student_logits[valid_tokens]
        teacher_logits = teacher_logits[valid_tokens]
        labels = labels[valid_tokens]

        if student_logits.numel() == 0:
            return student_logits.sum(), 0.0

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

        percent_teacher_is_better = 100 * teacher_is_better.sum().item() / labels.numel()

        student_logits = student_logits[teacher_is_better]
        teacher_logits = teacher_logits[teacher_is_better]

        if student_logits.numel() == 0:
            return student_logits.sum(), percent_teacher_is_better

        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits.float(), dim = -1)
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim = -1)

        return F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction = 'batchmean',
        ) * temperature, percent_teacher_is_better

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
                teacher_logits = self._forward_at_depth(
                    batch,
                    teacher_depth,
                    model = self.teacher_model,
                    is_teacher = True,
                )
                teacher_loss = self.next_token_prediction_loss(
                    teacher_logits.view(-1, teacher_logits.size(-1)),
                    labels.view(-1),
                )
            teacher_logits = teacher_logits.detach()
            distillation_loss, percent_teacher_is_better = self._distillation_loss(student_logits, teacher_logits, labels)
        else:
            teacher_loss = None
            distillation_loss = student_logits.new_zeros(())

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
