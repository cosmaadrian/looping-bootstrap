import lib
import time
import tqdm
import numpy as np
import torch
import torch.distributed as dist
from colorama import Fore, Style
from colorama import init as colorama_init
from .loggers import NoLogger
from .evaluator_extra import MetricCollection

colorama_init()


class NotALightningTrainer():

    def __init__(
        self,
        args,
        callbacks = None,
        logger = None,
        accelerator = None,
        scheduler = None,
        state_dict = None,
    ):
        self.args = args

        # This doesn't belong here. Do it like this for now.
        self.scheduler = scheduler

        self.state_dict = state_dict

        self.accelerator = accelerator

        # counters
        self.epoch = 0
        self.global_step = 0

        if state_dict is not None:
            self.epoch = state_dict['current_epoch']
            self.global_step = state_dict['current_iter']

        self.logger = logger

        if self.logger is None:
            self.logger = NoLogger()

        self.logger.trainer = self

        self.callbacks = callbacks
        if self.callbacks is None:
            self.callbacks = []

        for callback in self.callbacks:
            callback.trainer = self

        self.model_hook = None
        self.scaler = None

        self.original_eval_every_batches = self.args.eval_every_batches

    def run_evaluation(self):
        self.accelerator.synchronize()
        aggregation_evaluators = [evaluator for evaluator in self.evaluators if evaluator.is_aggregator]
        non_aggregation = [evaluator for evaluator in self.evaluators if not evaluator.is_aggregator]

        if len(aggregation_evaluators):
            metric_collection = MetricCollection()

        self.model_hook.train(False)
        with torch.no_grad():
            with torch.amp.autocast(lib.device_name, enabled = bool(self.args.use_amp)):
                for evaluator in non_aggregation:
                    self.accelerator.master_print(f'[{evaluator.display_name}] Running evaluation ...')
                    metric_collection = evaluator.evaluate_and_log(self.global_step)

                    if len(aggregation_evaluators):
                        metric_collection.extend(evaluator.current_metric_collection)

                for agg_evaluator in aggregation_evaluators:
                    metric_collection = agg_evaluator.evaluate_and_log(self.global_step, metrics = metric_collection)

        self.model_hook.train(True)

    def fit(self, model, optimizer, train_dataloader, evaluators = None):
        model.log = self.logger.log

        self.evaluators = evaluators

        if self.evaluators is None:
            self.evaluators = []

        for evaluator in self.evaluators:
            evaluator.trainer = self

        self.optimizer = optimizer

        model.trainer = self
        self.model_hook = model.model

        self.logger.watch(self.model_hook)
        self.scaler = torch.amp.GradScaler(lib.device_name, enabled = bool(self.args.use_amp))

        if self.state_dict is not None:
            self.scaler.load_state_dict(self.state_dict['scaler_state_dict'])
            model.model.load_state_dict(self.state_dict['model_state_dict'])
            model.load_checkpoint_state(self.state_dict)

        if self.accelerator.is_distributed:
            dist.barrier()

        model.dataset = train_dataloader.dataset
        model.training_start()

        if self.args.n_train_iters != -1:
            actual_epochs = int(np.ceil(self.args.n_train_iters / len(train_dataloader)))
            actual_n_train_iters = self.args.n_train_iters
        else:
            actual_epochs = self.args.epochs
            actual_n_train_iters = self.args.epochs * len(train_dataloader)

        if self.accelerator.is_master():
            pbar = tqdm.tqdm(
                range(self.global_step, actual_n_train_iters, 1),
                total = actual_n_train_iters,
                colour = 'cyan',
                dynamic_ncols = True,
                bar_format = '{desc}: {percentage:0.3f}%|{bar}{r_bar}',
            )
        else:
            pbar = range(self.global_step, actual_n_train_iters, 1)

        dataloader_iterator = iter(train_dataloader)
        for callback in self.callbacks:
            callback.on_epoch_start()

        model.training_epoch_start(self.epoch)
        for it in pbar:
            try:
                t00 = time.time()
                data = next(dataloader_iterator)
                i = self.global_step
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(lib.device, non_blocking = True)
                t01 = time.time()
            except StopIteration:
                if self.accelerator.is_distributed:
                    model.dataset.sampler.set_epoch(int(np.floor(self.epoch)) + 1)

                model.training_epoch_end(self.epoch)

                # Start of a new epoch
                for callback in self.callbacks:
                    callback.on_epoch_start()

                model.training_epoch_start(self.epoch)

                dataloader_iterator = iter(train_dataloader)
                t00 = time.time()
                data = next(dataloader_iterator)
                i = self.global_step
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].to(lib.device, non_blocking = True)
                t01 = time.time()

            self.epoch = self.global_step * (actual_epochs / actual_n_train_iters)
            self.logger.log('epoch', self.epoch, on_step = False, force_log = False)

            t0 = time.time()

            for callback in self.callbacks:
                callback.on_batch_start()

            # For debugging ...
            # self.run_evaluation()
            # exit(-1)

            # Autocast to automatically save memory with marginal loss of performance
            with torch.amp.autocast(lib.device_name, enabled = bool(self.args.use_amp)):
                if self.accelerator.is_distributed:
                    model.model.require_backward_grad_sync = (it + 1) % self.args.accumulation_steps == 0

                model.training_batch_start(data)
                loss = model.training_step(data, i)
                model.training_batch_end(data)

                if torch.isnan(loss):
                    self.accelerator.master_print('🤯 Loss is NaN. Stopping training ...')
                    self.accelerator.terminate()
                    exit(-1)

                loss = loss / self.args.accumulation_steps

            self.scaler.scale(loss).backward()

            if (it + 1) % self.args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)

                if self.args.log_grads:
                    grads = [param.grad.detach().flatten() for param in self.model_hook.parameters() if param.grad is not None]
                    norm = torch.cat(grads).norm().item()
                    self.logger.log('norm', norm, on_step = False, force_log = False)

                if bool(self.args.clip_grad_norm):
                    torch.nn.utils.clip_grad_norm_(self.model_hook.parameters(), self.args.max_grad_norm)

                scale_before_step = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none = True)

                # GradScaler lowers the scale when it skips an optimizer step
                # because of non-finite gradients. Keep post-step state such as
                # an EMA teacher aligned only with real student updates.
                optimizer_step_succeeded = self.scaler.get_scale() >= scale_before_step
                if optimizer_step_succeeded:
                    model.optimizer_step_end()

                if self.accelerator.is_distributed:
                    dist.barrier()

                for callback in self.callbacks:
                    callback.on_batch_end()

                t1 = time.time()
                dt = t1 - t0  # time difference in seconds
                dl = t01 - t00

                # self.logger.log('dt', dt, on_step = False, force_log = False)

                tokens_processed = self.args.batch_size * self.args.dataset_args.chunk_size * self.args.accumulation_steps * self.accelerator.world_size
                tokens_per_second = tokens_processed / dt

                if self.accelerator.is_master():
                    progress_string = f'[{Fore.GREEN}{self.args.group}{Style.RESET_ALL}:{Fore.RED}{self.args.name}{Style.RESET_ALL}] ' + \
                        f'{np.round(self.epoch)} / {actual_epochs} | {self.global_step} / {actual_n_train_iters} | dl: {dl*1000:.2f}ms | dt: {dt*1000:.2f}ms | toks/sec: {tokens_per_second:.2f} | ' + ' | '.join([
                        f'{k}={np.round(np.mean(v), 4)}' for k,v in self.logger.on_step_metrics.items()
                    ])

                    pbar.set_description(progress_string)

                ##############################################################
                ##############################################################
                if (self.args.eval_every_batches != -1 and ((self.global_step + 1) % self.args.eval_every_batches == 0)) or (self.global_step == actual_n_train_iters) or self.args.debug:
                    self.accelerator.master_print(f":::: Evaluating every {self.args.eval_every_batches} ::::")

                    self.run_evaluation()

                    for callback in self.callbacks:
                        callback.on_epoch_end()

                self.global_step += 1

                # Adjust evaluation frequency dynamically based on training progress
                # At the halfway point, evaluate half as often. At 75%, evaluate a quarter as often. This is because we usually want to evaluate more frequently at the beginning of training when things are changing rapidly, and less frequently towards the end when things are more stable.
                if self.global_step > actual_n_train_iters * 0.5:
                    self.args.eval_every_batches = self.original_eval_every_batches * 2

                if self.global_step > actual_n_train_iters * 0.75:
                    self.args.eval_every_batches = self.original_eval_every_batches * 4

                if self.args.debug:
                    self.accelerator.master_print('[🐞DEBUG MODE🐞] Breaking after one batch ... ')
                    break

                ##############################################################
                ##############################################################
                if self.global_step >= actual_n_train_iters:
                    break

                if self.global_step >= self.args.actually_stop_at:
                    self.accelerator.master_print(f":::: Stopping at {self.args.actually_stop_at} ::::")
                    break

        model.training_end()
