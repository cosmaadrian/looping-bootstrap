import torch

torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(threshold = 10_000)

import os
import glob

os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'

import numpy as np
import lib.callbacks as callbacks
from lib import NotALightningTrainer, device_name, nomenclature
from utils import compute_and_set_base_shapes, get_cosine_schedule_with_warmup
from colorama import init as colorama_init
from easydict import EasyDict
from torchinfo import summary
from lib.arg_utils import define_args
from lib.accelerator import AcumenAccelerator

colorama_init()

accelerator = AcumenAccelerator()
args = define_args(print_fn = accelerator.master_print, verbose = True)

args.environment.extra_args.wandb_entity = os.environ.get('WANDB_ENTITY', None)

if not args.resume_from == '':
    accelerator.master_print('💚 Trying to resume!')
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/{args.resume_from}/*.ckpt'
    checkpoints = glob.glob(checkpoint_path)

    # Get latest one
    checkpoint = sorted(checkpoints, key = lambda f: os.path.getmtime(f))[-1]

    map_location = {f'{device_name}:%d' % 0: f'{device_name}:%d' % accelerator.local_rank}
    state_dict = torch.load(checkpoint, map_location = map_location, weights_only = False)
    # add module. prefix
    state_dict['model_state_dict'] = {'module.' + k: v for k, v in state_dict['model_state_dict'].items()}

    if not bool(args.use_compile):
        state_dict['model_state_dict'] = {k.replace('._orig_mod.', '.'): v for k, v in state_dict['model_state_dict'].items()}

    accelerator.master_print(f'💚 Read the checkpoint at {args.resume_from} (Current iter: {state_dict["current_iter"]})')

###############################################################################
###############################################################################

accelerator.set_args(args)

if args.resume_from != '':
    accelerator.set_rng_state(state_dict['random_state'])

######################################################
######################################################
######################################################
architecture = nomenclature.MODELS[args.model](args)

try:
    accelerator.master_print(summary(architecture, verbose = 0))
except Exception as e:
    accelerator.master_print('::: ⚠️WARNING⚠️ could not create model summary ::: ', e)

# hack to make a base model
arg_copy = EasyDict(vars(args))
arg_copy.model_width_multiplier = 1
base_architecture = nomenclature.MODELS[args.model](arg_copy)

accelerator.master_print('🖥️🖥️🖥️ Computing Base Shapes (muP) 🖥️🖥️🖥️\n')
compute_and_set_base_shapes(model = architecture, base = base_architecture)

# remove the base_architecture, only needed the shapes
num_params_base = sum(p.numel() for p in base_architecture.parameters() if p.requires_grad)
del base_architecture
del arg_copy

architecture = accelerator.prepare_model(architecture)

print('::: Computing num params.')
num_params = sum(p.numel() for p in architecture.parameters() if p.requires_grad)
args.num_params = num_params
accelerator.master_print(f"::: Model has {num_params/1e9:.2f} billion parameters.")
accelerator.master_print(f"::: Base model has {num_params_base/1e9:.2f} billion parameters.")

train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)
model = nomenclature.TRAINERS[args.trainer](args, architecture)

# train for a constant amount of tokens, accounting for n_repetitions.
# args.n_train_iters = len(train_dataloader.dataset) * args.dataset_args.num_repetitions // (args.batch_size * accelerator.world_size)
# print(f"🔁 Setting n_train_iters to {args.n_train_iters} to account for num_repetitions = {args.dataset_args.num_repetitions}")

accelerator.master_print(f'🔁 Evaluating every {args.eval_every_batches} optimizer steps.')


def _evaluator_runtime_args(evaluator_args):
    runtime_args = EasyDict(evaluator_args.args if 'args' in evaluator_args else {})

    for key, value in evaluator_args.items():
        if key in ('args', 'name'):
            continue

        runtime_args[key] = value

    return runtime_args


evaluators = [nomenclature.EVALUATORS[evaluator_args.name](args, architecture, _evaluator_runtime_args(evaluator_args)) for evaluator_args in args.evaluators]

# Duplicate code, but it's fine for now ...
if args.n_train_iters != -1:
    actual_epochs = int(np.ceil(args.n_train_iters / len(train_dataloader)))
    actual_n_train_iters = args.n_train_iters
else:
    raise Exception('Currently, n_train_iters must be set!')

checkpoint_callback_iter = callbacks.IterationCheckpoint(
    args = args,
    name = '🔁 Iteration Checkpoint 🔁',
    monitor = args.model_checkpoint.monitor_quantity,
    dirpath = f'{args.model_checkpoint.base_dir}/{args.group}:{args.name}/iter/',
    save_best_only = False,
    direction = None,
    filename = 'epoch={epoch}-step={global_step}',
    interval = 4096,
)

max_lr = args.scheduler_args.max_lr

# https://arxiv.org/pdf/2409.19913
# lr_adjusted = lr_{len(smallest_dataset)} * np.pow(len(current_dataset) / len(smallest_dataset), -0.32)
# Therefore, lr_adjusted = lr_{mu=1} * (num_params(mu=current) / num_params(mu=1))^{-0.32}

if args.scheduler_args.base_batch_size != args.batch_size:
    max_lr = max_lr * ((args.batch_size * accelerator.world_size) / args.scheduler_args.base_batch_size)
    accelerator.master_print(f"💚 Adjusting max_lr from {args.scheduler_args.max_lr} to {max_lr} for (global) batch size {args.batch_size * accelerator.world_size}")

# accelerator.master_print(f"💚 Adjusting max_lr to {max_lr} for token_horizon (num_params = {num_params}, num_params_base = {num_params_base})")
# max_lr = max_lr * (num_params / num_params_base)**(-0.32)

# In total, lr is adjusted 3 ways:
# 2. global batch size
# 3. token horizon
# 1. muP (based on model width)

warmup_steps = args.scheduler_args.num_warmup_steps

if 0.0 < warmup_steps < 1.0:
    warmup_steps = int(actual_n_train_iters * warmup_steps)
    accelerator.master_print(f"💚 Setting num_warmup_steps to {warmup_steps} ({args.scheduler_args.num_warmup_steps} * {actual_n_train_iters})")

optimizer = model.configure_optimizers(lr = max_lr)
scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, num_training_steps = int(actual_n_train_iters * 1.5) if args.dataset != 'pretraining' else actual_n_train_iters, num_warmup_steps = warmup_steps, last_epoch = -1)

if args.resume_from != '':
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    if not bool(args.scheduler_args.reset_scheduler):
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

lr_callback = callbacks.LambdaCallback(on_batch_end = lambda: scheduler.step())

lr_logger = callbacks.LambdaCallback(on_batch_end = lambda: logger.log('lr', scheduler.get_last_lr()[0]))

if args.debug:
    accelerator.master_print('[🐞DEBUG MODE🐞] Removing ModelCheckpoint ... ')
    checkpoint_callback_iter.actually_save = False
    # checkpoint_callback_best.actually_save = False
else:
    checkpoint_callback_iter.actually_save = bool(args.model_checkpoint.save_model)
    # checkpoint_callback_best.actually_save = bool(args.model_checkpoint.save_model)

callbacks = [
    # checkpoint_callback_best,
    checkpoint_callback_iter,
    lr_callback,
    lr_logger,
]

accelerator.prepare_loggers()
logger = accelerator.get_logger()

trainer = NotALightningTrainer(
    args = args,
    callbacks = callbacks,
    accelerator = accelerator,
    logger = logger,
    scheduler = scheduler,
    state_dict = state_dict if args.resume_from != '' else None,
)

torch.backends.cudnn.benchmark = True
try:
    trainer.fit(model, optimizer, train_dataloader, evaluators = evaluators)
except KeyboardInterrupt:
    accelerator.master_print('::: 🛑🛑🛑 Training Interrupted 🛑🛑🛑 :::')
    accelerator.terminate()
    exit(-1)
finally:
    accelerator.terminate()
