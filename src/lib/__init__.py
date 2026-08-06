import os
import torch

# assert torch.cuda.is_available(), '::: CUDA is not available! Not even attempting to start training! :::'
if torch.cuda.is_available():
    device_name = 'cuda'
elif torch.backends.mps.is_available():
    device_name = 'mps'
else:
    device_name = 'cpu'

if os.environ.get('RANK', -1) != -1:
    device = torch.device(f'{device_name}:{os.environ["LOCAL_RANK"]}')
else:
    device = torch.device(device_name)

from .trainer import NotALightningTrainer
from ._nomenclature import NOMENCLATURE as nomenclature
from .dataset_extra import AcumenDataset
from .evaluator_extra import AcumenEvaluator

__all__ = ['nomenclature', 'NotALightningTrainer', 'AcumenDataset', 'AcumenEvaluator']
