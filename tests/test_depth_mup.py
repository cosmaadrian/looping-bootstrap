import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from models.utils import (configure_depth_mup_parameters, depth_mup_parameter_scales, depth_mup_residual_scale)
from utils import InfDim, InfShape

optim_spec = importlib.util.spec_from_file_location(
    'depth_mup_optim',
    Path(__file__).resolve().parents[1] / 'src' / 'trainers' / 'optim.py',
)
optim_module = importlib.util.module_from_spec(optim_spec)
optim_spec.loader.exec_module(optim_module)
MuAdamW = optim_module.MuAdamW


class DictNamespace(SimpleNamespace):

    def get(self, name, default = None):
        return getattr(self, name, default)


def depth_args(enabled = True, depth_multiplier = 4.0, alpha = 0.5, mean_recurrence = 8):
    return DictNamespace(
        depth_alpha_enabled = enabled,
        depth_multiplier = depth_multiplier,
        depth_alpha_exp = alpha,
        model_args = DictNamespace(mean_recurrence = mean_recurrence),
    )


class DepthMuPTest(unittest.TestCase):

    def test_residual_scale_tracks_sampled_depth(self):
        args = depth_args()

        self.assertAlmostEqual(depth_mup_residual_scale(args, intended_num_loops = 2), 1.0)
        self.assertAlmostEqual(depth_mup_residual_scale(args, intended_num_loops = 8), 0.5)

    def test_disabled_depth_mup_leaves_residuals_unscaled(self):
        args = depth_args(enabled = False)

        self.assertEqual(depth_mup_residual_scale(args, intended_num_loops = 32), 1.0)

    def test_invalid_depth_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'depth_multiplier'):
            depth_mup_residual_scale(depth_args(depth_multiplier = 0), 1)
        with self.assertRaisesRegex(ValueError, 'depth_alpha_exp'):
            depth_mup_residual_scale(depth_args(alpha = 0.25), 1)

    def test_only_recurrent_parameters_receive_depth_lr_scaling(self):
        module = nn.ModuleDict({
            'fixed': nn.Linear(4, 4),
            'recurrent': nn.Linear(4, 4),
        })
        args = depth_args()
        configure_depth_mup_parameters(module, module['recurrent'], args)

        for parameter in module.parameters():
            parameter.infshape = InfShape([InfDim(4, 8), InfDim(4, 8)]) if parameter.ndim == 2 else InfShape([InfDim(4, 8)])

        recurrent_lr_scale, recurrent_eps_scale = depth_mup_parameter_scales(module['recurrent'].weight)
        fixed_lr_scale, fixed_eps_scale = depth_mup_parameter_scales(module['fixed'].weight)

        self.assertAlmostEqual(recurrent_lr_scale, 0.5)
        self.assertAlmostEqual(recurrent_eps_scale, 0.25)
        self.assertAlmostEqual(fixed_lr_scale, 1.0)
        self.assertAlmostEqual(fixed_eps_scale, 0.5)

    def test_muadamw_combines_width_and_depth_scaling(self):
        module = nn.ModuleDict({
            'fixed': nn.Linear(4, 4, bias = False),
            'recurrent': nn.Linear(4, 4, bias = False),
        })
        args = depth_args()
        configure_depth_mup_parameters(module, module['recurrent'], args)

        for parameter in module.parameters():
            parameter.infshape = InfShape([InfDim(4, 8), InfDim(4, 8)])

        optimizer = MuAdamW(
            module.parameters(),
            lr = 0.01,
            eps = 1e-8,
            weight_decay = 0.1,
            fused = False,
        )

        groups_by_parameter = {id(parameter): group for group in optimizer.param_groups for parameter in group['params']}
        recurrent_group = groups_by_parameter[id(module['recurrent'].weight)]
        fixed_group = groups_by_parameter[id(module['fixed'].weight)]

        self.assertAlmostEqual(recurrent_group['lr'], 0.0025)
        self.assertAlmostEqual(recurrent_group['eps'], 2.5e-9)
        self.assertAlmostEqual(fixed_group['lr'], 0.005)
        self.assertAlmostEqual(fixed_group['eps'], 5e-9)


if __name__ == '__main__':
    unittest.main()
