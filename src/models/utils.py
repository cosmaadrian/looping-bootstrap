import torch
import torch.nn as nn
from torch.nn import Linear


def _depth_mup_config(args):
    enabled = bool(args.get('depth_alpha_enabled', False))
    depth_multiplier = float(args.get('depth_multiplier', 1.0))
    depth_alpha_exp = float(args.get('depth_alpha_exp', 1.0))

    if enabled:
        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier must be positive')
        if not 0.5 <= depth_alpha_exp <= 1.0:
            raise ValueError('depth_alpha_exp must be between 0.5 and 1.0')

    return enabled, depth_multiplier, depth_alpha_exp


def depth_mup_residual_scale(args, intended_num_loops):
    """Return the Width–Depth μP residual multiplier for one forward.

    ``depth_multiplier`` describes the configured mean recurrent depth relative
    to the base model. A sampled forward is additionally corrected by its loop
    count relative to ``mean_recurrence``.
    """
    intended_num_loops = int(intended_num_loops)
    if intended_num_loops < 1:
        raise ValueError('intended_num_loops must be at least 1')

    enabled, depth_multiplier, depth_alpha_exp = _depth_mup_config(args)
    if not enabled:
        return 1.0

    mean_recurrence = float(args.model_args.get('mean_recurrence', 1))
    if mean_recurrence <= 0:
        raise ValueError('mean_recurrence must be positive when depth μP is enabled')

    forward_depth_multiplier = depth_multiplier * intended_num_loops / mean_recurrence
    return forward_depth_multiplier**(-depth_alpha_exp)


def configure_depth_mup_parameters(module, depth_scaled_module, args):
    """Attach Width–Depth μP optimizer metadata to model parameters."""
    enabled, depth_multiplier, depth_alpha_exp = _depth_mup_config(args)

    for parameter in module.parameters():
        parameter._depth_mup_enabled = enabled
        parameter._depth_mup_scaled = False
        parameter._depth_mup_multiplier = depth_multiplier
        parameter._depth_mup_alpha = depth_alpha_exp

    if enabled:
        for parameter in depth_scaled_module.parameters():
            parameter._depth_mup_scaled = True


def depth_mup_parameter_scales(parameter):
    """Return Width–Depth μP learning-rate and Adam-epsilon scales."""
    if not getattr(parameter, '_depth_mup_enabled', False):
        return 1.0, 1.0

    width_multiplier = parameter.infshape.width_mult()
    if width_multiplier <= 0:
        raise ValueError('width multiplier must be positive')

    epsilon_scale = 1 / width_multiplier
    learning_rate_scale = 1.0

    if getattr(parameter, '_depth_mup_scaled', False):
        depth_multiplier = parameter._depth_mup_multiplier
        depth_alpha_exp = parameter._depth_mup_alpha
        learning_rate_scale = depth_multiplier**(depth_alpha_exp - 1)
        epsilon_scale *= depth_multiplier**(-depth_alpha_exp)

    return learning_rate_scale, epsilon_scale


def print_mask(mask):
    mask = mask.clone()
    mask = mask * -1
    mask[mask > 0] = 1
    mask = mask.squeeze(1).cpu().numpy()
    mask = mask.astype(int)
    mask = mask.astype(str)
    mask = mask.tolist()
    for m1 in mask:
        for m2 in m1:
            print(''.join(m2))


def print_padding_mask(mask):
    mask = mask.clone()
    mask = mask.cpu().numpy()
    mask = mask.astype(int)
    mask = mask.astype(str)
    mask = mask.tolist()
    for m1 in mask:
        print(''.join(m1))


class MuReadout(Linear):
    '''Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    '''

    def __init__(self, *argz, args, readout_zero_init = False, output_mult = 1.0, **kwargs):
        self.args = args
        self.output_mult = output_mult
        self.readout_zero_init = readout_zero_init
        super().__init__(*argz, **kwargs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.readout_zero_init:
            self.weight.data[:] = 0
            if self.bias is not None:
                self.bias.data[:] = 0
        else:
            super().reset_parameters()

    def width_mult(self):
        return self.args.model_width_multiplier

    def _rescale_parameters(self):
        '''Rescale parameters to convert SP initialization to μP initialization.

        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        '''

        # print("RESCALING PARAMETERS FOR MUREADOUT")
        if hasattr(self, '_has_rescaled_params') and self._has_rescaled_params:
            raise RuntimeError('`_rescale_parameters` has been called once before already. '
                               'Unless you know what you are doing, usually you should not be calling `_rescale_parameters` more than once.\n'
                               'If you called `set_base_shapes` on a model loaded from a checkpoint, '
                               'or just want to re-set the base shapes of an existing model, '
                               'make sure to set the flag `rescale_params=False`.\n'
                               'To bypass this error and *still rescale parameters*, set `self._has_rescaled_params=False` before this call.')
        if self.bias is not None:
            self.bias.data *= self.width_mult()**0.5
        self.weight.data *= self.width_mult()**0.5
        self._has_rescaled_params = True

    def forward(self, x):
        return super().forward(self.output_mult * x / self.width_mult())


class MuSharedReadout(MuReadout):
    '''`MuReadout` with weights shared with an `nn.Embedding` layer.

    Inputs:
        weight: should be weight of an `nn.Embedding` layer
        other inputs are fed to `MuReadout`
    '''

    def __init__(self, weight, bias = True, **kwargs):
        super().__init__(*weight.shape, bias = bias, **kwargs)
        self.weight = weight


def rescale_linear_bias(linear):
    '''Rescale bias in nn.Linear layers to convert SP initialization to μP initialization.

    Warning: This method is NOT idempotent and should be called only once
    unless you know what you are doing.
    '''
    if hasattr(linear, '_has_rescaled_params') and linear._has_rescaled_params:
        raise RuntimeError('`rescale_linear_bias` has been called once before already. Unless you know what you are doing, usually you should not be calling `rescale_linear_bias` more than once.\n'
                           'If you called `set_base_shapes` on a model loaded from a checkpoint, or just want to re-set the base shapes of an existing model, make sure to set the flag `rescale_params=False`.\n'
                           'To bypass this error and *still rescale biases*, set `linear._has_rescaled_params=False` before this call.')
    if linear.bias is None:
        return
    fanin_mult = linear.weight.infshape[1].width_mult()
    linear.bias.data *= fanin_mult**0.5
    linear._has_rescaled_params = True


class RMSNorm(nn.Module):

    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output


def init_method_normal(sigma):

    def init_(tensor):
        return nn.init.normal_(tensor, mean = 0.0, std = sigma)

    return init_


def replace_layernorm_inplace(module, name):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.LayerNorm:
            new_bn = RMSNorm(eps = 1e-5)
            setattr(module, attr_str, new_bn)
    for name, immediate_child_module in module.named_children():
        replace_layernorm_inplace(immediate_child_module, name)
